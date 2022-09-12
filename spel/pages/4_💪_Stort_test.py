# from curses.panel import update_panels
import streamlit as st
import numpy as np
import pandas as pd
from IPython.display import display
# import V75_scraping as vs
# from sklearn.ensemble import RandomForestRegressor
# import datetime
# import sklearn

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)


pref='../'

import pickle

from catboost import CatBoostClassifier, Pool
import concurrent.futures
import time

import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')

import typ_copy as tp
import travdata as td

st.set_page_config(page_title="Stort test av modeller", page_icon="💪")

st.markdown("# 💪 Stort test av modeller")
st.sidebar.header("💪 Stort test")

"""
## Först bas-modellerna
Starta tre år tidigare (2020-08-01).  
TimeSequence 3 folders maybe....   

0. Varje modell har indivduella kolumner samt strategi för att ta fram raden.  
    a. Speca per modell i Typ-klassen.
1. Loop över veckor. Learn fram till aktuell vecka och spara modell.  
    a. Fördel med cv? 
    b. Spara modell med aktuell datum i namnet.
2. Predict nästa vecka (generera rad enl modellens strategi)
3. Rätta, Hämta ev priser och Spara resultat som en df i en csv-fil.
4. plot resultat
5. Repeat för varje modell 

## Gör detsamma för meta-modeller
- Använd sparade modeller och generara stack-data från allt tom aktuell vecka  
- Lär upp meta-modeller på stack-data, använd sparade hyperparms 
- Hur vet vi hur länge meta-modellen skall köra?  
    - Kanske göra ett test innan på ganska stor test-data och spara som hyperparm
- Predict nästa vecka enl strategi för resp meta-modell 
- Rätta och spara resultat plus ev priser
- plot resultat
- Repeat för varje meta-modell

"""
# """ ## Presentation av resultat"""

# "Exempel på hur df_resultat skall se ut - alla värden ackumuleras" 

# df_resultat = pd.DataFrame({'datum': ['2021-01-01','2021-01-08','2021-01-15', '2021-01-22'],'typ1_pris':[0,300,300,350],'typ6_pris':[0,50,50,400]})
# df_resultat.set_index('datum', inplace=True)
# # the last row as a list of values
# curr=list(df_resultat.iloc[-1,:])
# # add a new row 
# df_resultat.loc['2021-01-29'] = [curr[0]+0,curr[1]+0]
# df_resultat
# "motsvarande kan göras för antal 7:or, 6:or and 5:or"

# "Grafen"
# st.line_chart(df_resultat, width=0, height=0, use_container_width=True)

#############################################################################
def välj_rad_orginal(df_meta_predict, max_insats=300):
    veckans_rad = df_meta_predict.copy()
    veckans_rad['välj'] = False   # inga rader valda ännu

    # first of all: select one horse per avd
    for avd in veckans_rad.avd.unique():
        max_pred = veckans_rad[veckans_rad.avd == avd]['meta_predict'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (veckans_rad.meta_predict == max_pred), 'välj'] = True
    # veckans_rad.query("välj==True").to_csv('veckans_basrad.csv')
    veckans_rad = veckans_rad.sort_values(by=['meta_predict'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)
    
    mest_diff=0
    #mest_diff = mesta_diff_per_avd(veckans_rad)
    
    cost = 0.5 # 1 rad
    
    # now select the rest of the horses one by one sorted by meta_predict
    for i, row in veckans_rad.iterrows():
        if row.avd == mest_diff.avd.iloc[0]: 
            continue
        if row.avd == mest_diff.avd.iloc[1]: 
            continue
        # print('i',i)
        veckans_rad.loc[i, 'välj'] = True
        #cost = compute_total_insats(veckans_rad[veckans_rad.välj])
        # print('cost',cost)
        if cost > max_insats:
            # veckans_rad.loc[i, 'välj'] = False
            break
        
    # print('cost', cost_before)
    veckans_rad.sort_values(by=['välj', 'avd'], ascending=[False, True], inplace=True)
    # display(veckans_rad[veckans_rad.välj])
    return veckans_rad


def compute_total_insats(veckans_rad):
    summa = veckans_rad.groupby('avd').avd.count().prod() / 2
    return summa

def rätta_rad1(df, datum, df_utdelning ):
    """
    Räkna ut antal 5:or, 6:or resp. 7:or
    Hämta ev utdelning
    Spara datum, resultat, utdelning och rad-kostnad
    """
    sjuor, sexor, femmor, utdelning = 0,0,0,0
    
    min_tabell = df[['y', 'avd', 'häst', 'rel_rank', 'välj']].copy()
    min_tabell.sort_values(by=['avd', 'y'], ascending=False,inplace=True)

    # 1. om jag har max 7 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 7:
        sjuor=1
        sexor = (min_tabell.groupby('avd').välj.sum()).sum()-7
        # antal femmor
        ant1 = min_tabell.query('avd==1 and välj==True').välj.sum()-1
        ant2 = min_tabell.query('avd==2 and välj==True').välj.sum()-1
        ant3 = min_tabell.query('avd==3 and välj==True').välj.sum()-1
        ant4 = min_tabell.query('avd==4 and välj==True').välj.sum()-1
        ant5 = min_tabell.query('avd==5 and välj==True').välj.sum()-1
        ant6 = min_tabell.query('avd==6 and välj==True').välj.sum()-1
        ant7 = min_tabell.query('avd==7 and välj==True').välj.sum()-1
        femmor = ant1*ant2+ant1*ant2+ant1*ant3+ant1*ant4+ant1*ant5+ant1*ant6+ant1*ant7 +\
                ant2*ant3+ant2*ant4+ant2*ant5+ant2*ant6+ant2*ant7 + \
                ant3*ant4+ant3*ant5+ant3*ant6+ant3*ant7 + \
                ant4*ant5+ant4*ant6+ant4*ant7 + \
                ant5*ant6+ant5*ant7 + \
                ant6*ant7

    # 2. jag har max 6 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 6:
        avd_fel = min_tabell.loc[((min_tabell.välj==False) & (min_tabell.y==1)),'avd'].values[0]
        print(min_tabell.query('avd== @avd_fel').välj.sum())
        sexor = min_tabell.query('avd==@avd_fel').välj.sum()
        # antal femmor
        femmor_fel, femmor_rätt = 0,0
        for avd in range(1,8):
            if avd == avd_fel:
                femmor_fel += min_tabell.loc[min_tabell.avd==avd_fel].välj.sum()  # antal tippade i avd som var fel
                
            femmor_rätt += min_tabell.query('avd==@avd and välj==True').välj.sum()-1
        print(f'femmor_rätt = {femmor_rätt} femmor_fel = {femmor_fel}')    
        femmor = femmor_fel * femmor_rätt

    # 3. jag har max 5 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 5:
        avd_fel = min_tabell.loc[((min_tabell.välj==False) & (min_tabell.y==1)),'avd'].values
        print(avd_fel)
        femmor = min_tabell.loc[min_tabell.avd==avd_fel[0]].välj.sum() * min_tabell.loc[min_tabell.avd==avd_fel[1]].välj.sum()
    
    # 4. utdelning 
    
    return sjuor, sexor, femmor, beräkna_utdelning(datum, sjuor,sexor,femmor, df_utdelning)

def starta_upp(df, back_in_time):
    import datetime
    curr_datix = len(df.datum.unique()) - back_in_time     # ant omg tillbaks
    startdatum = df.datum.unique()[curr_datix]            
    
    st.info(f'Startdatum = {startdatum}')
    
    ### init resutat-tabell
    df_resultat = pd.DataFrame(columns = ['datum',  't1_7', 't1_6', 't1_5', 't1_kostn', 't1_utd', 't1_vinst',
                                                    't2_7', 't2_6', 't2_5', 't2_kostn', 't2_utd', 't2_vinst',
                                                    't3_7', 't3_6', 't3_5', 't3_kostn', 't3_utd', 't3_vinst'])
    
    df_resultat.set_index('datum',drop=True, inplace=True)
    # print(df_resultat.head(0))
    df_resultat.loc[startdatum] = [0, 0, 0, 0, 0, 0, \
                                   0, 0, 0, 0, 0, 0, \
                                   0, 0, 0, 0, 0, 0]
    
    return curr_datix, df.datum.unique(), df_resultat
    

def skapa_data_för_datum(df_, datum):   
    df = df_.copy()
    X = df.query(f'datum < @datum')
    y = X.y
    X = X.drop('y', axis=1)
    X_test = df.query(f'datum > @datum')
    y_test = X_test.y
    X_test = X_test.drop('y', axis=1)
    X_curr = df.query(f'datum == @datum')
    y_curr = X_curr.y
    X_curr = X_curr.drop(['y'], axis=1)
    return X, y, X_test, y_test, X_curr, y_curr
    
    from sklearn.ensemble import RandomForestRegressor as rf

def kelly(proba, streck, odds):  # proba = prob winning, streck i % = streck
    # läs in streck_to_odds.pkl
    import pickle
    with open('rf_streck_odds.pkl', 'rb') as f:
        rf = pickle.load(f)
        
    if odds is None:
        o = rf.predict(streck.copy())
    else:
        o = rf.predict(streck.copy())

    # for each values > 40 in odds set to 1
    o[o > 40] = 1
    return (o*proba - (1-proba))/o

def compute_total_insats(veckans_rad):
    summa = veckans_rad.groupby('avd').avd.count().prod() / 2
    return summa

def beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning):
    datum = datum.strftime('%Y-%m-%d')
        
    min_utdelning = df_utdelning.loc[df_utdelning.datum==datum,['7rätt', '6rätt','5rätt']]
    
    return (min_utdelning['7rätt'] * sjuor + min_utdelning['6rätt'] * sexor + min_utdelning['5rätt'] * femmor).values[0]


def varje_avd_minst_en_häst(veckans_rad ):
    # ta ut en häst i varje avd - markera valda i df
    for avd in veckans_rad.avd.unique():
        # max av proba i veckans_rad 
        max_proba = veckans_rad[veckans_rad.avd == avd]['proba'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (veckans_rad.proba == max_proba), 'välj'] = True
    return veckans_rad

def hitta_spikar(veckans_rad, spikad_avd, spik_strategi, min_avst):
    print('spik_strategi', spik_strategi)
    assert spik_strategi in ['1a','1b','2a','2b'], "spik_strategi måste ha något av värdena i listn"
    # Hitta spik-kandidater
    if spik_strategi[0] in ['1','2']:
        ix_spik1 = veckans_rad.nlargest(1,'proba').index[0]   # ix for largest in dataset
        avd = veckans_rad.loc[ix_spik1,'avd']                 # avd med högsta proba totalt
        no2 = veckans_rad.query("avd==@avd").nlargest(2, 'proba').index[1]  # second i avd ovan
        print(f'hösta proba totalt={veckans_rad.loc[ix_spik1, "proba"]} finns i avd={avd}; no2 i avd={avd} är {veckans_rad.loc[no2,"proba"]}')
        avstånd = veckans_rad.loc[ix_spik1, 'proba'] - veckans_rad.loc[no2, 'proba']
        print('avst', avstånd)
        if (spik_strategi[1] == 'b') and (avstånd > min_avst):  # spik 1 om stort avst
            print('strategi', spik_strategi[1], 'valde spik i avd',avd)
            spikad_avd.append(avd)      # add avd to a list
            
            veckans_rad.loc[ix_spik1,'spik'] = True
            veckans_rad.loc[ix_spik1, 'välj'] = True
        elif spik_strategi[1] == 'a':   # forcerad spik 1
            print('strategi',spik_strategi[1], 'valde spik i avd',avd)
            spikad_avd.append(avd)      # add avd to a list
            veckans_rad.loc[ix_spik1,['spik']]= True
            veckans_rad.loc[ix_spik1, 'välj']= True
            
    if spik_strategi[0] == '2':
        spik2 = veckans_rad.nlargest(2,'proba').index[1] # second in dataset
        avd = veckans_rad.loc[spik2, 'avd']
        no2 = veckans_rad.query("avd==@avd").nlargest(2, 'proba').index[1]  # second in avd
        print(f'näst högsta proba totalt={veckans_rad.loc[spik2, "proba"]} finns i avd={avd}; no2 i avd={avd} är {veckans_rad.loc[no2,"proba"]}')
        avstånd = veckans_rad.loc[spik2, 'proba'] - veckans_rad.loc[no2, 'proba']
        print('avst',avstånd)
        if (spik_strategi[1] =='b') and (avstånd > min_avst):   # spik om stort avstånd till 2:an
            print('strategi', spik_strategi[1], 'valde spik i avd', avd)    
            spikad_avd.append(avd)
            veckans_rad.loc[spik2, 'spik'] = True
            veckans_rad.loc[spik2, 'välj'] = True
        elif spik_strategi[1] == 'a':   # forcerad spik 2
            print('strategi', spik_strategi[1], 'i avd',avd)
            spikad_avd.append(avd)
            veckans_rad.loc[spik2,'spik'] = True
            veckans_rad.loc[spik2,'välj'] = True
    return veckans_rad, spikad_avd

def plocka_en_efter_en(veckans_rad, spikad_avd, kelly_strategi, max_cost=300):
    """_summary_
    Args:
        veckans_rad (_type_): df att fylla
        spikad_avd (_type_): lista med spikade avdelningar
        kelly_strategi (_type_): 1 välj med Kelly varannan gång, annars ingen Kelly
        max_cost (int, optional): Max kostnad. Defaults to 300.

    Returns:
        _type_: df med veckans rad samt kostnad
    """
    cost = 0.5 # 1 rad
    while cost < max_cost:
        # d) plocka en och en - först proba sedan ev positiv kelly markera som valda i df
        curr_index = veckans_rad.query("välj==False and avd not in @spikad_avd").nlargest(1,'proba').index
        veckans_rad.loc[curr_index,'välj'] = True
        # e) avbryt vid 300:-
        cost = compute_total_insats(veckans_rad.query("välj==True"))
        if  cost > max_cost:
            veckans_rad.loc[curr_index, 'välj'] = False  # ta tillbaks den sist spelade
            break
        if kelly_strategi == '1' and veckans_rad.query("välj==False and avd not in @spikad_avd and kelly > 0").shape[0] > 0:
            curr_index = veckans_rad.query("välj==False and avd not in @spikad_avd and kelly > 0").nlargest(1,'kelly').index
            cost = compute_total_insats(veckans_rad.query("välj==True"))
            veckans_rad.loc[curr_index, 'välj'] = True
            if  cost > max_cost:
                veckans_rad.loc[curr_index, 'välj'] = False  # ta tillbaks den sist spelade
                break
    cost = compute_total_insats(veckans_rad.query("välj==True"))
    
    return veckans_rad, cost
    
def ta_fram_rad(veckans_rad_, spik_strategi,kelly_strategi, max_cost=300, min_avst=0.25):
    """ Denna funktion tar fram en rad för typ-modeller (ej meta-modell)
    df nnehåller _en omgång_
    _spik_strategi_: None - inget, '1a' - forcera 1 spik, '2a' - forcera 2 spikar, '1b' - 1 spik endast om klar favorit, '2b' - spikar för endast klara favoriter 
    _kelly_strategi_: None - ingen kelly, 1 - kelly varannan gång om positiv
    """
    veckans_rad = veckans_rad_.copy()
    veckans_rad['välj'] = False   # inga rader valda ännu
    veckans_rad['spik'] = False   # inga spikar valda ännu
    
    veckans_rad = varje_avd_minst_en_häst(veckans_rad )
    
    # b) leta 1-2 spikar om så begärs - markera valda i df
    spikad_avd = []
    if spik_strategi:
        veckans_rad, spikad_avd = hitta_spikar(veckans_rad, spikad_avd, spik_strategi, min_avst)
        
    # c) sortera upp i proba-ordning. Om kelly skapa en sortering efter kelly-ordning
    veckans_rad = veckans_rad.sort_values(by=['proba'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)
    if kelly_strategi == '1':
        veckans_kelly = veckans_rad.sort_values(by=['kelly'], ascending=False)
        veckans_kelly = veckans_kelly.reset_index(drop=True)
    
    ## plocka en efter en tills kostnaden är för stor
    # return veckans_rad, cost
    return plocka_en_efter_en(veckans_rad, spikad_avd, kelly_strategi, max_cost)
      
def rätta_rad2(df, datum, df_utdelning ):
    """
    Räkna ut antal 5:or, 6:or resp. 7:or
    Hämta ev utdelning
    Spara datum, resultat, utdelning och rad-kostnad
    """
    sjuor, sexor, femmor, utdelning = 0,0,0,0
    
    min_tabell = df[['y', 'avd', 'häst', 'rel_rank', 'välj']].copy()
    min_tabell.sort_values(by=['avd', 'y'], ascending=False,inplace=True)

    # 1. om jag har max 7 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 7:
        sjuor=1
        sexor = (min_tabell.groupby('avd').välj.sum()).sum()-7
        # antal femmor
        ant1 = min_tabell.query('avd==1 and välj==True').välj.sum()-1
        ant2 = min_tabell.query('avd==2 and välj==True').välj.sum()-1
        ant3 = min_tabell.query('avd==3 and välj==True').välj.sum()-1
        ant4 = min_tabell.query('avd==4 and välj==True').välj.sum()-1
        ant5 = min_tabell.query('avd==5 and välj==True').välj.sum()-1
        ant6 = min_tabell.query('avd==6 and välj==True').välj.sum()-1
        ant7 = min_tabell.query('avd==7 and välj==True').välj.sum()-1
        femmor = ant1*ant2+ant1*ant2+ant1*ant3+ant1*ant4+ant1*ant5+ant1*ant6+ant1*ant7 +\
                ant2*ant3+ant2*ant4+ant2*ant5+ant2*ant6+ant2*ant7 + \
                ant3*ant4+ant3*ant5+ant3*ant6+ant3*ant7 + \
                ant4*ant5+ant4*ant6+ant4*ant7 + \
                ant5*ant6+ant5*ant7 + \
                ant6*ant7

    # 2. om jag har max 6 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 6:
        avd_fel = min_tabell.loc[((min_tabell.välj==False) & (min_tabell.y==1)),'avd'].values[0]
        print(min_tabell.query('avd== @avd_fel').välj.sum())
        sexor = min_tabell.query('avd==@avd_fel').välj.sum()
        # antal femmor
        femmor_fel, femmor_rätt = 0,0
        for avd in range(1,8):
            if avd == avd_fel:
                femmor_fel += min_tabell.loc[min_tabell.avd==avd_fel].välj.sum()
                
            femmor_rätt += min_tabell.query('avd==@avd and välj==True').välj.sum()-1
        print(f'femmor_rätt = {femmor_rätt} femmor_fel = {femmor_fel}')    
        femmor = femmor_fel * femmor_rätt

    # 3. om jag har max 5 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 5:
        avd_fel = min_tabell.loc[((min_tabell.välj==False) & (min_tabell.y==1)),'avd'].values
        femmor = min_tabell.loc[min_tabell.avd==avd_fel[0]].välj.sum() * min_tabell.loc[min_tabell.avd==avd_fel[1]].välj.sum()
    
    return sjuor, sexor, femmor, beräkna_utdelning(datum, sjuor,sexor,femmor, df_utdelning)

def kör(df,test1, test2, typ6):   
    df_utdelning = pd.read_csv('utdelning.csv')
    back_in_time = 300  # antal dagar att gå tillbaka i tiden för backtesting
    
    curr_datum_ix, datumar, df_resultat = starta_upp(df, back_in_time)
    placeholderdat = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    for i in range(back_in_time):
        datum = datumar[curr_datum_ix]
        placeholderdat.empty()
        if i >0:
            placeholderdat.info(f'Aktuell datum: {  datum} {"        "} \nant_omgångar spelade: {  i}')
            
        # 1. learn fram till datum
        #### 1a. skapa train och test dataframes samt aktuell omgång som vi predictar - returnerar df_train, df_test, df_curr
        X,y,X_test,y_test, X_curr, y_curr  = skapa_data_för_datum(df, datum)
        if X.empty or X_test.empty or X_curr.empty:
            break
        
        print(f'learned fram till {datum}')
        veckans_rad1 = X_curr[['datum','avd','häst','bana','kusk','streck','streck_avst','rel_rank']].copy()
        veckans_rad1['y'] = y_curr
        veckans_rad2 = X_curr[['datum','avd','häst','bana','kusk','streck','streck_avst','rel_rank']].copy()
        veckans_rad2['y'] = y_curr
        veckans_rad3 = X_curr[['datum','avd','häst','bana','kusk','streck','streck_avst','rel_rank']].copy()
        veckans_rad3['y'] = y_curr
        
        #------------- Learn test-modeller-------------------------------------------
        params = {"depth": 2, "l2_leaf_reg": 3,"iterations": 500, "learning_rate": 0.008}
        model1 = test1.learn(X, y, X_test_= X_test, y_test=y_test,save=True, params=params.copy())
        model2 = test2.learn(X, y, X_test_= X_test, y_test=y_test,save=True, params=params.copy())
        with open('optimera/params_'+typ6.name+'.json', 'r') as f:
                    params6 = json.load(f)
                    params6 = params6['params']
        model3 = typ6.learn(X, y, X_test_= X_test, y_test=y_test, save=False,  params=params6)   # don't save. Use model3.predict() instead
        print('X_shape',X.shape)
        print('X_curr.shape',X_curr.shape, 'X_test.shape',X_test.shape)
        
        #------------- predict och skapa Kelly ----------------------------------------
        veckans_rad1['proba'] = test1.predict(X_curr)
        veckans_rad1['kelly'] = kelly(veckans_rad1.proba.copy(), veckans_rad1[['streck']].copy(), None)
        veckans_rad2['proba'] = test2.predict(X_curr)
        veckans_rad2['kelly'] = kelly(veckans_rad2.proba.copy(), veckans_rad2[['streck']].copy(), None)
        
        veckans_rad3['proba'] = typ6.predict(X_curr, model=model3)
        veckans_rad3['kelly'] = kelly(veckans_rad3.proba.copy(), veckans_rad3[['streck']].copy(), None)

        # ta fram rad för datum, rätta och spara
        # inkluderar spik_strategi,kelly_strategi,
        veckans_rad1, kostnad1 = ta_fram_rad(veckans_rad1, '2b', None, min_avst=0.4)
        veckans_rad2, kostnad2 = ta_fram_rad(veckans_rad2, None, '1', min_avst=0.4)
        veckans_rad3, kostnad3 = ta_fram_rad(veckans_rad3, None, None, min_avst=0.4)

        sjuor1, sexor1, femmor1, utdelning1 = rätta_rad2(veckans_rad1, datum, df_utdelning)
        sjuor2, sexor2, femmor2, utdelning2 = rätta_rad2(veckans_rad2, datum, df_utdelning)
        sjuor3, sexor3, femmor3, utdelning3 = rätta_rad2(veckans_rad3, datum, df_utdelning)
        # print('kostnad',kostnad, 'utdelning', utdelning)
        last_row = df_resultat.iloc[-1] + [sjuor1, sexor1,femmor1, kostnad1, utdelning1, utdelning1-kostnad1,
                                        sjuor2, sexor2,femmor2, kostnad2, utdelning2, utdelning2-kostnad2,
                                        sjuor3, sexor3,femmor3, kostnad3, utdelning3, utdelning3-kostnad3 ]
        df_resultat.loc[datum] = last_row
        
        df_resultat.to_csv('backtest_resultat.csv', index=False)
        
        # 3. plotta
        graf_data = df_resultat.copy()
        graf_data.index = pd.to_datetime(graf_data.index, format="%Y-%m-%d")
        placeholder1.empty()
        placeholder2.empty()
        placeholder3.empty()
        # df_resultat.loc['2019-08-01'] = [0, 0, 8,300,200, -100]
        # df_resultat.loc['2019-08-08'] = [0, 0, 0,300,0, -400]
        # df_resultat.loc['2019-08-15'] = [0, 1, 2,300,600, -100]
        # df_resultat.loc['2019-08-22'] = [1, 2, 8,300,1000, 900] 
        placeholder1.line_chart(graf_data[['t1_vinst','t2_vinst','t3_vinst']], width=0, height=0, use_container_width=True)
        placeholder2.line_chart(graf_data[['t1_7','t2_7', 't3_7']], width=0, height=0, use_container_width=True)
        # st.write(df_resultat.plot(kind='line',  y='t1_vinst', rot=45, legend=True, figsize=(20,10)))
        placeholder3.dataframe(df_resultat.tail(10))
        curr_datum_ix += 1 
        
    return veckans_rad1, veckans_rad2, veckans_rad3           
        
import json
def main():
    ## Skapa v75-instans
    v75 = td.v75(pref='')
    ## Hämta data från v75
    _ = v75.förbered_data( missing_num=False)  # num hanteras av catboost
    df = v75.test_lägg_till_kolumner()
    
    ###############################################################
    # Några idéer på nya kolumner:
    #  -   ❌ streck/sum(streck för avd) - fungerar inte bra. Nästan alla sum == 100 per avd
    #  a - ✔️ plats(streck)/ant_hästar_i_avd (antal startande hästar i avd)                    
    #  b - ❌ pris / tot_pris_i_avd - går inte att använda ju! pris är ju loppets 1.pris - samma för all i loppet 
    #  c - ✔️ kr / tot_kr_i_avd     rel_kr
    #  d - ✔️ Avståndet till ettan (streck)
    #  e - ✔️ hx_bana samma som bana
    #  f - ✔️ hx_kusk samma som kusk
    #  META
    #  g - meta får annan input än bara typ-resultat, tex, plats i avd, ettans avstånd till tvåan
    #
    # Några idéer på regler för att selektera raden:
    #  1 - 1 avd med favorit som spik
    #  2 - 2 avd med var sin favorit som spik
    #  3 - Endast solklara favoriter - beror på avståndet till tvåan
    #  4 - Inga forcerade favoriter
    #  5 - Välj den högsta positiva Kelly efter vald proba - om vartannat
    ###############################################################
    #  Minska max-kostnad för en rad  - 384 är för mycket
    ###############################################################
    # Använd typ9 som grund-modell och lägg till resp ta bort kolumner per test-typ
    # genererara alla kolumner som vi sedan selekterar från
    # Namnge modeller efter konfig samt selektering tex typ_abcdef235
    
    #-------------- skapa test-modeller
    #              name,   ant_hästar  proba,  kelly,   motst_ant,  motst_diff,  ant_favoriter,  only_clear, streck, test, pref
    test1 = tp.Typ('test1',  True,    True,     False,       0,        False,          0,           False,    True,  True, pref='')
    test2 = tp.Typ('test2',  True,    True,     False,       0,        False,          0,           False,    False, True, pref='')
    typ6 = tp.Typ('typ6',    True,    True,     False,       0,        False,          0,           False,    True,  False, pref='')
    
    # st.dataframe(df)
    
    if st.button('kör'):
        veckans_rad1,veckans_rad2,veckans_rad3 = kör(df, test1, test2, typ6)
    


if __name__ == "__main__":
    main()