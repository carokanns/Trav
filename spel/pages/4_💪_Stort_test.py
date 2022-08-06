
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


pref=''   # '../'

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
""" ## Presentation av resultat"""

"Exempel på hur df_resultat skall se ut - alla värden ackumuleras" 

df_resultat = pd.DataFrame({'datum': ['2021-01-01','2021-01-08','2021-01-15', '2021-01-22'],'typ1_pris':[0,300,300,350],'typ6_pris':[0,50,50,400]})
df_resultat.set_index('datum', inplace=True)
# the last row as a list of values
curr=list(df_resultat.iloc[-1,:])
# add a new row 
df_resultat.loc['2021-01-29'] = [curr[0]+0,curr[1]+0]
df_resultat
"motsvarande kan göras för antal 7:or, 6:or and 5:or"

"Grafen"
st.line_chart(df_resultat, width=0, height=0, use_container_width=True)

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


def ta_fram_rad(df_, model, spik_strategi,kelly_strategi, max_cost=300, min_avst=30):
    """ Denna funktion tar fram en rad för typ-modeller (ej meta-modell)
    df nnehåller _en omgång_
    _spik_strategi_: None - inget, '1a' - forcera 1 spik, '2a' - forcera 2 spikar, '1b' - 1 spik endast om klar favorit, '2b' - spikar för endast klara favoriter 
    _kelly_strategi_: None - ingen kelly, 1 - kelly varannan gång om positiv
    """

    veckans_rad = df_.copy()
    veckans_rad['välj'] = False   # inga rader valda ännu
    veckans_rad['spik'] = False   # inga spikar valda ännu
    
    # a) ta ut en häst i varje avd - markera valda i df
    for avd in veckans_rad.avd.unique():
        # max av proba i veckans_rad 
        max_proba = veckans_rad[veckans_rad.avd == avd]['proba'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (veckans_rad.proba == max_proba), 'välj'] = True
    
    # b) leta 1-2 spikar om så begärs - markera valda i df
    if spik_strategi:
        assert spik_strategi in ['1a','1b','2a','2b'], "spik_strategi måste ha något av värdena i listn"
        # Hitta spik-kandidater
        spik2 = veckans_rad.nlargest(2,'proba').index[1] 
        if spik_strategi[0] in ['1','2']:
            spik1 = veckans_rad.nlargest(1,'proba').index[0] 
            if spik_strategi[1] =='b' and veckans_rad.iloc[spik1].streck_avst > min_avst:
                veckans_rad.iloc[spik1].spik = True
                veckans_rad.iloc[spik1].välj = True
            elif spik_strategi == 'a':
                veckans_rad.iloc[spik1].spik = True
                veckans_rad.iloc[spik1].välj = True
                
        if spik_strategi[0] == '2':
            spik2 = veckans_rad.nlargest(2,'proba').index[1] 
            if spik_strategi[1] =='b' and veckans_rad.iloc[spik2].streck_avst > min_avst:
                veckans_rad.iloc[spik2].spik = True
                veckans_rad.iloc[spik2].välj = True
            elif spik_strategi == 'a':
                veckans_rad.iloc[spik2].spik = True
                veckans_rad.iloc[spik2].välj = True
                
    # c) sortera upp i proba-ordning. Om kelly skapa en sortering efter kelly-ordning
    veckans_rad = veckans_rad.sort_values(by=['proba'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)
    if kelly_strategi == '1':
        veckans_kelly = veckans_rad.sort_values(by=['kelly'], ascending=False)
        veckans_kelly = veckans_kelly.reset_index(drop=True)
    
    cost = 0.5 # 1 rad
    while cost < max_cost:
        # d) plocka en och en - först proba sedan ev positiv kelly markera som valda i df
        veckans_rad.loc[veckans_rad.query("välj==False").nlargest(1,'proba').index,'välj'] = True
        # e) avbryt vid 300:-
        cost = compute_total_insats(veckans_rad)
        if  cost >= max_cost:
            break
        if kelly_strategi == '1' and veckans_rad.query("välj==False and kelly > 0").shape[0] > 0:
            veckans_rad.loc[veckans_rad.query("välj==False and kelly > 0").nlargest(1,'kelly').index,'välj'] = True    
        cost = compute_total_insats(veckans_rad)
        
    return veckans_rad, cost

def beräkna_utdelning(df_):
    
    return 0

def rätta_rad(df, datum, df_utdelning ):
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
                femmor_fel += min_tabell.loc[min_tabell.avd==avd_fel].välj.sum()
                
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

def main():
    ## Skapa v75-instans
    v75 = td.v75(pref=pref)
    ## Hämta data från v75
    _ = v75.förbered_data( missing_num=False)  # num hanteras av catboost
    df = v75.test_lägg_till_kolumner()
    ## Skapa modell
    #              name,   ant_hästar  proba,  kelly,  motst_ant,   motst_diff,  ant_favoriter,  only_clear, streck, pref
    test1 = tp.Typ('test1',  True,    True,     False,       0,          False,          0,        False,    True)
    
    st.dataframe(df[["streck_avst",'rel_kr','streck',"rel_rank","h1_samma_bana","h2_samma_bana","h3_samma_bana","h1_samma_kusk","h2_samma_kusk","h3_samma_kusk"]])
    
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
    
    ## TESTA DETTA I IPYNB 
    df_utdelning = pd.read_csv(pref+'/utdelning.csv')
    
    startdatum = '1900-01-01'
    # 0. ta fram startdatum  (datum=startdatum)
    ant_datum = len(df.datum.unique)
    startdatum = df.datum.unique[ant_datum-200]   # ca 3 år tillbaks
    datum =startdatum
    # 1. learn fram till datum
    # 2. ta fram rad för datum, rätta och spara
    veckans_rad, kostnad = ta_fram_rad( df.query("datum==@datum"), test1, '1a', '1', min_avst=30) # inkluderar spik_strategi,kelly_strategi,
    sjuor, sexor, femmor, utdelning = rätta_rad(veckans_rad, datum, df_utdelning )
    # 3. plotta
    
    # 4. startdatum+1
    # 5. gå till 1

if __name__ == "__main__":
    main()
