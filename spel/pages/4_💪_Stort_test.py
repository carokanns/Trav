import pickle
import streamlit as st
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier, Pool
import sys
import typ_copy as tp
import travdata as td
import json
import concurrent.futures
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
pref = ''
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')

import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
plt.style.use('ggplot')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

st.set_page_config(page_title="Stort test av modeller", page_icon="💪")

st.markdown("# 💪 Stort test av modeller")
st.sidebar.header("💪 Stort test")

exp = st.expander('Beskrivning av testet')
exp.write("""
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

## Gör detsamma för meta-modeller (rf, knn, ridge)
- Använd sparade typ-modeller och generara stack-data från allt tom aktuell vecka  
- vad skall tas med i stacken?
    1. Alla predict från typ-modellerna
    2. Kelly-data
    3. bana? kusk? distans? streck för någon meta? 
- Lär upp meta-modeller på stack-data, använd sparade hyperparms 
- Hur vet vi hur länge meta-modellen skall köra?  
    - Kanske göra ett test innan på ganska stor test-data och spara som hyperparm
- Predict nästa vecka enl strategi för resp meta-modell 

- Rätta och spara resultat plus ev priser
- plot resultat
- Repeat för varje meta-modell

"""
)


# def välj_rad_orginal(df_meta_predict, max_insats=300):
#     veckans_rad = df_meta_predict.copy()
#     veckans_rad['välj'] = False   # inga rader valda ännu

#     # first of all: select one horse per avd
#     for avd in veckans_rad.avd.unique():
#         max_pred = veckans_rad[veckans_rad.avd == avd]['meta_predict'].max()
#         veckans_rad.loc[(veckans_rad.avd == avd) & (
#             veckans_rad.meta_predict == max_pred), 'välj'] = True
#     # veckans_rad.query("välj==True").to_csv('veckans_basrad.csv')
#     veckans_rad = veckans_rad.sort_values(by=['meta_predict'], ascending=False)
#     veckans_rad = veckans_rad.reset_index(drop=True)

#     mest_diff = 0
#     #mest_diff = mesta_diff_per_avd(veckans_rad)

#     cost = 0.5  # 1 rad

#     # now select the rest of the horses one by one sorted by meta_predict
#     for i, row in veckans_rad.iterrows():
#         if row.avd == mest_diff.avd.iloc[0]:
#             continue
#         if row.avd == mest_diff.avd.iloc[1]:
#             continue
#         # print('i',i)
#         veckans_rad.loc[i, 'välj'] = True
#         #cost = compute_total_insats(veckans_rad[veckans_rad.välj])
#         # print('cost',cost)
#         if cost > max_insats:
#             # veckans_rad.loc[i, 'välj'] = False
#             break

#     # print('cost', cost_before)
#     veckans_rad.sort_values(by=['välj', 'avd'], ascending=[
#                             False, True], inplace=True)
#     # display(veckans_rad[veckans_rad.välj])
#     return veckans_rad

#%%
def compute_total_insats(veckans_rad):
    summa = veckans_rad.groupby('avd').avd.count().prod() / 2
    return summa


def starta_upp(df, start_ix=220):
    import datetime
    
    startdatum = df.datum.unique()[start_ix]

    st.info(f'Startdatum = {startdatum}')

    # init resutat-tabell
    df_resultat = pd.DataFrame(columns=['datum', 't1_vinst', 't2_vinst', 't3_vinst', 't4_vinst',
                                                 't1_utd',   't2_utd',   't3_utd',   't4_utd', 
                                                 't1_kostn', 't2_kostn', 't3_kostn', 't4_kostn',
                                                 't1_7', 't2_7', 't3_7', 't4_7', 
                                                 't1_6', 't2_6', 't3_6', 't4_6', 
                                                 't1_5', 't2_5', 't3_5', 't4_5' 
                                         ])

    df_resultat.set_index('datum', drop=True, inplace=True)
    # print(df_resultat.head(0))
    df_resultat.loc[startdatum] = [0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0]
    # for all columns except datum make int
    for col in df_resultat.columns:
        if col != 'datum':
            df_resultat[col] = df_resultat[col].astype(int)

    return  df.datum.unique(), df_resultat

#%%
def skapa_data_för_datum(df_, curr_datum_ix, frac=0.5):
    df = df_.copy()
    datumar = df.datum.unique()
    curr_datum = datumar[curr_datum_ix]
    base_datum_ix = int(len(datumar[:curr_datum_ix]) * frac)  #base models
    
    base_datum = datumar[base_datum_ix]
    X_train = df.query(f'datum < @base_datum')
    y_train = X_train.y
    X_train = X_train.drop('y', axis=1)
    
    X_meta = df.query(f'datum >= @base_datum and datum < @curr_datum')
    y_meta = X_meta.y
    X_meta = X_meta.drop('y', axis=1)
    
    X_curr = df.query(f'datum == @curr_datum')
    y_curr = X_curr.y
    X_curr = X_curr.drop(['y'], axis=1)
    
    return X_train, y_train, X_meta, y_meta, X_curr, y_curr


    # from sklearn.ensemble import RandomForestRegressor as rf

#%%
def kelly(proba_, streck, odds):  # proba = prob winning, streck i % = streck
    proba = proba_.copy()
    # läs in streck_to_odds.pkl
    # import pickle
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

#%%
def beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning):
    datum = datum.strftime('%Y-%m-%d')

    min_utdelning = df_utdelning.loc[df_utdelning.datum == datum, [
        '7rätt', '6rätt', '5rätt']]

    return (min_utdelning['7rätt'] * sjuor + min_utdelning['6rätt'] * sexor + min_utdelning['5rätt'] * femmor).values[0]


def varje_avd_minst_en_häst(veckans_rad):
    # ta ut en häst i varje avd - markera valda i df
    for avd in veckans_rad.avd.unique():
        # max av proba i veckans_rad
        max_proba = veckans_rad[veckans_rad.avd == avd]['proba'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (
            veckans_rad.proba == max_proba), 'välj'] = True
    return veckans_rad

#%%
def hitta_spikar(veckans_rad, spikad_avd, spik_strategi, min_avst):
    print('spik_strategi', spik_strategi)
    assert spik_strategi in [
        '1a', '1b', '2a', '2b'], "spik_strategi måste ha något av värdena i listan"
    # Hitta spik-kandidater
    if spik_strategi[0] in ['1', '2']:
        ix_spik1 = veckans_rad.nlargest(1, 'proba').index[0]   # ix for largest in dataset
        # avd med högsta proba totalt
        avd = veckans_rad.loc[ix_spik1, 'avd']
        no2 = veckans_rad.query("avd==@avd").nlargest(2,'proba').index[1]  # second i avd ovan
        # print(
        #     f'hösta proba totalt={veckans_rad.loc[ix_spik1, "proba"]} finns i avd={avd}; no2 i avd={avd} är {veckans_rad.loc[no2,"proba"]}')
        avstånd = veckans_rad.loc[ix_spik1, 'proba'] - \
            veckans_rad.loc[no2, 'proba']
    
        if (spik_strategi[1] == 'b') and (avstånd > min_avst):  # spik 1 om stort avst
            print('strategi', spik_strategi, 'valde spik i avd', avd, 'då avst', avstånd,'> min',min_avst )
            spikad_avd.append(avd)      # add avd to a list

            veckans_rad.loc[ix_spik1, 'spik'] = True
            veckans_rad.loc[ix_spik1, 'välj'] = True
        elif spik_strategi[1] == 'a':   # forcerad spik 1
            print('strategi', spik_strategi, 'valde forcerad spik i avd', avd)
            spikad_avd.append(avd)      # add avd to a list
            veckans_rad.loc[ix_spik1, ['spik']] = True
            veckans_rad.loc[ix_spik1, 'välj'] = True
        else:
            print('strategi', spik_strategi, 'inte vald', 'då avst', avstånd,'< min',min_avst )    

    if spik_strategi[0] == '2':
        spik2 = veckans_rad.nlargest(2, 'proba').index[1]  # second in dataset
        avd = veckans_rad.loc[spik2, 'avd']
        no2 = veckans_rad.query("avd==@avd").nlargest(2,'proba').index[1]  # second in avd
        # print(
        #     f'näst högsta proba totalt={veckans_rad.loc[spik2, "proba"]} finns i avd={avd}; no2 i avd={avd} är {veckans_rad.loc[no2,"proba"]}')
        avstånd = veckans_rad.loc[spik2, 'proba'] - \
                  veckans_rad.loc[no2, 'proba']
    
        # spik om stort avstånd till 2:an
        if (spik_strategi[1] == 'b') and (avstånd > min_avst):
            print('strategi', spik_strategi, 'valde spik i avd', avd, 'då avst', avstånd,'> min',min_avst )
            spikad_avd.append(avd)
            veckans_rad.loc[spik2, 'spik'] = True
            veckans_rad.loc[spik2, 'välj'] = True
        elif spik_strategi[1] == 'a':   # forcerad spik 2
            print('strategi', spik_strategi, 'valde forcerad spik i avd', avd)
            spikad_avd.append(avd)
            veckans_rad.loc[spik2, 'spik'] = True
            veckans_rad.loc[spik2, 'välj'] = True
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
    cost = 0.5  # 1 rad
    while cost < max_cost:
        # d) plocka en och en - först proba sedan ev positiv kelly markera som valda i df
        curr_index = veckans_rad.query(
            "välj==False and avd not in @spikad_avd").nlargest(1, 'proba').index
        veckans_rad.loc[curr_index, 'välj'] = True
        # e) avbryt vid 300:-
        cost = compute_total_insats(veckans_rad.query("välj==True"))
        if cost > max_cost:
            # ta tillbaks den sist spelade
            veckans_rad.loc[curr_index, 'välj'] = False
            break
        
        if kelly_strategi == 1:
            pass
            # veckans_kelly = veckans_rad.query(
            #     "välj==False and kelly > 0 and avd not in @spikad_avd ")
            # veckans_kelly = veckans_kelly.sort_values(
            #     by=['kelly'], ascending=False)
            # try:
            #     True
            #     # print('kelly')
            # except:
            #     print('no kelly', veckans_kelly.shape)

            # if veckans_kelly.iloc[0]['kelly'] > 0:
            #     curr_index = veckans_kelly.iloc[0].name
            #     veckans_rad.loc[curr_index, 'välj'] = True
            #     veckans_rad.loc[curr_index, 'kelly_val'] = True
            #     cost = compute_total_insats(veckans_rad.query("välj==True"))

            # if cost > max_cost:
            #     # ta tillbaks den sist spelade
            #     veckans_rad.loc[curr_index, 'välj'] = False
            #     break
    cost = compute_total_insats(veckans_rad.query("välj==True"))

    return veckans_rad, cost


def ta_fram_rad(veckans_rad_, spik_strategi, kelly_strategi, max_cost=300, min_avst=0.09):
    """ Denna funktion tar fram en rad för typ-modeller (ej meta-modell)
    df nnehåller _en omgång_
    _spik_strategi_: None - inget, '1a' - forcera 1 spik, '2a' - forcera 2 spikar, '1b' - 1 spik endast om klar favorit, '2b' - spikar för endast klara favoriter 
    _kelly_strategi_: None - ingen kelly, 1 - kelly varannan gång om positiv
    """
    assert False, 'Denna skall inte köras'
    veckans_rad = veckans_rad_.copy()
    # veckans_rad['kelly_val'] = False
    veckans_rad['välj'] = False   # inga rader valda ännu
    veckans_rad['spik'] = False   # inga spikar valda ännu

    veckans_rad = varje_avd_minst_en_häst(veckans_rad)

    # b) leta 1-2 spikar om så begärs - markera valda i df
    spikad_avd = []
    if spik_strategi:
        veckans_rad, spikad_avd = hitta_spikar(
            veckans_rad, spikad_avd, spik_strategi, min_avst)

    # c) sortera upp i proba-ordning. Om kelly skapa en sortering efter kelly-ordning
    veckans_rad = veckans_rad.sort_values(by=['proba'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)

    # plocka en efter en tills kostnaden är för stor
    # return veckans_rad, cost
    return plocka_en_efter_en(veckans_rad, spikad_avd, kelly_strategi, max_cost)


def ta_fram_meta_rad(veckans_rad_, meta_modeller, spik_strategi, kelly_strategi, max_cost=300, min_avst=0.09):
    """ Denna funktion tar fram en rad för meta-modellerna genom mean value
    df nnehåller _en omgång_
    _spik_strategi_: None - inget, '1a' - forcera 1 spik, '2a' - forcera 2 spikar, '1b' - 1 spik endast om klar favorit, '2b' - spikar för endast klara favoriter 
    _kelly_strategi_: None - ingen kelly, 1 - kelly varannan gång om positiv
    """
    veckans_rad = veckans_rad_.copy()
    # veckans_rad['kelly_val'] = False
    veckans_rad['välj'] = False   # inga rader valda ännu
    veckans_rad['spik'] = False   # inga spikar valda ännu

    veckans_rad['proba'] = 0
    # veckans_rad['kelly'] = 0

    ### Här taar vi medelvärdet av predict för meta-modellerna ######
    for enum, key in enumerate(meta_modeller.keys()):
        veckans_rad['proba'] += veckans_rad[key]
        # veckans_rad['kelly'] += veckans_rad['kelly'+enum]

    veckans_rad['proba'] /= len(meta_modeller)
    # veckans_rad['kelly'] /= len(meta_modeller)
    #################################################################

    veckans_rad = varje_avd_minst_en_häst(veckans_rad)

    # b) leta 1-2 spikar om så begärs - markera valda i df
    spikad_avd = []
    if spik_strategi:
        veckans_rad, spikad_avd = hitta_spikar(veckans_rad, spikad_avd, spik_strategi, min_avst)
    else:
        print('ingen spik-strategi')    

    # c) sortera upp i proba-ordning. Om kelly skapa en sortering efter kelly-ordning
    veckans_rad = veckans_rad.sort_values(by=['proba'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)

    # plocka en efter en tills kostnaden är för stor
    # return veckans_rad, cost
    return plocka_en_efter_en(veckans_rad, spikad_avd, kelly_strategi, max_cost)


def rätta_rad(df, datum, df_utdelning):
    """
    Räkna ut antal 5:or, 6:or resp. 7:or
    Hämta ev utdelning
    Spara datum, resultat, utdelning och rad-kostnad
    """
    sjuor, sexor, femmor, utdelning = 0, 0, 0, 0

    min_tabell = df[['y', 'avd', 'häst', 'rel_rank', 'välj']].copy()
    min_tabell.sort_values(by=['avd', 'y'], ascending=False, inplace=True)
    
    print('Antal rätt', min_tabell.query('välj==True and y==1').y.sum())
   
    # 1. om jag har max 7 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 7:
        sjuor = 1
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
        avd_fel = min_tabell.loc[((min_tabell.välj == False) & (
            min_tabell.y == 1)), 'avd'].values[0]
        # print(min_tabell.query('avd== @avd_fel').välj.sum())
        sexor = min_tabell.query('avd==@avd_fel').välj.sum()
        # antal femmor
        femmor_fel, femmor_rätt = 0, 0
        for avd in range(1, 8):
            if avd == avd_fel:
                femmor_fel += min_tabell.loc[min_tabell.avd == avd_fel].välj.sum()

            femmor_rätt += min_tabell.query('avd==@avd and välj==True').välj.sum()-1
        # print(f'femmor_rätt = {femmor_rätt} femmor_fel = {femmor_fel}')
        femmor = femmor_fel * femmor_rätt

    # 3. om jag har max 5 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 5:
        avd_fel = min_tabell.loc[((min_tabell.välj == False) & (
            min_tabell.y == 1)), 'avd'].values
        femmor = min_tabell.loc[min_tabell.avd == avd_fel[0]].välj.sum() * min_tabell.loc[min_tabell.avd == avd_fel[1]].välj.sum()

    return sjuor, sexor, femmor, beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning)


# def initiera_veckans_rader(X_curr, y_curr, antal_rader):
#     # ---------- initier veckans rad med aktuell omgång ----------------------
#     veckans_rader = []
#     for i in range(antal_rader):
#         veckans_rader.append(X_curr[['datum', 'avd', 'häst', 'bana',
#                                      'kusk', 'streck', 'streck_avst', 'rel_rank']].copy())
#         veckans_rader[i]['y'] = y_curr
#         veckans_rader[i]['välj'] = False

#     return veckans_rader

#%%


def skapa_stack_data(model, name, X_meta, stack_data):
    """Skapa stack_data inklusive Kelly"""
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    this_proba = model.predict(X_meta)
    # print(f'X_meta.shape = {X_meta.shape} this_proba.shape={this_proba.shape}')

    # Bygg up meta-kolumnerna (proba och Kelly) för denns typ
    nr = name[3:]
    stack_data['proba'+nr] = this_proba
    # stack_data['kelly'+nr] = kelly(this_proba, X_meta[['streck']], None)
    # print(f'stack_data.shape = {stack_data.shape}')
    return stack_data


def prepare_stack_data(stack_data_):
    # """Hantera missing values, NaN, etc för meta-modellerna"""

    assert 'y' in stack_data_.columns, 'y is missing in stack_data'
    stack_data = stack_data_.copy()
    stack_data.y = stack_data.y.astype(int)

    #""" rensa bort features som inte ska användas """
    # stack_data.drop(['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
    #             'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'], axis=1, inplace=True)

    #""" Fyll i saknade numeriska värden med 0 """
    numericals = stack_data.drop('y', axis=1).select_dtypes(
        exclude=['object']).columns
    stack_data[numericals] = stack_data[numericals].fillna(0)

    #""" Fyll i saknade kategoriska värden med 'missing' """
    categoricals = stack_data.select_dtypes(include=['object']).columns
    stack_data[categoricals] = stack_data[categoricals].fillna('missing')

    # """ Hantera high cardinality """
    # cardinality_list=['häst','kusk','h1_kusk','h2_kusk','h3_kusk','h4_kusk','h5_kusk']

    #""" Target encoding"""
    target_encode_list = ['bana', 'häst', 'kusk', 'kön', 'h1_kusk', 'h1_bana', 'h2_kusk', 'h2_bana',
                          'h3_kusk', 'h3_bana', 'h4_kusk', 'h4_bana', 'h5_kusk', 'h5_bana']

    y = stack_data['y']
    enc = TargetEncoder(cols=target_encode_list,
                        min_samples_leaf=20, smoothing=10).fit(stack_data, y)
    stack_data = enc.transform(stack_data)

    return stack_data, enc


def first_learn_modeller(modeller, X_train, y_train, X_meta, y_meta):
    ############################################################################################################
    #                        Här görs en första learn av modeller och sedan skapas stack_data
    #                        - Learn modeller på X,y
    #                        - Ha en egen skapa_stack_funktion (som också används längre ner)
    #                           - Skapa stack_data med predict X_meta med nya modellerna
    #                           - Spara även X_meta, y_meta i stack_data
    #                           - Spara även Kelly i stack_data
    ############################################################################################################
    stack_data = X_meta.copy()
    stack_data['y'] = y_meta
    assert 'y' in stack_data.columns, '1. y is missing in stack_data'
    for model in modeller:
        name = model.name
        print(f'first Learn {name} {X_train.datum.min()} -{X_train.datum.max()}')
        
        model.learn(X_train, y_train, params=None, save=True)

        
        stack_data = skapa_stack_data(model, name, X_meta, stack_data)
        assert 'y' in stack_data.columns, '2. y is missing in stack_data'

    assert 'y' in stack_data.columns, '3. y is missing in stack_data'
    stack_data, enc = prepare_stack_data(stack_data)

    return stack_data

############################################################################################################
#                       Här gör vi learn av meta_modeller på stack_data
############################################################################################################

def learn_meta_models(meta_modeller, stack_data, save=True):
    # global ENC
    # print('Learn meta_modeller på stack_data')
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    stack_data.to_csv('stack_data_före_drop.csv', index=False)

    X_meta, ENC = prepare_stack_data(stack_data)
    X_meta = stack_data.drop(['datum', 'avd', 'y'], axis=1)
    y_meta = stack_data.y
    X_meta.to_csv('X_meta_Learn.csv', index=False)
    print(f'Learn meta {stack_data.datum.min()} - {stack_data.datum.max()}')
    # print(X_meta.columns)
    for key, items in meta_modeller.items():
        meta_model = items['model']

        items['model'] = meta_model.fit(X_meta, y_meta)
        meta_modeller[key] = items

        if save:
            with open(pref+'modeller/'+key+'.model', 'wb') as f:
                pickle.dump(meta_model, f)

    # print('Done Learn meta_modeller på stack_data')
    return meta_modeller


def final_learn_modeller(modeller, X, y, X_meta, y_meta):
    print('Final_learn modeller ')
    assert False, 'Detta skall inte köras'
    X_train = pd.concat([X, X_meta])
    y_train = pd.concat([y, y_meta])
    print(
        f'X_train.shape = {X_train.shape} X.shape = {X.shape}, X_meta.shape = {X_meta.shape}')
    # print(f'X_train.columns = {X_train.columns}')
    for model in modeller:
        model.learn(X_train, y_train, save=True)
    print('Done Final_learn modeller')
    return modeller


def initiera_veckans_rader(X_curr, y_curr, antal_rader):
    # ---------- initier veckans rad med aktuell omgång ----------------------
    veckans_rader = []
    for i in range(antal_rader):
        veckans_rader.append(X_curr[['datum', 'avd', 'häst', 'bana',
                                     'kusk', 'streck', 'streck_avst', 'rel_rank']].copy())
        veckans_rader[i]['y'] = y_curr
        veckans_rader[i]['välj'] = False

    return veckans_rader


def predict_curr_omgang(modeller, meta_modeller, X_curr, y_curr):
    """
        Här tas meta_modellernas prediktioner fram
        - modeller     : predict X_curr och använd skapa_stack_data funktionen
        - meta_modeller: predict på stack_datat och fyll i veckans rader
        veckans_rader innehåller nu prediktioner från alla meta_modeller plus X_curr, y_curr
    """
    stack_data = X_curr.copy()
    stack_data['y'] = y_curr
    print(f'Predict curr_omgang {stack_data.datum.min()} - {stack_data.datum.max()}')

    # ------------- predict aktuell omgång och skapa Kelly -------------------
    for model in modeller:
        # nr = model.name[3:]
        # stack_data['proba'+nr] = model.predict(X_curr)
        # stack_data['kelly'+nr] = kelly(stack_data['proba'+nr],
        #                                stack_data[['streck']], None)

        stack_data = skapa_stack_data(model, model.name, X_curr, stack_data)

    veckans_rad, enc = prepare_stack_data(stack_data)
    # veckans_rad = stack_data.copy()

    # veckans_rad.to_csv('veckans_rad.csv', index=False)
    # X_curr.to_csv('X_curr.csv', index=False)
    # print(f'X_curr.shape = {X_curr.shape}')
    # print(f'predict curr veckans_rad.shape = {veckans_rad.shape}')
    temp = veckans_rad.copy().drop(['datum', 'avd', 'y'], axis=1)
    # print(f'predict curr temp.shape = {temp.shape}  - columns:')
    # print(temp.columns)

    for key, values in meta_modeller.items():
        print(f'{key} predicts')
        meta_model = values['model']

        # PREDICTORS???
        if 'ridg' in key:
            veckans_rad[key] = meta_model.predict(temp)
        else:
            veckans_rad[key] = meta_model.predict_proba(temp)[:, 1]

    return veckans_rad

    
def learn_all_and_predict(modeller, meta_modeller, X_train, y_train, X_meta, y_meta, X_curr, y_curr):
    """Learn alla modeller och meta_modeller och gör prediktioner på X_curr"""

    # display(X.shape)

    # """ Learn modeller part och skapa stack_data """
    stack_data = first_learn_modeller(modeller, X_train, y_train, X_meta, y_meta)
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    stack_data.to_csv('first_stack_data.csv', index=False)

    # """ Learn meta_modeller på stack_data """
    meta_modeller = learn_meta_models(meta_modeller, stack_data)
    
    # Learn modeller på allt utom X_curr, y_curr
    #### NEJ NEJ NEJ! Det går ju inte att lära modellerna på X_meta, y_meta som är framtid ### 
    # modeller = final_learn_modeller(modeller, X_train, y_train, X_meta, y_meta)

    # fyll i curr_stack and predict current omgång
    # curr_stack = pd.DataFrame()
    # curr_stack['y'] = y_curr
    # for model in modeller:
    #     curr_stack = skapa_stack_data(model, model.name, X_curr, curr_stack)
    #     # for key, values in meta_modeller.items():
    #     #     meta_model = values['model']

    # # Vilka predictors skall vi ha i learn_meta_models? Det måste vara samm här.

    veckans_rad = predict_curr_omgang(modeller, meta_modeller, X_curr, y_curr)

    return veckans_rad

def backtest(df, df_resultat, modeller, meta_modeller, datumar, gap=0, proba_val=0.6, base_ix=100, meta_ix=150, cv=False, step=1):    
    """ Backtesting anpassad för travets omgångar, dvs datum istf dagar"""
    
    assert base_ix < meta_ix, f'base_ix ({base_ix}) måste vara mindrea än meta_ix ({meta_ix})'
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()

    df_utdelning = pd.read_csv('utdelning.csv')

    for curr_datum_ix in range(base_ix, len(datumar), step):
        curr_datum = datumar[curr_datum_ix]
        placeholder0.empty()
        placeholder0.info(
            f'Aktuell datum: {curr_datum} {"        "} \nant_omgångar spelade: {curr_datum_ix}')

        X_train, y_train, X_meta, y_meta, X_curr, y_curr = skapa_data_för_datum(df, curr_datum_ix)
        if X_train.empty or X_meta.empty or X_curr.empty:
            break        
            
        print(f'learn fram till {curr_datum}')
        veckans_rad = learn_all_and_predict(modeller, meta_modeller, X_train, y_train, X_meta, y_meta, X_curr, y_curr)

        assert cv == False, 'cv==True not implemented'

        spik_strategier = ['1a', '1b', '2b', None]
        kelly_strategier = [None, None,  1,    1]

        # ta fram rader och rättaa dem  
        femmor, sexor, sjuor, utdelning, kostnad, vinst = [], [], [], [], [], []
        last_row = df_resultat.iloc[-1]
        for enum, strategi in enumerate(spik_strategier):
            veckans_rad, cost = ta_fram_meta_rad(
                veckans_rad, meta_modeller, spik_strategier[enum], kelly_strategier[enum], min_avst=0.09)
            kostnad.append(cost)
            veckans_rad.to_csv('veckans_rad'+str(enum)+'.csv', index=False)
            sju, sex, fem, utd = rätta_rad(veckans_rad, curr_datum, df_utdelning)
            sjuor.append(int(sju))
            sexor.append(int(sex))
            femmor.append(int(fem))
            utdelning.append(int(utd))
            vinst.append(int(utdelning[enum] - kostnad[enum]))

        cols = ['t1_vinst','t2_vinst', 't3_vinst', 't4_vinst',
                't1_utd','t2_utd', 't3_utd', 't4_utd',
                't1_kostn', 't2_kostn', 't3_kostn', 't4_kostn', ]
        last_row += vinst + utdelning + kostnad + sjuor + sexor + femmor
        df_resultat.loc[curr_datum] = last_row

        # make all int
        for col in df_resultat.columns:
            df_resultat[col] = df_resultat[col].astype(int)
            
            
        

        df_resultat.to_csv('backtest_resultat.csv', index=True)

        # 3. plotta
        graf_data = df_resultat.copy()
        
        # graf_data.index = pd.to_datetime(graf_data.index, format="%Y-%m-%d")

        placeholder1.empty()
        placeholder2.empty()
        placeholder3.empty()
        
        # Backtest klart och nu plot
        placeholder1.line_chart(graf_data[[
                                't1_vinst', 't2_vinst', 't3_vinst', 't4_vinst']], use_container_width=True)
        placeholder2.line_chart(graf_data[[
                                't1_7', 't2_7', 't3_7', 't4_7']], width=16, height=14, use_container_width=True)
        
        # st.write(df_resultat.plot(kind='line',  y='t1_vinst', rot=45, legend=True, figsize=(20,10)))
        placeholder3.dataframe(df_resultat.sort_index(ascending=False).head(40))
        
    return df_resultat


def kör(df, modeller, meta_modeller, cv=False):
    
    base_ix = 100  # antal omgångar som vi startar bas-modellerna från i backtesting
    meta_ix = 150  # antal omgångar som vi startar meta-modellerna från i backtesting

    ##################################################################################
    # Bestäm i förväg vilka predictors som varje meta-model skall använda?           #
    # Bestäm också spik-strategi och kelly-strategi för varje meta-model             #
    # Kanske en dict är bra?                                                         #
    ##################################################################################

    datumar, df_resultat = starta_upp(df, base_ix)

    # backtesting
    df_resultat = backtest(df, df_resultat, modeller, meta_modeller,
                           datumar, gap=0, proba_val=0.6, base_ix=base_ix, meta_ix=meta_ix, cv=cv, step=1)

    return df_resultat

def main():
    # Skapa v75-instans
    v75 = td.v75(pref='')
    # Hämta data från v75
    _ = v75.förbered_data(missing_num=False)  # num hanteras av catboost
    df = v75.test_lägg_till_kolumner()

    st.info('Kolla om och varför Ridge ger 0 hela tiden ', icon='i')
    st.info('hittar inget datum-överlapp - vad mer kan det vara?')
    
    
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

    # -------------- skapa test-modeller
    #               name,   #häst     proba,    kelly,  #motst,  motst_diff, #fav, only_cl, streck, test,  pref
    test1 = tp.Typ('test1',  True,    True,     False,       0,    False,      0,   False,    True,  True, pref=pref)
    test2 = tp.Typ('test2',  True,    True,     False,       0,    False,      0,   False,    False, True, pref=pref)
    test3 = tp.Typ('test3',  True,    True,     False,       0,    False,      0,   False,    False, True, pref=pref)
    test4 = tp.Typ('test4',  True,    True,     False,       0,    False,      0,   False,    True,  False, pref=pref)

    modeller=[test1, test2, test3, test4]
    
    ##### RandomForestClassifier
    with open('optimera/params_meta1_rf.json', 'r') as f:
        params = json.load(f)
        rf_params = params['params']
    rf_model = RandomForestClassifier(**rf_params, n_jobs=6, random_state=2022)

    ##### RidgeClassifier
    with open('optimera/params_meta2_ridge.json', 'r') as f:
        ridge_params = json.load(f)['params']
        # st.write(params)
    ridge_model = RidgeClassifier(**ridge_params, random_state=2022)

    ##### KNN classifier
    with open('optimera/params_meta3_knn.json', 'r') as f:
        knn_params = json.load(f)['params']
    KNN_model = KNeighborsClassifier(**knn_params, n_jobs=6)

    meta_modeller = {'meta1_rf': {'model': rf_model, 'params': rf_params},
                     'meta2_ridge': {'model': ridge_model, 'params': ridge_params},
                     'meta3_knn': {'model': KNN_model, 'params': knn_params}
                     }

    if st.button('kör'):
        if st.button('med cv'):
            st.warning(f'df_resultat = kör(df, modeller, cv=True)  är inte klar!')
        else:
            df_resultat = kör(df, modeller, meta_modeller, cv=False)


if __name__ == "__main__":
    main()
