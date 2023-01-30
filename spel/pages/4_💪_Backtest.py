import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier, Pool
import sys
import typ as tp
import travdata as td
import json
import concurrent.futures
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
pref = ''
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import skapa_modeller as mod

plt.style.use('fivethirtyeight')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

import logging
logging.basicConfig(level=logging.DEBUG, filemode='w' , filename='v75.log', force=True, encoding='utf-8', format='Backtest: %(asctime)s - %(levelname)s - %(lineno)d - %(message)s ')
logging.info('Startar')
   
st.set_page_config(page_title="Backtest av modeller + meta", page_icon="💪")

st.markdown("# 💪 Backtest av modeller + meta")
st.sidebar.header("💪 Backtest")

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

- Ta medelvärdet av meta-modelerna aritmetiskt eller gemoetriskt
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

# %%
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

    return df.datum.unique(), df_resultat

# %%


def skapa_data_för_datum(df_, curr_datum_ix, frac=0.5):
    df = df_.copy()
    datumar = df.datum.unique()
    curr_datum = datumar[curr_datum_ix]
    base_datum_ix = int(len(datumar[:curr_datum_ix]) * frac)  # base models

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

# %%


def compute_total_insats(veckans_rad):
    summa = veckans_rad.groupby('avd').avd.count().prod() / 2
    return summa

# %%


# def beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning):
#     datum = datum.strftime('%Y-%m-%d')

#     min_utdelning = df_utdelning.loc[df_utdelning.datum == datum, [
#         '7rätt', '6rätt', '5rätt']]

#     tot_utdelning = (min_utdelning['7rätt'] * sjuor + min_utdelning['6rätt']
#                      * sexor + min_utdelning['5rätt'] * femmor).values[0]

#     print('utdelning', tot_utdelning)

#     return tot_utdelning


def varje_avd_minst_en_häst(veckans_rad):
    # ta ut en häst i varje avd - markera valda i df
    for avd in veckans_rad.avd.unique():
        # max av proba i veckans_rad
        max_proba = veckans_rad[veckans_rad.avd == avd]['proba'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (
            veckans_rad.proba == max_proba), 'välj'] = True
    return veckans_rad

# %%


def hitta_spikar(veckans_rad, spikad_avd, spik_strategi, min_avst):
    print('spik_strategi', spik_strategi)
    if spik_strategi == None:
        return veckans_rad, []

    assert spik_strategi in [
        '1a', '1b', '2a', '2b'], "spik_strategi måste ha något av värdena i listan"

    # avdelningar med största avstånd
    avd_avst = []
    for avd in [1, 2, 3, 4, 5, 6, 7]:
        två_bästa = veckans_rad.query('avd == @avd').nlargest(2, 'proba')
        avd_avst.append(två_bästa.proba.values[0] - två_bästa.proba.values[1])

    # print('avstånd',avd_avst)

    # index to the two largest values
    max1 = max(avd_avst)
    avd1 = avd_avst.index(max1)+1
    max2 = max([x for x in avd_avst if x != max1])
    avd2 = avd_avst.index(max2)+1

    if spik_strategi == '1a':
        print('strategi', spik_strategi, 'valde forcerad spik i avd', avd1)
        spikad_avd.append(avd1)      # add avd to a list
        ix = veckans_rad.query('avd == @avd1').nlargest(1,
                                                        'proba').index[0]   # ix for largest in avd
        veckans_rad.loc[ix, ['spik']] = True
        veckans_rad.loc[ix, ['välj']] = True

    elif spik_strategi == '1b':
        if max1 <= min_avst:
            print('strategi', spik_strategi,
                  'inte vald då avst', max1, '< min', min_avst)
            return veckans_rad, []

        print('strategi', spik_strategi, 'valde spik i avd',
              avd1,  'då avst', max1, '> min', min_avst)
        spikad_avd.append(avd1)      # add avd to a list
        ix = veckans_rad.query('avd == @avd1').nlargest(1,
                                                        'proba').index[0]   # ix for largest in avd
        veckans_rad.loc[ix, ['spik']] = True
        veckans_rad.loc[ix, ['välj']] = True

    elif spik_strategi == '2a':
        print('strategi', spik_strategi, 'valde forcerad spik i avd', avd1)
        spikad_avd.append(avd1)      # add avd to a list
        ix = veckans_rad.query('avd == @avd1').nlargest(1,
                                                        'proba').index[0]   # ix for largest in avd
        veckans_rad.loc[ix, ['spik']] = True
        veckans_rad.loc[ix, ['välj']] = True
        print('strategi', spik_strategi, 'valde 2:a forcerad spik i avd', avd2)
        spikad_avd.append(avd2)      # add avd to a list
        ix = veckans_rad.query('avd == @avd2').nlargest(1,
                                                        'proba').index[0]   # ix for largest in avd
        veckans_rad.loc[ix, ['spik']] = True
        veckans_rad.loc[ix, ['välj']] = True

    elif spik_strategi == '2b':
        if max1 <= min_avst:
            print('strategi', spik_strategi,
                  'inget valt då avst', max1, '< min', min_avst)
            return veckans_rad, []

        print('strategi', spik_strategi, 'valde spik i avd',
              avd1,  'då avst', max1, '> min', min_avst)
        spikad_avd.append(avd1)      # add avd to a list
        ix = veckans_rad.query('avd == @avd1').nlargest(1, 'proba').index[0]
        veckans_rad.loc[ix, ['spik']] = True
        veckans_rad.loc[ix, ['välj']] = True

        if max2 <= min_avst:
            print(print('strategi', spik_strategi,
                  '2:an inte vald då avst', max2, '< min', min_avst))
            return veckans_rad, spikad_avd

        print('strategi', spik_strategi, 'valde 2:a spik i avd',
              avd2,  'då avst', max2, '> min', min_avst)
        spikad_avd.append(avd2)      # add avd to a list
        ix = veckans_rad.query('avd == @avd2').nlargest(1, 'proba').index[0]
        veckans_rad.loc[ix, ['spik']] = True
        veckans_rad.loc[ix, ['välj']] = True

    # print('spikad_avd', spikad_avd)

    return veckans_rad, spikad_avd


def plocka_en_efter_en(veckans_rad, spikad_avd, max_cost=300):
    """_summary_
    Args:
        veckans_rad (_type_): df att fylla
        spikad_avd (_type_): lista med spikade avdelningar
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
        # e) avbryt vid max_cost
        cost = compute_total_insats(veckans_rad.query("välj==True"))
        if cost > max_cost:
            # ta tillbaks den sist spelade
            veckans_rad.loc[curr_index, 'välj'] = False
            break

    cost = compute_total_insats(veckans_rad.query("välj==True"))

    return veckans_rad, cost


def ta_fram_meta_rad(veckans_rad_, meta_modeller, spik_strategi, max_cost=300, min_avst=0.07, mean='geometric'):
    """ Denna funktion tar fram en rad för meta-modellerna via medelvärdet på alla meta-modeller
    veckans_rad innehåller _en omgång_
    _spik_strategi_: None - inget, '1a' - forcera 1 spik, '2a' - forcera 2 spikar, '1b' - 1 spik endast om klar favorit, '2b' - 2 spikar för endast klara favoriter 
    _mean_: 'arithmetic or geometric formula for mean
    """
    veckans_rad = veckans_rad_.copy()
    # veckans_rad['kelly_val'] = False
    veckans_rad['välj'] = False   # inga rader valda ännu
    veckans_rad['spik'] = False   # inga spikar valda ännu

    veckans_rad['proba'] = 0
    # veckans_rad['kelly'] = 0

    ### Här tar vi medelvärdet av predict_proba från meta-modellerna ######
    for enum, key in enumerate(meta_modeller.keys()):
        if mean == 'arithmetic':
            veckans_rad['proba'] += veckans_rad[key]
            # veckans_rad['kelly'] += veckans_rad['kelly'+enum]
        else:
            if enum == 0:
                veckans_rad['proba'] = veckans_rad[key].copy()
                continue
            veckans_rad['proba'] *= veckans_rad[key]

    if mean == 'arithmetic':
        veckans_rad['proba'] /= len(meta_modeller)

    else:  # geometric
        veckans_rad['proba'] = veckans_rad['proba'] ** (1/len(meta_modeller))

    #################################################################

    veckans_rad = varje_avd_minst_en_häst(veckans_rad)

    spikad_avd = []
    if spik_strategi:
        veckans_rad, spikad_avd = hitta_spikar(
            veckans_rad, spikad_avd, spik_strategi, min_avst)
    else:
        print('ingen spik-strategi')

    # sortera upp i proba-ordning. Om kelly skapa en sortering efter kelly-ordning
    veckans_rad = veckans_rad.sort_values(by=['proba'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)

    # plocka en efter en tills kostnaden är för stor
    # return veckans_rad, cost
    return plocka_en_efter_en(veckans_rad, spikad_avd, max_cost)


# def rätta_rad(df, datum, df_utdelning):
#     """
#     Räkna ut antal 5:or, 6:or resp. 7:or
#     Hämta ev utdelning
#     Spara datum, resultat, utdelning och rad-kostnad
#     """
#     sjuor, sexor, femmor, utdelning = 0, 0, 0, 0

#     min_tabell = df[['y', 'avd', 'häst', 'rel_rank', 'välj']].copy()
#     min_tabell.sort_values(by=['avd', 'y'], ascending=False, inplace=True)

#     print('Antal rätt', min_tabell.query('välj==True and y==1').y.sum())

#     # 1. om jag har max 7 rätt
#     if min_tabell.query('välj==True and y==1').y.sum() == 7:
#         sjuor = 1
#         sexor = (min_tabell.groupby('avd').välj.sum()).sum()-7
#         # antal femmor
#         ant1 = min_tabell.query('avd==1 and välj==True').välj.sum()-1
#         ant2 = min_tabell.query('avd==2 and välj==True').välj.sum()-1
#         ant3 = min_tabell.query('avd==3 and välj==True').välj.sum()-1
#         ant4 = min_tabell.query('avd==4 and välj==True').välj.sum()-1
#         ant5 = min_tabell.query('avd==5 and välj==True').välj.sum()-1
#         ant6 = min_tabell.query('avd==6 and välj==True').välj.sum()-1
#         ant7 = min_tabell.query('avd==7 and välj==True').välj.sum()-1
#         femmor = ant1*ant2+ant1*ant2+ant1*ant3+ant1*ant4+ant1*ant5+ant1*ant6+ant1*ant7 +\
#             ant2*ant3+ant2*ant4+ant2*ant5+ant2*ant6+ant2*ant7 + \
#             ant3*ant4+ant3*ant5+ant3*ant6+ant3*ant7 + \
#             ant4*ant5+ant4*ant6+ant4*ant7 + \
#             ant5*ant6+ant5*ant7 + \
#             ant6*ant7

#     # 2. om jag har max 6 rätt
#     if min_tabell.query('välj==True and y==1').y.sum() == 6:
#         avd_fel = min_tabell.loc[((min_tabell.välj == False) & (
#             min_tabell.y == 1)), 'avd'].values[0]
#         # print(min_tabell.query('avd== @avd_fel').välj.sum())
#         sexor = min_tabell.query('avd==@avd_fel').välj.sum()
#         # antal femmor
#         femmor_fel, femmor_rätt = 0, 0
#         for avd in range(1, 8):
#             if avd == avd_fel:
#                 femmor_fel += min_tabell.loc[min_tabell.avd ==
#                                              avd_fel].välj.sum()

#             femmor_rätt += min_tabell.query(
#                 'avd==@avd and välj==True').välj.sum()-1
#         # print(f'femmor_rätt = {femmor_rätt} femmor_fel = {femmor_fel}')
#         femmor = femmor_fel * femmor_rätt

#     # 3. om jag har max 5 rätt
#     if min_tabell.query('välj==True and y==1').y.sum() == 5:
#         avd_fel = min_tabell.loc[((min_tabell.välj == False) & (
#             min_tabell.y == 1)), 'avd'].values
#         femmor = min_tabell.loc[min_tabell.avd == avd_fel[0]].välj.sum(
#         ) * min_tabell.loc[min_tabell.avd == avd_fel[1]].välj.sum()

#     return sjuor, sexor, femmor, beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning)

# TODO: Använd mod.rätta_rad istället och gör datum till en string datum.strftime('%Y-%m-%d') innan anrop

# def initiera_veckans_rader(X_curr, y_curr, antal_rader):
#     # ---------- initier veckans rad med aktuell omgång ----------------------
#     veckans_rader = []
#     for i in range(antal_rader):
#         veckans_rader.append(X_curr[['datum', 'avd', 'häst', 'bana',
#                                      'kusk', 'streck', 'streck_avst', 'rel_rank']].copy())
#         veckans_rader[i]['y'] = y_curr
#         veckans_rader[i]['välj'] = False

#     return veckans_rader

# %%
def build_stack_data(modeller, X_meta, y_meta):
    """ Bygg stack_data inklusive ev Kelly
        X_meta:     innehåller datum och avd men ej y
        stack_data: innehåller 'allt' i slutet av funktionen
    """

    stack_data = X_meta.copy()
    stack_data['y'] = y_meta
    data = X_meta.copy()
    for model in modeller:
        this_proba = model.predict(data)
        nr = model.name[3:]
        # print(nr, 'data.shape', data.shape)
        # print(nr, 'stack_data.shape', stack_data.shape)
        stack_data['proba'+nr] = this_proba

    return stack_data


def prepare_stack_data(stack_data_, y, ENC=None):
    """Hantera missing values, NaN, etc för meta-modellerna"""
    assert 'y' not in stack_data_.columns, "y shouldn't be in stack_data"

    stack_data = stack_data_.copy()

    if ENC is None:
        # a new encode needs y
        assert y is not None, "y is needed for new encoding"
    else:
        # use the existing encoder - y is not used
        assert y is None, "y should be None for existing encoding"

    #""" Fyll i saknade numeriska värden med 0"""
    numericals = stack_data.select_dtypes(exclude=['object']).columns
    stack_data[numericals] = stack_data[numericals].fillna(0)

    #""" Fyll i saknade kategoriska värden med 'missing'"""
    categoricals = stack_data.select_dtypes(include=['object']).columns
    stack_data[categoricals] = stack_data[categoricals].fillna('missing')

    #""" Hantera high cardinality"""
    # cardinality_list=['häst','kusk','h1_kusk','h2_kusk','h3_kusk','h4_kusk','h5_kusk']

    #""" Target encoding"""
    target_encode_list = ['bana', 'häst', 'kusk', 'kön', 'h1_kusk', 'h1_bana', 'h2_kusk', 'h2_bana',
                          'h3_kusk', 'h3_bana', 'h4_kusk', 'h4_bana', 'h5_kusk', 'h5_bana']

    if ENC == None:
        ENC = TargetEncoder(cols=target_encode_list,
                            min_samples_leaf=20, smoothing=10).fit(stack_data, y)

    # print('stack_data.shape = ', stack_data.shape)
    stack_data = ENC.transform(stack_data)

    # stack_data[target_encode_list] = stack_data_encoded[target_encode_list]  # update with encoded values
    return stack_data, ENC


def train_modeller(modeller, X_train, y_train):
    ############################################################################################################
    #                        Här görs en learn av modeller och sedan skapas stack_data
    #                        - Learn modeller på X_train,y_train
    #                        - Skapa stack_data med learned model på X_meta
    #                        - prepare stack_data
    ############################################################################################################

    # features = X_train.drop(['datum','avd','y'], axis=1).columns.tolist()
    # X_train skall innehålla datum och avd
    for model in modeller:
        name = model.name
        print(f'Learn {name} {X_train.datum.min()} t.o.m {X_train.datum.max()}')
        # print(X_train)
        model.learn(X_train, y_train, params=None, save=True)

    return

############################################################################################################
#                       Här gör vi learn av meta_modeller på stack_data
############################################################################################################


def learn_meta_models(meta_modeller, stack_data, save=True):
    # global ENC
    # print('Learn meta_modeller på stack_data')
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    stack_data.head(10).to_csv('stack_data_före_drop.csv', index=False)
    y = stack_data.y.astype(int)
    meta_features = stack_data.drop(['datum', 'avd', 'y'], axis=1).columns.to_list()
    X_meta, ENC = prepare_stack_data(stack_data[meta_features], y)

    with open(pref+'modeller/test_encoder.pkl', 'wb') as f:
        pickle.dump(ENC, f)

    # X_meta.drop(['streck'], axis=1, inplace=True)

    y_meta = y
    # X_meta.to_csv('X_meta_Learn.csv', index=False)
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


# Denna skall inte köras
def final_train_modeller(modeller, X, y, X_meta, y_meta):
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

    print(f'Predict curr_omgang {X_curr.datum.min()} - {X_curr.datum.max()}')

    # ------------- predict aktuell omgång för att skapa stack_data -------------------
    stack_data = build_stack_data(modeller, X_curr, y_curr)

    # load ENC
    with open(pref+'modeller/test_encoder.pkl', 'rb') as f:
        ENC = pickle.load(f)

    veckans_rad, _ = prepare_stack_data(
        stack_data.drop(['datum', 'avd', 'y'], axis=1), None, ENC)
    temp = veckans_rad.copy()

    # temp.drop(['streck'], axis=1, inplace=True)    # ta bort streck från meta_features

    for key, values in meta_modeller.items():
        print(f'{key} predicts')
        meta_model = values['model']

        if 'ridg' in key:
            veckans_rad[key] = meta_model._predict_proba_lr(temp)[:, 1]
        else:
            veckans_rad[key] = meta_model.predict_proba(temp)[:, 1]
    veckans_rad[['datum', 'avd', 'y']] = stack_data[['datum', 'avd', 'y']]

    return veckans_rad


def learn_all_and_predict(modeller, meta_modeller, X_train, y_train, X_meta, y_meta, X_curr, y_curr):
    """Learn alla modeller och meta_modeller och gör prediktioner på X_curr"""

    # display(X.shape)

    #""" train modeller"""
    train_modeller(modeller, X_train, y_train)

    # bygg stack_data från modeller
    stack_data = build_stack_data(modeller, X_meta, y_meta)
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    stack_data.to_csv('first_stack_data.csv', index=False)

    #""" Learn meta_modeller på stack_data"""
    meta_modeller = learn_meta_models(meta_modeller, stack_data)

    # Vilka predictors skall vi ha i learn_meta_models? Det måste vara samm här.
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
            f'Aktuell datum: {curr_datum} {"        "} \nomgång: {curr_datum_ix}')

        X_train, y_train, X_meta, y_meta, X_curr, y_curr = skapa_data_för_datum(
            df, curr_datum_ix)
        if X_train.empty or X_meta.empty or X_curr.empty:
            break

        print(f'learn fram till {curr_datum}')
        veckans_rad = learn_all_and_predict(modeller,
                                            meta_modeller,
                                            X_train, y_train,
                                            X_meta, y_meta,
                                            X_curr, y_curr)

        assert cv == False, 'cv==True not implemented'

        spik_strategier = ['1a', '1b', '2b', None]

        # ta fram rader och rättaa dem
        femmor, sexor, sjuor, utdelning, kostnad, vinst = [], [], [], [], [], []
        last_row = df_resultat.iloc[-1]
        for enum, strategi in enumerate(spik_strategier):
            veckans_rad, cost = ta_fram_meta_rad(veckans_rad, meta_modeller, spik_strategier[enum])
            print('cost', cost)
            kostnad.append(cost)
            veckans_rad.to_csv('veckans_rad'+str(enum)+'.csv', index=False)
            sju, sex, fem, utd = mod.rätta_rad(veckans_rad, curr_datum, df_utdelning)
            sjuor.append(int(sju))
            sexor.append(int(sex))
            femmor.append(int(fem))
            utdelning.append(int(utd))
            vinst.append(int(utdelning[enum] - kostnad[enum]))

        cols = ['t1_vinst', 't2_vinst', 't3_vinst', 't4_vinst',
                't1_utd', 't2_utd', 't3_utd', 't4_utd',
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

        # Backtest klart och nu pandas plot med gridlines
        placeholder1.line_chart(graf_data[[
                                't1_vinst', 't2_vinst', 't3_vinst', 't4_vinst']], use_container_width=True)

        # placeholder2.line_chart(graf_data[[
        #                         't1_7', 't2_7', 't3_7', 't4_7']], width=16, height=14, use_container_width=True)

        # st.write(df_resultat.plot(kind='line',  y='t1_vinst', rot=45, legend=True, figsize=(20,10)))
        placeholder3.dataframe(df_resultat.sort_index(ascending=False).head(40))
        df_resultat.to_csv('backtest_resultat.csv', index=True)
        
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

    ###############################################################
    # Några idéer på nya kolumner:
    #  -   ❌ streck/sum(streck för avd) - fungerar inte bra. Nästan alla sum == 100 per avd
    #  a - ✔️ plats(streck)/ant_hästar_i_avd (antal startande hästar i avd)
    #  b - ❌ pris / tot_pris_i_avd - går inte att använda ju! pris är ju loppets 1.pris - samma för all i loppet
    #  c - ✔️ kr / tot_kr_i_avd     rel_kr
    #  d - ✔️ Avståndet till ettan (streck)
    #  e - ✔️ hx_bana samma som bana
    #  f - ✔️ hx_kusk samma som kusk

    # -------------- skapa test-modeller
    #               name,   #häst  #motst,  motst_diff, streck, test,  pref
    test1 = tp.Typ('test1', False,   0,     False,      False,  True,  pref=pref)
    test2 = tp.Typ('test2', False,   3,     True,       False,  True,  pref=pref)
    test3 = tp.Typ('test3', True,    0,     False,      False,  False, pref=pref)
    test4 = tp.Typ('test4', True,    3,     True,       False,  False, pref=pref)

    modeller = [test1, test2, test3, test4]

    # RandomForestClassifier
    with open('optimera/params_meta1_rf.json', 'r') as f:
        params = json.load(f)
        rf_params = params['params']
    rf_model = RandomForestClassifier(**rf_params, n_jobs=6, random_state=2022)

    # RidgeClassifier
    with open('optimera/params_meta2_ridge.json', 'r') as f:
        ridge_params = json.load(f)['params']
        # st.write(params)
    ridge_model = RidgeClassifier(**ridge_params, random_state=2022)

    # KNN classifier
    with open('optimera/params_meta3_knn.json', 'r') as f:
        knn_params = json.load(f)['params']
    KNN_model = KNeighborsClassifier(**knn_params, n_jobs=6)

    # ExtraTreesClassifier
    with open('optimera/params_meta4_et.json', 'r') as f:
        params = json.load(f)
        et_params = params['params']
    et_model = ExtraTreesClassifier(**et_params, n_jobs=6, random_state=2022)

    meta_modeller = {
        # 'meta1_rf': {'model': rf_model, 'params': rf_params},
        'meta2_ridge': {'model': ridge_model, 'params': ridge_params},
        'meta3_knn': {'model': KNN_model, 'params': knn_params},
        # testa et istället för rf
        'meta4_et': {'model': et_model, 'params': et_params},
    }

    if st.button('kör'):
        st.write('Införa kör med cv?')
        df_resultat = kör(df, modeller, meta_modeller, cv=False)
    # elif st.button('kör med cv'):
    #     st.warning(f'df_resultat = kör(df, modeller, cv=True)  är inte klar!')


if __name__ == "__main__":
    main()
