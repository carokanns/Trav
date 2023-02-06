#%%
import sys
sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')
import typ as tp
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier, Pool

import travdata as td
import json
import concurrent.futures
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
pref = ''

import skapa_modeller as mod

plt.style.use('fivethirtyeight')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

import logging
logging.basicConfig(level=logging.DEBUG, filemode='w' , filename='v75.log', force=True, encoding='utf-8', format='Backtest: %(asctime)s - %(levelname)s - %(lineno)d - %(message)s ')
logging.info('***************** Startar *******************')
   
st.set_page_config(page_title="Backtesting av L1 + L2", page_icon="💪")

st.markdown("# 💪 Backtesting av modeller + meta")
st.sidebar.header("💪 Backtesting")

#%%
exp = st.expander('Beskrivning av testet')
exp.write("""
## Först L1-modellerna
Sätt start_datum tex start_datum = '2016-08-01.    

0. Varje modell har indivduella kolumner samt strategi för att ta fram raden.  
    a. Speca per modell i Typ-klassen.
1. Loop över veckor. Learn fram till aktuell vecka och spara modell.  
    a. Skit i cv tillsvidare
    b. Ev Spara modell med aktuell datum i namnet.
2. Predict nästa vecka (generera rad enl modellens strategi)
3. Rätta, Hämta ev priser och Spara resultat som en df i en csv-fil.
4. plot resultat
5. Repeat för varje modell och strategi 

## Gör detsamma för meta-modeller (rf, knn, ridge)
- Använd sparade L1-modeller och generara stack-data från start_datum till aktuell_datum  
- Lär upp L2-modeller på stack-data, använd strategins hyperparms 
- Hur vet vi hur länge meta-modellen skall köra?  
    - Kanske göra ett test innan på ganska stor test-data och spara som hyperparm
- Predict nästa vecka enl strategi för resp L2_modell skapa kolumnerna 'proba_cat1L2', etc  

- Beräkna viktade meta_värden från L2-modellernas output ('proba_cat1L2', etc)
- Ta medelvärdet av L2-modelerna aritmetiskt eller gemoetriskt
- Rätta och spara resultat plus ev priser
- plot resultat
- Repeat för varje meta-modell

"""
          )

def log_print(text, logging_level='d'):
    """Skriver ut på loggen och gör en print samt returnerar strängen (för assert)"""
    if logging_level == 'd':
        logging.debug(text)
    else:
        if logging_level == 'i':
            logging.info(text)
        elif logging_level == 'w':
            logging.warning(text)
        elif logging_level == 'e':
            logging.error(text)
        print(text)

    return text

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

# %%
def compute_total_insats(veckans_rad):
    summa = veckans_rad.groupby('avd').avd.count().prod() / 2
    return summa

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

def ta_fram_meta_rad(veckans_rad_, L2_modeller, strategi, max_cost=300, min_avst=0.07, mean='geometric'):
    """ Denna funktion tar fram en rad för meta-modellerna via medelvärdet på alla meta-modeller
    veckans_rad innehåller _en omgång_
    strategi: None - inget, vi får se
    mean: 'arithmetic or geometric formula for mean
    """
    veckans_rad = veckans_rad_.copy()
    # veckans_rad['kelly_val'] = False
    veckans_rad['välj'] = False   # inga rader valda ännu
    veckans_rad['spik'] = False   # inga spikar valda ännu

    veckans_rad['proba'] = 0
    # veckans_rad['kelly'] = 0

    ### Här tar vi medelvärdet av predict_proba från meta-modellerna ######
    for enum, key in enumerate(L2_modeller.keys()):
        if mean == 'arithmetic':
            veckans_rad['proba'] += veckans_rad[key]
            # veckans_rad['kelly'] += veckans_rad['kelly'+enum]
        else:
            if enum == 0:
                veckans_rad['proba'] = veckans_rad[key].copy()
                continue
            veckans_rad['proba'] *= veckans_rad[key]

    if mean == 'arithmetic':
        veckans_rad['proba'] /= len(L2_modeller)

    else:  # geometric
        veckans_rad['proba'] = veckans_rad['proba'] ** (1/len(L2_modeller))

    #################################################################

    veckans_rad = varje_avd_minst_en_häst(veckans_rad)

    spikad_avd = []
    if strategi:
        print('strategi ej klar')
    else:
        print('ingen spik-strategi')

    # sortera upp i proba-ordning. Om kelly skapa en sortering efter kelly-ordning
    veckans_rad = veckans_rad.sort_values(by=['proba'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)

    # plocka en efter en tills kostnaden är för stor
    # return veckans_rad, cost
    return plocka_en_efter_en(veckans_rad, spikad_avd, max_cost)

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

def train_L1_models(L1_modeller, X_train, y_train):
    ############################################################################################################
    #                        Här görs en learn av modeller
    #                        - Learn modeller på X_train,y_train
    ############################################################################################################

    # features = X_train.drop(['datum','avd','y'], axis=1).columns.tolist()
    # X_train skall innehålla datum och avd
    for model in L1_modeller:
        name = model.name
        print(f'Learn {name} {X_train.datum.min()} t.o.m {X_train.datum.max()}')
 
        # Learn(X_, y=None, X_test_=None, y_test=None, params=None, use_L2_features_=None, save=True)
        model.learn(X_train, y_train, params=None, save=True)

    return

############################################################################################################
#                       Här gör vi learn av meta_modeller på stack_data
############################################################################################################

def train_L2_models(meta_modeller, stack_data, save=True):
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

def initiera_veckans_rader_old(X_curr, y_curr, antal_rader):
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

def old_learn_all_and_predict(L1_models, L2_models, X_train, y_train, X_meta, y_meta, X_curr, y_curr):
    """Learn alla modeller och meta_modeller och gör prediktioner på X_curr"""

    # train L1-modeller på allt <= L1_datum
    train_L1_models(L1_models, X_train, y_train)

    # bygg stack_data från L1-modeller på date > L1_datum och <= L2_datum
    stack_data = build_stack_data(L1_models, X_meta, y_meta)
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    stack_data.to_csv('first_stack_data.csv', index=False)

    #""" Learn meta_modeller på stack_data"""
    L2_models = train_L2_models(L2_models, stack_data)

    # predict date==curr_datum
    veckans_rad = predict_curr_omgang(L1_models, L2_models, X_curr, y_curr)

    return veckans_rad

def learn_modeller_och_predict(df_, curr_datum, L2_datum, L1_modeller, L2_modeller, use_features):
    """ Skapar en DataFrame som blir grunden till veckans-rad
    Args:
        df_ (DataFrame): allt data rensat och tvättat
        curr_datum (string): aktuell datum som skall predikteras
        L2_datum (string): bryt-datum mellan Layer1 och Layer2
        L1_modeller (list): Layer1-modeller
        L2_modeller (list): Layer2-modeller
        use_features (list): Features för Layer1-modeller
    Returns:
        DataFrame: Färdig df med proba-kolumner plus meta-kolumn
    """
    WEIGHTS = [0.25, 0.25, 0.25, 0.25]
    log_print(f'learn_modeller_och_predict: med curr_datum={curr_datum} och L2_datum={L2_datum}','i')
    log_print(f'learn_modeller_och_predict: orginaldata={df_.shape}','i')
    
    ############# Learn L1-modeller på allt <= L2_datum ############################
    L1_learn_input_df = df_.query('datum < @L2_datum').copy()
    assert L1_learn_input_df.shape[0] > 0, log_print(f'L1_learn_input_df är tom','e')
     
    log_print(f'learn_modeller_och_predict: L1_input_df shape={L1_learn_input_df.shape} min_dat= \
    {L1_learn_input_df.datum.min()}, max_dat={L1_learn_input_df.datum.max()}','i')
    
    L1_modeller = mod.learn_L1_modeller(L1_modeller, L1_learn_input_df, use_features, save=True)
    log_print(f'learn_modeller_och_predict: L1_modeller är nu lärt {L1_modeller.keys()}','i')
  
    ############# Skapa data till L2 på datum >= L2_datum och datum < curr_datum #############
    stack_df = df_.query('datum >= @L2_datum and datum < @curr_datum').copy()   # skall utökas med probas
    assert stack_df.shape[0] > 0, log_print(f'stack_df är tom', 'e')

    stack_df, L2_features = mod.create_L2_input(stack_df, L1_modeller, use_features, with_y=True) 
    assert stack_df.shape[0] > 0, log_print(f'stack_df är tom', 'e')
    log_print(f'learn_modeller_och_predict: L2_learn_input_df shape={stack_df.shape} min_dat=\
    {stack_df.datum.min()}, max_dat={stack_df.datum.max()}', 'i')
    
    # filter features in L2_features starting with 'proba_'
    assert len([x for x in L2_features if x.startswith('proba_')])==4, log_print(f'Antal proba-kolumner skall vara 4', 'e')
    ############# Nu har vi adderat proba_kolumner till stack_df ############################
    
    ############# Learn L2-modeller på stack_df med proba_kolumner ############################
    L2_modeller = mod.learn_L2_modeller(L2_modeller, stack_df, L2_features, save=True)
    log_print(f'learn_modeller_och_predict: L2_modeller är nu lärt {L2_modeller.keys()}','i')
    
    ############# Skapa data till L2 för curr_datum ############################
    curr_data_df = df.query('datum == @curr_datum').copy()
    log_print(f'learn_modeller_och_predict: curr_data_df shape={curr_data_df.shape} min_dat=\
    {curr_data_df.datum.min()}, max_dat={curr_data_df.datum.max()}', 'i')

    curr_stack_df, L2_features = mod.create_L2_input(curr_data_df, L1_modeller, use_features, with_y=True)
    assert curr_stack_df.shape[0] > 0, log_print(f'curr_stack_df är tom', 'e')
    
    log_print(f'learn_modeller_och_predict: curr_stack_df shape={curr_stack_df.shape} min_dat=\
    {curr_stack_df.datum.min()}, max_dat={curr_stack_df.datum.max()}', 'i')
    
    log_print(f'learn_modeller_och_predict: L2_features {L2_features[-5:]}', 'i')
    
    assert len([col for col in curr_stack_df.columns if 'proba' in col]) == 4, \
        log_print(f'curr_stack_df skall ha 4 proba-kolumner', 'e')      
    ############# Nu ligger proba_kolumner i curr_stack_df  ############################

    ############# predict curr_stack_df med L2 dvs läggg till viktad meta #####################
    curr_stack_df = mod.predict_med_L2_modeller(L2_modeller, curr_stack_df, L2_features, weights=WEIGHTS)
    assert curr_stack_df.shape[0] > 0, log_print(f'curr_stack_df är tom'  ,'e')
    assert 'meta' in curr_stack_df.columns, log_print(f'curr_stack_df saknar meta-kolumn'  ,'e')
    
    log_print(f'learn_modeller_och_predict: curr_stack_df shape={curr_stack_df.shape} min_dat=\
    {curr_stack_df.datum.min()}, max_dat={curr_stack_df.datum.max()}','i')
    ############# Nu ligger proba_kolumner plus viktad meta i den nya curr_stack_df ############################
    
    return curr_stack_df

def välj_ut_hästar_för_spel(strategi, df_proba):
    
    print(f'df_proba {df_proba.shape}')
    return mod.välj_rad(df_proba)

def beräkna_resultat(veckans_rad, curr_datum):
    df_utdelning = pd.read_csv(pref+'data/utdelning.csv')
    # se sju, sex, fem, utd = mod.rätta_rad(veckans_rad, curr_datum, df_utdelning)
    return None

def plot_resultat(df_resultat):
    # Beräkna cumulativ vinst_förlust per strategi över datum
    df_resultat['cumulative_profit'] = df_resultat.groupby('strategi')['vinst_förlust'].cumsum()

    # Skapa en line plot för cumulativ vinst_förlust per strategi över datum
    fig, ax = plt.subplots()
    for strategy in df_resultat['strategy'].unique():
        strategy_data = df_resultat[df_resultat['strategy'] == strategy]
        ax.plot(strategy_data['datum'],
                strategy_data['cumulative_profit'], label=strategy)
    ax.set_xlabel('Datum')
    ax.set_ylabel('Kumulativ vinst/förlust')
    ax.legend()
    plt.show()

    # Skapa en bar plot för antal 7-5 rätt per strategi per datum
    fig, ax = plt.subplots()
    for strategy in df_resultat['strategi'].unique():
        strategy_data = df_resultat[df_resultat['strategy'] == strategy]
        ax.bar(strategy_data['datum'], strategy_data['7_rätt'],
               label=strategy + ' - 7 rätt')
        ax.bar(strategy_data['datum'], strategy_data['6_rätt'],
               label=strategy + ' - 6 rätt')
        ax.bar(strategy_data['datum'], strategy_data['5_rätt'],
               label=strategy + ' - 5 rätt')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Antal 7-5 rätt')
    ax.legend()
    plt.show()

def bygg_strategi1(a):
    return None

def bygg_strategi2(b,c):
    return None

def next_datum(df, curr_datum=None, step=1):
    """ Tar fram nästa datum för testet
    Args:
        df (dataframe): all_data.csv
        curr_datum (string, optional): aktuell datum för testet. None betyder första gången.
        step (int, optional): antal datum som ska stegas framåt. Defaults to 1.
        gap (int, optional): antal datum utöver nästa mellan learn och predict. Defaults to 0.
    Returns:
        tuple: L2_datum, ny curr_datum
    """
    
    alla_datum = df.datum.astype(str).unique().tolist()
    if curr_datum is None:
        curr_datum = alla_datum[200]  # första gången är det 200 veckor för rejäl start av training   
    
    log_print(f'next_datum 1: curr_datum: {curr_datum}, type: {type(curr_datum)}', 'i')
    ix2 = np.where(alla_datum == curr_datum)[0][0]+step   # index till ny curr_datum
    ix1 = int(round(ix2/2)+0.5) # index till L2_datum   (learn fr o m L2_datum till curr_datum)
    # learn L1_modeller på allt fram till L2_datum
    
    ############################### L1_datum    
    # Gör så att next_datum är L2_datum.index + 1
    # beräkan 50/50 mha next_datum för L1_datum och L2_datum
    
    if ix2 < len(alla_datum):
        curr_datum = alla_datum[ix2] # nästa omgång att testa på
        L2_datum = alla_datum[ix1]  # till_datum för att lära L1-modellern (L2-modellerna startar from L2_datum)
    else:
        curr_datum, L2_datum = None, None
    
    log_print(f'next_datum: curr_datum={curr_datum} type={type(curr_datum)}','i'    ) 
    return L2_datum, curr_datum

def backtest(df, L1_modeller, L2_modeller, step=1, gap=0, proba_val=0.6):
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    
    use_features, cat_features, num_features = mod.read_in_features()
    # Bygg strategier
    strategier = [
        # name           function, modell_lista, hyperparams, use_features
        (bygg_strategi1, 'a'),
        (bygg_strategi2, 'b','c'),
        # ... fler strategier ...
    ]
    df_resultat = pd.DataFrame(columns=['datum', 'strategi', 'vinst_forlust', '7_rätt', '6_rätt', '5_rätt'])
    df_utdelning = pd.read_csv('utdelning.csv')
    
    # Dela upp datan i lär- och test-delar
    L2_datum, curr_datum = next_datum(df, curr_datum=None, step=1)

    # För varje unikt datum i test-delen
    while curr_datum is not None: 
        # for strategi_name, (strategi, L1_modeller, L2_modeller, use_features) in strategier.items():
        for strategi in strategier:
            placeholder0.empty()
            placeholder0.info(f'Aktuell datum: {curr_datum} {"        "} \n strategi: {strategi}')
            
            # kanske bättre att skicka med func in i de olika funktionerna nedan
            func = strategi[0]
            args = strategi[1:]
            strategy = func(*args)
            # kanske bättre att skicka med func in i de olika funktionerna nedan

            # TODO: Använd curr_datum mfl för att skapa L1_input och L2_input
            
            # Lär modeller och predict aktuell datum
            L2_output_df = learn_modeller_och_predict(
                df, curr_datum, L2_datum, L1_modeller, L2_modeller, use_features)
            assert L2_output_df.shape[0] > 0, 'L2_output_df.shape[0] empty'
            
            # Välj ut hästar för spel df_proba skall innehålla data för en omgång
            veckans_rad, kostnad = välj_ut_hästar_för_spel(strategi, L2_output_df)
            log_print(f'veckans_rad.shape: {veckans_rad.shape}  Kostnad: {kostnad}')
            
            # Beräkna resultat
            utdelning, _7_rätt, _6_rätt, _5_rätt = mod.rätta_rad(veckans_rad, curr_datum, df_utdelning)
            vinst_forlust =  utdelning - kostnad
            
            # spara resultat i dataframe df_resultat
            df_resultat.loc[len(df_resultat)] = [curr_datum, strategy, vinst_forlust, _7_rätt, _6_rätt, _5_rätt]


            # Plot resultat
            placeholder1.empty()
            placeholder2.empty()
            placeholder3.empty()
            plot_resultat(df_resultat)
            
        L1_datum, L2_datum, curr_datum = next_datum(df, curr_datum, step, gap)
        
def backtest_old(df, df_resultat, modeller, meta_modeller, datumar, gap=0, proba_val=0.6, base_ix=100, meta_ix=150, cv=False, step=1):
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
        veckans_rad = old_learn_all_and_predict(modeller,
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

def kör(df, L1_modeller, L2_modeller, cv=False):

    base_ix = 100  # antal omgångar som vi startar bas-modellerna från i backtesting
    meta_ix = 150  # antal omgångar som vi startar meta-modellerna från i backtesting

    ##################################################################################
    # Bestäm i förväg vilka predictors som varje meta-model skall använda?           #
    # Bestäm också spik-strategi och kelly-strategi för varje meta-model             #
    # Kanske en dict är bra?                                                         #
    ##################################################################################

    datumar, df_resultat = starta_upp(df, base_ix)

    # backtesting
    assert 'y' in df,log_print(f'y saknas i df')  
    df_resultat = backtest(df, L1_modeller, L2_modeller, gap=0, proba_val=0.6, step=1)

    return df_resultat

def main():
    # Skapa v75-instans
    v75 = td.v75(pref='')
    # Hämta data från v75
    df,_ = v75.förbered_data(extra=True, missing_num=False)  # num hanteras av catboost
    # df = v75.lägg_till_kolumner()

    ###############################################################
    # Några idéer på nya kolumner:
    #  -   ❌ streck/sum(streck för avd) - fungerar inte bra. Nästan alla sum == 100 per avd
    #  a - ✔️ plats(streck)/ant_hästar_i_avd (antal startande hästar i avd)
    #  b - ❌ pris / tot_pris_i_avd - går inte att använda ju! pris är ju loppets 1.pris - samma för all i loppet
    #  c - ✔️ kr / tot_kr_i_avd     rel_kr
    #  d - ✔️ Avståndet till ettan (streck)
    #  e - ✔️ hx_bana samma som bana
    #  f - ✔️ hx_kusk samma som kusk

    # -------------- skapa modeller för backtesting
    L1_modeller, L2_modeller = mod.skapa_modeller()
    
    if st.button('kör'):
        st.write('Införa kör med cv?')
        df_resultat = kör(df, L1_modeller, L2_modeller, cv=False)

if __name__ == "__main__":
    # Skapa v75-instans
    v75 = td.v75(pref='')
    # Hämta data från v75
    # num hanteras av catboost
    df, encoder = v75.förbered_data(extra=True, missing_num=False)

    # -------------- skapa modeller för backtesting
    L1_modeller, L2_modeller = mod.skapa_modeller()

    st.write('Införa kör med cv?')
    df_resultat = kör(df, L1_modeller, L2_modeller, cv=False)
