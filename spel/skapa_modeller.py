#%%
import logging
import numpy as np
import pandas as pd
from IPython.display import display
import V75_scraping as vs
import travdata as td
import typ as tp

import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')

#%%

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

pref = ''   # '../'

###################################################################################

def skapa_modeller():  

    logging.info('skapa_modeller: Initierar dict med modeller')

    # skapar dict med modeller
    modell_dict = {'cat1': {'#hästar': False, '#motst': 3, 'motst_diff': True, 'streck': False},
                'cat2': {'#hästar': True,  '#motst': 3, 'motst_diff': True, 'streck': True},
                'xgb1': {'#hästar': False, '#motst': 3, 'motst_diff': True, 'streck': False},
                'xgb2': {'#hästar': True,  '#motst': 3, 'motst_diff': True, 'streck': True}
                }

    L1_modeller = dict()
    L2_modeller = dict()

    for key, value in modell_dict.items():
        L1_key = key + 'L1'
        model = tp.Typ(L1_key, value['#hästar'], value['#motst'], value['motst_diff'], value['streck'])
        L1_modeller[L1_key] = model

        L2_key = key + 'L2'
        model = tp.Typ(L2_key, value['#hästar'], value['#motst'], value['motst_diff'], value['streck'])
        L2_modeller[L2_key] = model

    print('keys and names i modeller')
    # print keys in dict modeller
    for key, value in L1_modeller.items():
        assert key == value.name, "key and value.name should be the same in modeller"
        logging.info(f'skapa_modeller: {key} klar')

    print('keys and names i meta_modeller')
    for key, value in L2_modeller.items():
        assert key == value.name, "key and value.name should be the same in meta_modeller"
        logging.info(f'skapa_modeller: {key} klar')
    
    return L1_modeller, L2_modeller    

# make a unit test
def test_skapa_modeller(): 


    logging.basicConfig(level=logging.DEBUG, filemode='w', filename='v75_debug.log', force=True,
                    encoding='utf-8', format='DBG:' '%(asctime)s - %(levelname)s - %(message)s')

    logging.debug('Startar test_skapa_modeller')
    L1_modeller, L2_modeller = skapa_modeller()
    assert len(L1_modeller) == 4, "L1_modeller should have 4 elements"
    assert len(L2_modeller) == 4, "L2_modeller should have 4 elements"
    for key, value in L1_modeller.items():
        assert key == value.name, "key and value.name should be the same in modeller"
    for key, value in L2_modeller.items():
        assert key == value.name, "key and value.name should be the same in meta_modeller"
test_skapa_modeller()


def create_L2_input(X_, L1_modeller, L1_features):
    logging.info('create_L2_input: Startar create_L2_input')
    
    X = X_.copy()

    X = X.reset_index(drop=True)
    proba_data = pd.DataFrame()
    for model_name, typ in L1_modeller.items():
        logging.info(f'create_L2_input: predict med {model_name}')
        proba_data['proba_'+model_name] = typ.predict(X, L1_features)

    proba_data = proba_data.reset_index(drop=True)

    ####### kolla om det finns NaNs i X eller proba_data
    X_na = X.isna()
    X_missing = X[X_na.any(axis=1)]
    proba_data_na = proba_data.isna()
    proba_data_missing = proba_data[proba_data_na.any(axis=1)]

    if X_missing.shape[0] > 0:
        logging.warning(f'NaNs i X {X_missing}')
        print('NaNs i X', X_missing)

    if proba_data_missing.shape[0] > 0:
        print(f'NaNs i proba_data_missing {proba_data_missing}')
        logging.warning(f'NaNs i proba_data_missing {proba_data_missing}')
    ####### slutkollat

    assert X.shape[0] == proba_data.shape[0], f'X.shape[0] != proba_data.shape[0] {X.shape[0]} != {proba_data.shape[0]}'

    assert len(proba_data) == len(X), f'proba_data {len(proba_data)} is not the same length as X {len(X)} innan concat'
    assert 'bana' in X.columns, f'bana not in X.columns {X.columns} innan concat'
    
    logging.info('create_L2_input: concat X and proba_data')
    X = pd.concat([X, proba_data], axis=1, ignore_index=False) # eftersom index är kolumn-namn (axis=1)
    assert len(proba_data) == len(X), f'proba_data {len(proba_data)} is not the same length as X {len(X)} efter concat'
    assert 'bana' in X.columns, f'bana not in X.columns {X.columns} efter concat'

    assert X.shape[0] == proba_data.shape[0], f'X.shape[0] != proba_data.shape[0] {X.shape[0]} != {proba_data.shape[0]}'
    
    logging.info('create_L2_input: Är klar')
    return X
