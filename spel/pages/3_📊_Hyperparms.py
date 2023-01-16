#%%
########################################################################################################################
#                               gridsearch för att optimera params                                                     #
########################################################################################################################

import json
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
import streamlit as st
import sys
import pandas as pd
import numpy as np
import time

import logging
    
# %%

logging.basicConfig(level=logging.DEBUG, filemode='w' , filename='v75.log', force=True, encoding='utf-8', format='Hyperparms:' '%(asctime)s - %(levelname)s - %(message)s')
logging.info('Startar')
   
logging.debug("Detta är ett test debug-mess.")
logging.warning("Detta är ett test warning-mess.")
logging.error("Detta är ett test error-mess.")
logging.critical("Detta är ett test critical-mess.")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

import pickle

import datetime

from IPython.display import display

sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')

import typ as tp
import travdata as td

pref=''   # '../'


#%%

###########################################################################
#                       skapa modellerna                                  #
###########################################################################
print('Skapar dict med modeller')
#TODO: Egen funktion i py-fil
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
    model = tp.Typ(L1_key, value['#hästar'], value['#motst'],
                   value['motst_diff'], value['streck'])
    L1_modeller[L1_key] = model

    L2_key = key + 'L2'
    model = tp.Typ(L2_key, value['#hästar'], value['#motst'],
                   value['motst_diff'], value['streck'])
    L2_modeller[L2_key] = model

print('keys and names i modeller')
# print keys in dict modeller
for key, value in L1_modeller.items():
    assert key == value.name, "key and value.name should be the same in modeller"
    print(key)

print('keys and names i meta_modeller')
for key, value in L2_modeller.items():
    assert key == value.name, "key and value.name should be the same in meta_modeller"
    print(key)


#%%
def create_L2_input(X_, L1_features) :
    #TODO: Egen funktion i py-fil
    X = X_.copy()
    
    X = X.reset_index(drop=True)
    proba_data = pd.DataFrame()
    for model_name, typ in L1_modeller.items():
        proba_data['proba_'+model_name] = typ.predict(X, L1_features)  
    
    proba_data = proba_data.reset_index(drop=True)

        
    X_na = X.isna()
    X_missing = X[X_na.any(axis=1)]
    proba_data_na = proba_data.isna()
    proba_data_missing = proba_data[proba_data_na.any(axis=1)]
    
    if X_missing.shape[0] > 0:
        print('NaNs i X', X_missing)
    
    if proba_data_missing.shape[0] > 0:
        print('NaNs i proba_data_missing', proba_data_missing)
    
        
    assert X.shape[0] == proba_data.shape[0], f'X.shape[0] != proba_data.shape[0] {X.shape[0]} != {proba_data.shape[0]}'
    
    assert len(proba_data) == len(X), f'proba_data {len(proba_data)} is not the same length as X {len(X)} innan concat'
    assert 'bana' in X.columns, f'bana not in X.columns {X.columns} innan concat'
    X = pd.concat([X, proba_data], axis=1, ignore_index=False) # eftersom index är kolumn-namn (axis=1)   
    assert len(proba_data) == len(X), f'proba_data {len(proba_data)} is not the same length as X {len(X)} efter concat'
    assert 'bana' in X.columns, f'bana not in X.columns {X.columns} efter concat'
    
    assert X.shape[0] == proba_data.shape[0], f'X.shape[0] != proba_data.shape[0] {X.shape[0]} != {proba_data.shape[0]}'
    return X   

def gridsearch_typ(typ, params, proba_kolumner=[], folds=5, save=False):
    """ 
    TODO: Ta bort alla onödiga print
    Sätt upp en gridsearch för att optimera parametrar för typ
    presentera resultat
    spara resultat
    """
    global DATA
    try :
        _=type(DATA)
    except:
        print('DATA is not defined')
        DATA = None
            
    # Läs in NUM_FEATURES.txt och CAT_FEATURES.txt
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()

    # läs in NUM_FEATURES.txt till num_features
    with open(pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
        num_features = f.read().split()
    L1_features = cat_features + num_features
    
    if DATA is None:
        print('DATA is None, load_data')
        DATA=load_data()
        
    df = DATA.copy()
    X = typ.prepare_for_model(df.drop(['y'], axis=1))
    y = df.y.copy()

    assert X.shape[0] == DATA.shape[0], 'X.shape[0] != DATA.shape[0]'
    assert X[cat_features].isna().sum().sum()==0, 'cat_features contains NaN i början gridsearch_typ'
 
    assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'

    use_features = L1_features
    if len(proba_kolumner) > 0:
        # Kör L2-modeller
        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features before create_L2_input'
      
        X = create_L2_input(X, L1_features)
     
        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features after create_L2_input'
    
        use_features += proba_kolumner
   
        
    assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'
    
    res = do_grid_search(X, y, typ, params, use_features, cat_features, folds=folds, randomsearch=True)  
    
    print('reultat från gridsearch',res)
    if save:
        with open('optimera/params_'+typ.name+'.json', 'w') as f:
            json.dump(res, f)
    
    return res

def do_grid_search(X,y, typ, params, use_features, cat_features, folds=5,randomsearch=False, verbose=False):
     
    tscv = TimeSeriesSplit(n_splits=folds)
    grid = eval(params)  # str -> dict

    if not typ.streck:
        print('remove streck')
        use_features.remove('streck')
        
    if 'cat' in typ.name:
        X = tp.prepare_for_catboost(X)
        
        model = CatBoostClassifier(iterations=500, loss_function='Logloss', eval_metric='AUC',
                                use_best_model=False, early_stopping_rounds=200, verbose=verbose,)
        if randomsearch:
            st.info(f'Randomized search {typ.name}')
            grid_search_result = model.randomized_search(grid,
                                            X=Pool(X[use_features], y, cat_features=cat_features),
                                            cv=tscv.split(X),
                                            shuffle=False,
                                            search_by_train_test_split=False,
                                            verbose=verbose,
                                            plot=True)           
        else:
            st.info(f'Grid search')
            grid_search_result = model.grid_search(grid,
                                                X=Pool(X[use_features], y, cat_features=cat_features),
                                                cv=tscv.split(X),
                                                shuffle=False,
                                                search_by_train_test_split=False,
                                                verbose=verbose,
                                                plot=True)
            
        best_params = grid_search_result['params']
        AUC = max(grid_search_result['cv_results']['test-AUC-mean'])
        
        Logloss = min(grid_search_result['cv_results']['test-Logloss-mean'])
        ix = np.argmax(grid_search_result['cv_results']['test-AUC-mean'])

        assert AUC==grid_search_result['cv_results']['test-AUC-mean'][ix], f'AUC != mean_test_roc_auc[{ix}]'
        grid_search_result = {'params': best_params, 'AUC': round(AUC,5), 'Logloss': round(Logloss,5)}    
            
    elif 'xgb' in typ.name:
        # xgb_encoder till ENC
        with open(pref+'xgb_encoder.pkl', 'rb') as f:
            ENC = pickle.load(f)
            
        X, ENC = tp.prepare_for_xgboost(X, encoder=ENC)
        
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=2023)

        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features inna xgb gridsearch'
        if randomsearch:
            st.info(f'Randomized search {typ.name}')
            grid_search_result = RandomizedSearchCV(model, param_distributions=grid,
                                                    cv=tscv.split(X[use_features]),
                                                    scoring=['roc_auc','neg_log_loss'],
                                                    random_state=2023,
                                                    refit='roc_auc',
                                                    n_jobs=-1,
                                                    verbose=1)
            grid_search_result.fit(X[use_features], y)
            
        else:
            st.info(f'Grid search')
            grid_search_result = model.grid_search(grid,
                                                   X=Pool(
                                                   X[use_features], y, cat_features=cat_features),
                                                   cv=tscv.split(X),
                                                   search_by_train_test_split=False,
                                                   verbose=verbose,
                                                   plot=True)
            
        best_params = grid_search_result.best_params_
        AUC = grid_search_result.best_score_
        ix = grid_search_result.best_index_
        Logloss = -grid_search_result.cv_results_['mean_test_neg_log_loss'][ix]
        assert AUC==grid_search_result.cv_results_['mean_test_roc_auc'][ix], f'AUC != mean_test_roc_auc[{ix}]'
        
        grid_search_result = {'params': best_params, 'AUC': round(AUC,5), 'Logloss': round(Logloss,5)}    
    else:
        raise ValueError('typ.name must include cat or xgb')
    
    print('---------------------------------------')
    display(AUC, Logloss)
    print('---------------------------------------')
    display(best_params)
    print('---------------------------------------')
    print()
    return grid_search_result

#%%

############################################
#   Init session_state                     #
############################################
if 'loaded' not in st.session_state:
    print('init session_state')
    st.session_state['loaded'] = False
    st.session_state['model'] = True
    st.session_state['meta'] = True
    st.session_state['välj'] = False
    DATA = None
    
top = st.container()
buttons = st.container()

top.header('📊 Optimera hyperparametrar')

##### @st.cache(suppress_st_warning=True)
def load_data(): 
    global DATA
    v75 = td.v75(pref=pref)
    st.session_state['v75'] = v75
    DATA,_ = v75.förbered_data(extra=True)
    st.info(f'Total Data loaded: {len(DATA)}')
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()
    assert DATA[cat_features].isna().sum().sum()==0, 'cat_features contains NaN'
    
    return DATA
    
if st.session_state['loaded'] == False:
    DATA=load_data()
    st.session_state['loaded'] = True
    
###########################################
# control flow with buttons               #
###########################################
    
def optimera_model(typ, folds, proba_kolumner = []):
    name= typ.name
    st.info(name)
    start_time = time.time()
    try:
        with open('optimera/params_'+name+'.json', 'r') as f:
            params = json.load(f)
            
        info = f' AUC = {params["AUC"]}'
        if 'Logloss' in params:
            info += f', Logloss = {params["Logloss"]}'
            
        st.info(f"Current values: {info}")
        
        # loop over all keys and items in prms
        for key, value in params['params'].items():
            params['params'][key] = [value]
            
    except:
        st.info('params_'+name+'.json'+' not found start a new one')
        params={'params':{'depth':[4], 'parm2': [1,2,3]} , 'AUC': 0, 'Logloss': 999}
        
    opt_params = st.text_area(f'Parametrar att optimera för {name}', params['params'], height=110)
    
    if st.button('run'):
        
        result = gridsearch_typ(typ,opt_params,proba_kolumner=proba_kolumner,folds=folds)
        
        st.write(result)

        elapsed = round(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)

        st.info(f'✔️ {name} optimering done in {minutes}:{seconds}')
        
        st.write(f'res {result["AUC"]} {result["Logloss"] if "Logloss" in result else ""}')
        if result["AUC"] > params["AUC"]:
            with open(pref+'optimera/params_'+name+'.json', 'w') as f:
                json.dump(result, f)
            st.success(f'✔️ {name} optimering saved')
            
with buttons:
    folds = st.sidebar.number_input('Folds', 3, 15, 5)
    st.session_state['folds'] = folds
    if st.sidebar.radio('Välj optimering:', ['L1-model', 'L2-model', ]) == 'L1-model':
        st.sidebar.write('---')
        opt = st.sidebar.radio('Optimera L1 parms', ['cat1L1', 'cat2L1', 'xgb1L1', 'xgb2L1'])
        
        for L1_name,L1_typ in L1_modeller.items():
            if opt == L1_typ.name:
                optimera_model(L1_typ,folds=folds)
                break
    else:        
        st.sidebar.write('---')
        opt = st.sidebar.radio('Optimera L2 parms', ['cat1L2', 'cat2L2', 'xgb1L2', 'xgb2L2'])
        proba_kolumner = ['proba_cat1L1', 'proba_cat2L1', 'proba_xgb1L1', 'proba_xgb2L1']
        for model_name, L2_typ in L2_modeller.items():
            if opt == L2_typ.name:
                optimera_model(L2_typ, folds=folds, proba_kolumner=proba_kolumner)
                break
        
    st.sidebar.write('---')
    if st.sidebar.button('start allover'):
        st.session_state.clear()
        
print('END', datetime.datetime.now().strftime("%H.%M.%S"))
