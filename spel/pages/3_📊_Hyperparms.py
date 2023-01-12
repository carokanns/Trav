#%%
########################################################################################################################
#                               gridsearch för att optimera params                                                     #
########################################################################################################################

import json
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
        
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
import streamlit as st
import sys
import pandas as pd
import numpy as np
import time

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

import pickle

import datetime

from IPython.display import display

sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')

# sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
# import V75_scraping as vs
import typ as tp
import travdata as td

pref=''   # '../'


#%%
###########################################################################
#                       skapa modellerna                                  #
###########################################################################
print('Skapar dict med modeller')
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


def skapa_stack(X_, y):
    X=X_.copy()
    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
        
    stacked_data = pd.DataFrame(columns=meta_features)
    for model_name, typ in L1_modeller.items():
            nr = model_name[2:]
            stacked_data['proba'+nr] = typ.predict(X)

    assert list(stacked_data.columns) == meta_features, f'columns in stacked_data is wrong {list(stacked_data.columns)}'
    assert len(stacked_data) == len(y), f'stacked_data {len(stacked_data)} and y {len(y)} should have same length'
    return stacked_data[meta_features],y   # enbart stack-info

#%% 

def confusion_matrix_graph(y_true, y_pred, title='Confusion matrix'):
    # confusion matrix graph
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # make a graph 
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.set(font_scale=2.0)
    sns.heatmap(cm, annot=True, fmt=".2%", linewidths=.5, square=True, cmap='Blues_r')
    
    # plt.figure(figsize=(10,10))
    #increase font size
    plt.rcParams['font.size'] = 20
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    st.write(fig)
    # st.write(plt.show())
    
# write the scores    
def display_scores(y_true, y_pred, spelade):    
    st.write('AUC',round(roc_auc_score(y_true, y_pred),5),'F1',round(f1_score(y_true, y_pred),5),'Acc',round(accuracy_score(y_true, y_pred),5),'MAE',round(mean_absolute_error(y_true, y_pred),5), '\n', spelade)


def plot_confusion_matrix(y_true, y_pred, typ, fr=0.05, to=0.3, step=0.001):

    #### Först:  hitta ett treshold som tippar ca 2.5 hästar per avd ####
    tresh = 0
    for tresh in np.arange(fr, to, step):
        cost = 12*sum(y_pred > tresh)/len(y_pred)
        if cost < 2.5:
            break
    tresh = round(tresh, 4)
    # print(f'Treshold: {tresh}\n')
    y_pred = (y_pred > tresh).astype(int)
    # confusion_matrix_graph(y_true, y_pred, f'{typ} treshold={tresh}')

    #### Sedan: confusion matrix graph ####
    title = f'{typ} treshold={tresh}'
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots()
    sns.set(font_scale=2.0)
    sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5,
                square=True, cmap='Blues_r')

    # increase font size
    plt.rcParams['font.size'] = 20
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    # plot fig
    
    st.write(fig)

    #### print scores ####
    display_scores(y_true, y_pred, f'spelade per lopp: {round(12 * sum(y_pred)/len(y_pred),4)}' )
#%%
def prepare_for_meta(v75, name):
    df = v75.förbered_data()
    X,y = skapa_stack(df.drop(['y'],axis=1), df.y.copy())
    return X, y


#%%
def gridsearch_meta(v75, meta_name, params, folds=5,randomsearch=True, save=False):
    
    X, y = prepare_for_meta(v75, meta_name)

    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
    assert meta_features == list(X.columns), f'X.columns {list(X.columns)} is not the same as meta_features {meta_features}'
    
    print(meta_name)
    scoring='roc_auc'
    params = None if len(params)==0 else eval(params)
    
    if meta_name == 'knn_meta':
        meta = KNeighborsClassifier(n_jobs=-1)
    elif meta_name == 'ridge':
        meta = RidgeClassifier(random_state=2022)
    elif meta_name == 'lasso':
        scoring = None
        meta = Lasso(random_state=2022)
    elif meta_name == 'rf':
        meta = RandomForestClassifier(n_jobs=-1,random_state=2022)  
    elif meta_name == 'et':
        meta = ExtraTreesClassifier(n_jobs=-1,random_state=2022)  
    elif meta_name == 'lgbm':
        meta = lgb.LGBMClassifier(n_jobs=-1, random_state=2022)
    else:
        assert False, f'{meta_name} is not a valid meta-model'
            
    tscv = TimeSeriesSplit(n_splits=folds)
    
    grid = RandomizedSearchCV(meta, params, cv=tscv.split(X), scoring=scoring,
                              return_train_score=False, refit=True, verbose=5, n_jobs=4)

    # fitting the grid search
    res = grid.fit(X, y)
    d = {'params': res.best_params_, 'AUC': round(res.best_score_,6)}

    if save:
        with open('optimera/params_'+meta_name+'.json', 'w') as f:
            json.dump(d, f)

    return d

# KNN as model
def gridsearch_knn(v75, params, folds=5,randomsearch=True, save=False):  
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_absolute_error
    X,y = prepare_for_KNN(v75)
    
    knn = KNeighborsClassifier(n_jobs=4)
    tscv = TimeSeriesSplit(n_splits=folds)

    grid_params = eval(params)  # str -> dict
    grid = RandomizedSearchCV(knn, grid_params, cv=tscv.split(X), scoring='roc_auc', 
                              return_train_score=False, refit =True, verbose=5, n_jobs=4)
    
    # fitting the grid search
    res = grid.fit(X, y)
    d = {'params': res.best_params_, 'AUC': round(res.best_score_,5)}

    if save:
        with open('optimera/params_'+'knn_model'+'.json', 'w') as f:
            json.dump(d, f)
            
    return d

    
def gridsearch_typ(v75, typ, params, folds=5, save=False):
    """ 
    Sätt upp en gridsearch för att optimera parametrar för typ
    presentera resultat
    spara resultat
    """
    res = create_grid_search(v75, typ, params, randomsearch=True)
    
    d = {'params': res['params'], 'AUC': round(max(res['cv_results']['test-AUC-mean']),5), 'Logloss': round(min(res['cv_results']['test-Logloss-mean']),5)}

    if save:
        with open('optimera/params_'+typ.name+'.json', 'w') as f:
            json.dump(d, f)
    
    return d

def create_grid_search(v75, typ, params, randomsearch=False, verbose=False):
        
    # Läs in NUM_FEATURES.txt och CAT_FEATURES.txt
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()

    # läs in NUM_FEATURES.txt till num_features
    with open(pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
        num_features = f.read().split()
    use_features = cat_features + num_features
    
    if not typ.streck:
        print('remove streck')
        use_features.remove('streck')
        
    if 'cat' in typ.name:
        df,_=v75.förbered_data(extra=True)
        X = typ.prepare_for_model(df.drop(['y'],axis=1))
        y = df.y.copy()
        X = tp.prepare_for_catboost(X)
        # print('cat_features\n', cat_features)
        # num_features = list(X.select_dtypes(include=[np.number]).columns)
        # cat_features = list(X.select_dtypes(include=['object']).columns)
        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'
        
        tscv = TimeSeriesSplit(n_splits=5)

        model = CatBoostClassifier(iterations=500, loss_function='Logloss', eval_metric='AUC',
                                use_best_model=False, early_stopping_rounds=100, verbose=verbose,)
        
        grid = eval(params)  # str -> dict
    
        if randomsearch:
            st.info(f'Randomized search')
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
    elif 'xgb' in typ.name:
        df,_=v75.förbered_data(extra=True)
        X = typ.prepare_for_model(df.drop(['y'],axis=1))
        y = df.y.copy()
        X = tp.prepare_for_xgboost(X, cat_features=cat_features)
        
        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'

        tscv = TimeSeriesSplit(n_splits=5)

        model = xgb.XGBClassifier(**params,
                        #   iterations=iterations,
                        early_stopping_rounds=Typ.EARLY_STOPPING_ROUNDS if X_test is not None else None,
                        objective='binary:logistic', eval_metric='auc')

        grid = eval(params)  # str -> dict

        if randomsearch:
            st.info(f'Randomized search')
            grid_search_result = model.randomized_search(grid,
                                                         X=Pool(
                                                             X[use_features], y, cat_features=cat_features),
                                                         cv=tscv.split(X),
                                                         shuffle=False,
                                                         search_by_train_test_split=False,
                                                         verbose=verbose,
                                                         plot=True)
        else:
            st.info(f'Grid search')
            grid_search_result = model.grid_search(grid,
                                                   X=Pool(
                                                       X[use_features], y, cat_features=cat_features),
                                                   cv=tscv.split(X),
                                                   shuffle=False,
                                                   search_by_train_test_split=False,
                                                   verbose=verbose,
                                                   plot=True)
    
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
    DATA = v75.get_work_df()
    st.info(f'Data loaded in real: {len(DATA)}')
    return DATA
    
if st.session_state['loaded'] == False:
    DATA=load_data()
    st.session_state['loaded'] = True
    
###########################################
# control flow with buttons               #
###########################################

def optimera_model(v75,typ,folds):
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
        st.info('params_'+name+'.json'+' not found')
        params={'params':{'depth':[4], 'parm2': [1,2,3]} , 'AUC': 0, 'Logloss': 999}
        
    opt_params = st.text_area(f'Parametrar att optimera för {name}', params['params'], height=110)
    
    if st.button('run'):
        result = gridsearch_typ(v75,typ,opt_params,folds=folds)
        
        st.write(result)

        elapsed = round(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)

        st.info(f'✔️ {name} optimering done in {minutes}:{seconds}')
        
        st.write(f'res {result["AUC"]} {result["Logloss"] if "Logloss" in result else ""}')
        if result["AUC"] > params["AUC"]:
            with open('optimera/params_'+name+'.json', 'w') as f:
                json.dump(result, f)
            st.success(f'✔️ {name} optimering saved')
            


def optimera_meta(v75, name,folds=5):
    try:
        with open('optimera/params_'+name+'.json', 'r') as f:
            params = json.load(f)
        info = f' AUC = {params["AUC"]}'
        if 'Logloss' in params:
            info += f', Logloss: = {params["Logloss"]}'
        st.info(f"Current values: {info}")
        
        for key, value in params['params'].items():
            params['params'][key] = [value]
    except:
        st.info('params_'+name+'.json'+' not found')
        params = {'params': {'depth': [4], 'parm2': [
            1, 2, 3]}, 'AUC': 0, 'Logloss': 999}

    opt_params = st.text_area(f'params att optimera för {name}', params['params'], height=110)

    if st.button('run'):
        start_time = time.time()
        result = gridsearch_meta(v75, name, opt_params,folds=folds)

        st.write(result)
        
        elapsed = round(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)
        
        st.info(f'✔️ {name} optimering done in {minutes}: {seconds}')

        st.write(f'res {result["AUC"]} {result["Logloss"] if "Logloss" in result else ""}')
        if result["AUC"] > params["AUC"]:
            with open('optimera/params_'+name+'.json', 'w') as f:
                json.dump(result, f)
            st.success(f'✔️ {name} optimering saved')

with buttons:
    folds = st.sidebar.number_input('Folds', 3, 15, 5)
    st.session_state['folds'] = folds
    if st.sidebar.radio('Välj optimering:', ['model', 'meta-model', ]) == 'model':
        st.sidebar.write('---')
        opt = st.sidebar.radio('Optimera model parms', ['cat1L1', 'cat2L1', 'xgb1L1', 'xgb2L1'])
        
        for model_name,typ in L1_modeller.items():
            if opt == typ.name:
                optimera_model(st.session_state.v75,typ,folds=folds)
                break
    else:        
        st.sidebar.write('---')
        st.warning('Meta-models not implemented yet')
        
    st.sidebar.write('---')
    if st.sidebar.button('start allover'):
        st.session_state.clear()
        
print('END', datetime.datetime.now().strftime("%H.%M.%S"))
