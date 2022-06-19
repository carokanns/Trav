#%%
########################################################################################################################
#                               gridsearch för att optimera params                                                     #
########################################################################################################################

import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
import streamlit as st
import sys
import pandas as pd
import numpy as np

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
#              name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck
typ6 = tp.Typ('typ6', True,       True, False,     0,      False,          0,            False,    True,  pref)
typ1 = tp.Typ('typ1', False,      True, False,     2,      True,           2,            True,     False, pref)
typ9 = tp.Typ('typ9', True,       True, True,      2,      True,           2,            True,     True,  pref)
# typ16 = tp.Typ('typ16', True,      True, True,      2, True,           2,            False,    True,  pref)

typer = [typ6, typ1, typ9]  # , typ16]

    

#%%
# Kelly-värde baserat på streck omvandlat till odds
def kelly(proba, streck, odds):  # proba = prob winning, streck i % = streck
    with open(pref+'rf_streck_odds.pkl', 'rb') as f:
        rf = pickle.load(f)

    if odds is None:
        o = rf.predict(streck.copy())
    else:
        o = rf.predict(streck.copy())

    # for each values > 40 in odds set to 1
    o[o > 40] = 1
    return (o*proba - (1-proba))/o

def skapa_stack(X_, y):
    X=X_.copy()
    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
        
    stacked_data = pd.DataFrame(columns=meta_features)
    for typ in typer:
            nr = typ.name[3:]
            stacked_data['proba'+nr] = typ.predict(X)
            stacked_data['kelly' + nr] = kelly(stacked_data['proba' + nr], X[['streck']], None)

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
    sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r')
    
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
    st.write('AUC',roc_auc_score(y_true, y_pred),'F1',f1_score(y_true, y_pred),'Acc',accuracy_score(y_true, y_pred),'MAE',mean_absolute_error(y_true, y_pred), '\n', spelade)


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
def prepare_for_KNN(v75):  # as model
    X, y = v75.förbered_data()
    X.drop(['avd','datum','streck'], axis=1, inplace=True)
    
    cat_features = list(X.select_dtypes(include=['object']).columns)
    num_features = list(X.select_dtypes(include=[np.number]).columns)
    
    X[cat_features] = X[cat_features].fillna('missing')
    X[num_features] = X[num_features].fillna(0)
    
    X[cat_features] = X[cat_features].astype('category')

    # make categorical features numeric
    for col in cat_features:
        X[col] = X[col].cat.codes
    
    return X, y


def prepare_for_meta(v75, name):
    X,y = v75.förbered_data()
    X,y = skapa_stack(X, y)
    return X, y


#%%
def gridsearch_meta(v75, meta_name, params, randomsearch=True):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    
    X, y = prepare_for_meta(v75, meta_name)

    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
    assert meta_features == list(X.columns), f'X.columns {list(X.columns)} is not the same as meta_features {meta_features}'
    
    print(meta_name)
    scoring='roc_auc'
    params = eval(params)
    if meta_name == 'knn_meta':
        meta = KNeighborsClassifier(n_jobs=-1)
    elif meta_name == 'ridge':
        meta = RidgeClassifier(random_state=2022)
    elif meta_name == 'lasso':
        meta = Lasso(random_state=2022)
        scoring='neg_mean_squared_error'
    elif meta_name == 'rf':
        meta = RandomForestClassifier(n_jobs=-1,random_state=2022)  
    else:   
        assert False, f'{meta_name} is not a valid meta-model'
            
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid = RandomizedSearchCV(meta, params, cv=tscv.split(X), scoring=scoring,
                              return_train_score=False, refit=True, verbose=5, n_jobs=4)

    # fitting the grid search
    res = grid.fit(X, y)
    d = {'params': res.best_params_, 'AUC': round(res.best_score_,5)}

    with open('optimera/params_'+meta_name+'.json', 'w') as f:
        json.dump(d, f)

    return d

# KNN as model
def gridsearch_knn(v75, params, randomsearch=True):  
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_absolute_error
    X,y = prepare_for_KNN(v75)
    
    knn = KNeighborsClassifier(n_jobs=4)
    tscv = TimeSeriesSplit(n_splits=5)

    grid_params = eval(params)  # str -> dict
    grid = RandomizedSearchCV(knn, grid_params, cv=tscv.split(X), scoring='roc_auc', 
                              return_train_score=False, refit =True, verbose=5, n_jobs=4)
    
    # fitting the grid search
    res = grid.fit(X, y)
    d = {'params': res.best_params_, 'AUC': round(res.best_score_,4)}

    with open('optimera/params_'+'knn_model'+'.json', 'w') as f:
        json.dump(d, f)
        
    return d

    
def gridsearch_typ(v75, typ, params):
    """ 
    Sätt upp en gridsearch för att optimera parametrar för typ
    presentera resultat
    spara resultat
    """
    res = create_grid_search(v75, typ, params, randomsearch=True)
    
    d = {'params': res['params'], 'AUC': max(res['cv_results']['test-AUC-mean']), 'Logloss': min(res['cv_results']['test-Logloss-mean'])}

    with open('optimera/params_'+typ.name+'.json', 'w') as f:
        json.dump(d, f)
    return d

def create_grid_search(v75, typ, params, randomsearch=False, verbose=False):
    X,y=v75.förbered_data()
    
    # remove features
    X = typ.prepare_for_model(X)
    if not typ.streck:
        X.drop('streck', axis=1, inplace=True)

    X, cat_features = tp.prepare_for_catboost(X, remove=False)
    print('cat_features\n', cat_features)
    num_features = list(X.select_dtypes(include=[np.number]).columns)
    cat_features = list(X.select_dtypes(include=['object']).columns)
    assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'
    
    tscv = TimeSeriesSplit(n_splits=5)

    model = CatBoostClassifier(iterations=500, loss_function='Logloss', eval_metric='AUC',
                               use_best_model=False, early_stopping_rounds=100, verbose=verbose,)
    
    grid = eval(params)  # str -> dict
    
    if randomsearch:
        st.info(f'Randomized search {grid}')
        grid_search_result = model.randomized_search(grid,
                                           X=Pool(X, y, cat_features=cat_features),
                                           cv=tscv.split(X),
                                           shuffle=False,
                                           search_by_train_test_split=False,
                                           verbose=verbose,
                                           plot=True)
    else:
        st.info(f'Grid search {grid}')
        grid_search_result = model.grid_search(grid,
                                            X=Pool(X, y, cat_features=cat_features),
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

top.header('optimera')

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

def optimera_model(v75,typ):
    name= 'knn_model' if typ=='knn' else typ.name
    st.info(name)
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
        
    params = st.text_area(f'params att optimera för {name}', params['params'], height=110)
    
    if st.button('run'):
        if typ=='knn':
            result = gridsearch_knn(v75, params)
        else:
            result = gridsearch_typ(v75,typ,params)
        
        st.write(result)
        st.success(
            f'✔️ {name} optimering done {datetime.datetime.now().strftime("%H.%M.%S")}')


def optimera_meta(v75, name):
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

    params = st.text_area(f'params att optimera för {name}', params['params'], height=110)

    if st.button('run'):
    
        result = gridsearch_meta(v75, name, params)

        st.write(result)

        st.success(
            f'✔️ {name} optimering done {datetime.datetime.now().strftime("%H.%M.%S")}')


with buttons:
    if st.sidebar.radio('Välj optimering:', ['model', 'meta', ]) == 'model':
        st.sidebar.write('---')
        opt = st.sidebar.radio('Optimera model parms', ['typ6', 'typ1', 'typ9', 'knn model'])
        if opt=='knn model':
            optimera_model(st.session_state.v75, 'knn')
        for typ in typer:
            if opt == typ.name:
                optimera_model(st.session_state.v75,typ)
                break
    else:        
        st.sidebar.write('---')
        meta = st.sidebar.radio('Optimera meta parms', ['rf', 'ridge', 'lasso', 'knn_meta'])
        if st.session_state.meta:
            optimera_meta(st.session_state.v75,meta)
            # df = st.session_state.df
            # stacked_data = TimeSeries_learning(df, typer, n_splits=3, meta_fraction=0.2, meta='rf', save=True, learn_models=True)
            st.success(f'✔️ {meta} optimering done')
            # st.dataframe(load_data())
            
    st.sidebar.write('---')
    if st.sidebar.button('start allover'):
        st.session_state.clear()
        
print('END', datetime.datetime.now().strftime("%H.%M.%S"))
