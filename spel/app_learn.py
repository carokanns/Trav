from category_encoders import TargetEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
import streamlit as st
import sys
import time
import json
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

import pickle

from IPython.display import display
    
import concurrent.futures

sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')

# sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs

import typ as tp

pref=''   # '../'

###################################################################################
#%%
# skapa modeller
#           name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck
typ6 = tp.Typ('typ6', True,       True, False,     0,  False,          0,            False,    True,  pref)
typ1 = tp.Typ('typ1', False,      True, False,     2,  True,           2,            True,     False, pref)
typ9 = tp.Typ('typ9', True,       True, True,      2,  True,           2,            True,     True,  pref)
# typ16 = tp.Typ('typ16', True,      True, True,      2, True,           2,            False,    True,  pref)

typer = [typ6, typ1, typ9]  # load a file with pickl

###################################################################################

#%%
################################################
#              Web scraping                    #
################################################

def v75_scraping():
    df = vs.v75_scraping(history=True, resultat=True, headless=True)
    
    for f in ['häst','bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
    return df

def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    df.drop(['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
            'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'], axis=1, inplace=True)
    if remove_mer:
        df.drop(remove_mer, axis=1, inplace=True)

    return df


###############################################
#              LEARNING                       #
###############################################

def förbered(df,meta_fraction):
    # Följande datum saknar avd==5 och kan inte användas
    saknas = ['2015-08-15', '2016-08-13', '2017-08-12']
    df = df[~df.datum.isin(saknas)]
    X = df.copy()
    X.drop('plac', axis=1, inplace=True)
    
    # läs in FEATURES.txt
    with open(pref+'FEATURES.txt', 'r',encoding='utf-8') as f:    
        features = f.read().splitlines()
     
    X=X[features]
    
    assert len(features) == len(X.columns), f'features {len(features)} and X.columns {len(X.columns)} are not the same length'   
    assert set(features) == set(X.columns), f'features {set(features)} and X.columns {set(X.columns)} are not the same'
    
    y = (df.plac == 1)*1   # plac 1 eller 0

    for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        X[f] = X[f].str.lower()

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    if meta_fraction==0: 
        # no meta data
        return X,y,None,None
    
    # use a fraction for meta data
    meta_antal = int(len(X.datum.unique())*meta_fraction)
    meta_datum = X.datum.unique()[-meta_antal:]

    X_val = X.loc[X.datum.isin(meta_datum)]
    y_val = y[X_val.index]
    X=X.loc[~X.datum.isin(meta_datum)]
    y=y.loc[X.index]
    return X, y, X_val, y_val

def concat_data(df_all, df_ny,save=True):
    df_ny=df_ny[df_all.columns]
    df_all = pd.concat([df_all, df_ny])
    # remove duplicates
    all_shape=df_all.shape
    
    df_all = df_all.drop_duplicates(subset=['datum','avd','häst'])
    assert df_all.shape[0]+90 > all_shape[0], f'{df_all.shape[0]+90} should be more than {all_shape[0]}'
    assert df_all.shape[1] == all_shape[1], f'{df_all.shape[1]} should be {all_shape[1]}'
    if save==True:
        df_all.to_csv(pref+'all_data.csv',index=False)
    return df_all

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

# TimeSeriesSplit learning models
def TimeSeries_learning(df_ny_, typer, n_splits=5,meta_fraction=0.2, meta='rf', save=True, learn_models=True):
    """
    Skapar en stack av X_test och y_test från alla typer. Används som input till meta_model.
        - learn_models=True betyder att vi både gör en learning och skapar en stack
        - learn_models=False betyder att vi bara skapar en stack och då har save ingen funktion
    """
    
    df_all = pd.read_csv(pref+'all_data.csv')
        
    if df_ny_ is not None:
        df_ny = df_ny_.copy()
        df_all = concat_data(df_all.copy(), df_ny, save=True)
    
    X, y, _, _ = förbered(df_all, meta_fraction=meta_fraction)
    # print('X shape',X.shape)
    st.info(f'sista datum {X.datum.iloc[-1]} resp  {df_all.datum.iloc[-1]}')
    
    ts = TimeSeriesSplit(n_splits=n_splits)
    # läs in meta_features
    with open(pref+'META_FEATURES.txt', 'r',encoding='utf-8') as f:    
        meta_features = f.read().splitlines()
        
    stacked_data=pd.DataFrame(columns=meta_features + ['y'])
    
    my_bar = st.progress(0)   

    step=1/(n_splits*len(typer))-0.0000001
    steps=0.0
    # my_bar = st.progress(0)
    for enum,(train_index, test_index) in enumerate(ts.split(X,y)):
        print('shape of X', X.shape, 'shape of X_train', X.iloc[train_index].shape, 'shape of X_test', X.iloc[test_index].shape)
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        temp_df = pd.DataFrame()
        temp_df['y'] =  y_test
        for typ in typer:
            steps += step
            # progress bar continues to complete from 0 to 100
            my_bar.progress(steps)
            
            if learn_models:
                with open('optimera/params_'+typ.name+'.json', 'r') as f:
                    params = json.load(f)

                params = params['params']
                cbc = typ.learn(X_train, y_train, X_test, y_test,params=params,iterations=100,save=save)
                
            nr = typ.name[3:]
            this_proba=typ.predict(X_test)
            
            temp_df['proba'+nr] = this_proba

            this_kelly=kelly(this_proba, X_test[['streck']], None)
            temp_df['kelly' + nr] = this_kelly
        
        stacked_data = pd.concat([stacked_data, temp_df],ignore_index=True)
        stacked_data.y = stacked_data.y.astype(int)
        
    ###############################################################################    
    #         Make sure that the meta features are in the right order             #
    ###############################################################################
   
    stacked_data = stacked_data[meta_features+['y']]

    my_bar.progress(1.0)
    # my_bar.empty()
    
    # step 2: learn models on all of X - what iteration to use?
    st.write('Full X learning')
    my_bar2 = st.progress(0)
    step = 1/(1+len(typer)) - 0.0000001
    steps = 0.0

    for typ in typer:
        steps += step
        my_bar2.progress(steps)
        if learn_models:
            with open('optimera/params_'+typ.name+'.json', 'r') as f:
                params = json.load(f)
                
            params = params['params']   
            cbc = typ.learn(X, y, None, None, iterations=100, params=params,save=save)
     
    my_bar2.progress(1.0)
    # my_bar.empty()

    # step 3: learn meta model
    steps += step
    my_bar2.progress(steps)
    
    _,_,_,_= learn_meta_models(stacked_data[meta_features], stacked_data['y'], alpha=0.001,
                              n_estimators=360,  max_iter=55, max_depth=5, class_weight=None)

    my_bar2.progress(1.0)
    # my_bar2.empty()

    return stacked_data[meta_features + ['y']]

def skapa_stack_learning(X_, y, save=True):
    # För validate
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

##### RidgeClassifier (meta model) #####
def learn_meta_ridge_model(X, y, alpha=1, class_weight='balanced', save=True):
    from sklearn.linear_model import RidgeClassifier

    ridge_model = RidgeClassifier(
        alpha=alpha, class_weight=class_weight, random_state=2022)
    ridge_model.fit(X, y)
    # pickle save stacking
    if save:
        with open(pref+'modeller/meta_ridge_model.model', 'wb') as f:
            pickle.dump(ridge_model, f)

    return ridge_model

##### RandomForestClassifier (meta model) #####
def learn_meta_rf_model(X, y, n_estimators=100, max_depth=None, class_weight='balanced', save=True):
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=6, class_weight=class_weight, random_state=2022)
    rf_model.fit(X, y)
    # pickle save stacking
    if save:
        with open(pref+'modeller/meta_rf_model.model', 'wb') as f:
            pickle.dump(rf_model, f)

    return rf_model

##### LassoClassifier (meta model) #####
def learn_meta_lasso_model(X, y, alpha=1, max_iter=1000,  save=True):
    from sklearn.linear_model import Lasso

    lasso_model = Lasso(alpha=alpha, max_iter=max_iter, random_state=2022)
    lasso_model.fit(X, y)
    # pickle save stacking
    if save:
        with open(pref+'modeller/meta_lasso_model.model', 'wb') as f:
            pickle.dump(lasso_model, f)

    return lasso_model

##### KNeighborsClassifier (meta model) #####
def learn_meta_knn_model(X, y, n_jobs=6, save=True):
    from sklearn.neighbors import KNeighborsClassifier

    knn_model = KNeighborsClassifier(algorithm = 'auto', leaf_size = 1, metric = 'chebyshev', n_neighbors = 600, p = 1, weights = 'distance', n_jobs=n_jobs)
    knn_model.fit(X, y)
    # pickle save stacking
    if save:
        with open(pref+'modeller/meta_knn_model.model', 'wb') as f:
            pickle.dump(knn_model, f)

    return knn_model


def learn_meta_models(X, y, alpha=1, n_estimators=100, max_iter=1000, max_depth=None, class_weight='balanced', save=True):
    """ all meta models will be fitted on X and y """
    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
    assert meta_features == list(X.columns), f'X.columns {list(X.columns)} is not the same as meta_features {meta_features}'    
    RidgeClassifier =  learn_meta_ridge_model(X, y, alpha=alpha, class_weight=class_weight, save=save)
    RandomForestClassifier = learn_meta_rf_model(X, y, n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, save=save)
    Lasso = learn_meta_lasso_model(X, y, alpha=alpha, max_iter=max_iter, save=save)
    Knn   = learn_meta_knn_model(X, y, save=save)
    return RidgeClassifier, RandomForestClassifier, Lasso, Knn
    
#%% 
##############################################################
#                     VALIDATE                               #
##############################################################

def predict_meta_ridge_model(X, ridge_model=None):
    if ridge_model is None:
        with open(pref+'modeller/meta_ridge_model.model', 'rb') as f:
            ridge_model = pickle.load(f)

    return ridge_model._predict_proba_lr(X)

def predict_meta_rf_model(X, rf_model=None):
    if rf_model is None:
        with open(pref+'modeller/meta_rf_model.model', 'rb') as f:
            rf_model = pickle.load(f)
            
    return rf_model.predict_proba(X)

def predict_meta_lasso_model(X, lasso_model=None):
    if lasso_model is None:
        with open(pref+'modeller/meta_lasso_model.model', 'rb') as f:
            lasso_model = pickle.load(f)

    return lasso_model.predict(X)

def predict_meta_knn_model(X, knn_model=None):
    if knn_model is None:
        with open(pref+'modeller/meta_knn_model.model', 'rb') as f:
            knn_model = pickle.load(f)

    return knn_model.predict_proba(X)

def predict_meta_model(X, meta_model=None):
    
    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
    assert meta_features == list(X.columns), f'X.columns {list(X.columns)} is not the same as meta_features {meta_features}'

    if meta_model == 'ridge':
        return predict_meta_ridge_model(X)[:, 1]
    elif meta_model == 'rf':
        return predict_meta_rf_model(X)[:, 1]
    elif meta_model == 'lasso':
        return predict_meta_lasso_model(X)
    elif meta_model == 'knn':
        return predict_meta_knn_model(X)[:, 1]
    elif meta_model == None:
        assert False, 'ingen meta_model angiven'
    else:
        return meta_model.predict(X)


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

    
def validate(drop=[]):
    st.info('skall inte köras efter "Final Learning" - vi sparar modeller baserade på mindra data')
    df_all = pd.read_csv(pref+'all_data.csv')
    print('sista datum',df_all.datum.iloc[-1])
    
    _, _, X_val, y_val = förbered(df_all, meta_fraction=0.2)
    
    # create the stack from validation data
    
    stacked_meta_val, y_val = skapa_stack_learning(X_val, y_val)
    stacked_meta_val = stacked_meta_val.drop(drop, axis=1)
    
    ##############################################################
    #                          Meta models                       #
    ##############################################################
    
    y_true = y_val.values
    y_pred_rf = predict_meta_model(stacked_meta_val, meta_model='rf')  # default
    
    placeholder = st.info('förbereder rf plot')
    plot_confusion_matrix(y_true, y_pred_rf, 
                        'rf', fr=0.1, to=0.5, step=0.0001)
    placeholder.empty()

    st.write('\n')
    placeholder = st.info('förbereder knn plot')
    plot_confusion_matrix(y_true, predict_meta_model(stacked_meta_val, meta_model='knn'),
                          'knn', fr=0.01, to=0.5, step=0.0001)
    placeholder.empty()

    st.write('\n')
    placeholder = st.info('förbereder ridge plot')
    plot_confusion_matrix(y_true, predict_meta_model(stacked_meta_val, meta_model='ridge'), 
                        'ridge', fr=0.2, to=0.5, step=0.0001)
    placeholder.empty()
    
    st.write('\n')
    placeholder = st.info('förbereder lasso plot')
    plot_confusion_matrix(y_true, predict_meta_model(stacked_meta_val, meta_model='lasso'),
                        'lasso', fr=0.05, to=0.5, step=0.0001)
    placeholder.empty()
    
    stacked_meta_val['meta'] = y_pred_rf # default
    stacked_meta_val['y'] = y_true
    stacked_meta_val['avd'] = X_val.avd.values
    
    
    ################################################################
    #                         proba 6, 1, 9, (16)                  #
    ################################################################
    st.write('\n')
    for typ in typer:
        st.write('\n')
        name = 'proba' + typ.name[3:]
        y_pred = stacked_meta_val[name]
        plot_confusion_matrix(y_true, y_pred, name, fr=0.05, to=0.3, step=0.0001)
    
#%% 
##############################################################
#            FINAL LEARNING steps                            #
##############################################################
def final_learning(typer, n_splits=5,learn_models=True):
    placeholder = st.empty()

    with st.empty():
        
        # step 1: learn models and produce the stacked data
        placeholder.info('Step 1: Final learn models and produce the stacked data')
        stacked_data = TimeSeries_learning(None, typer, n_splits=n_splits,meta_fraction=0, save=True, learn_models=learn_models)
        
        # step 2: learn meta model
        placeholder.info('Step 2: Final learn meta model')
        _, _, _, _= learn_meta_models(stacked_data.drop(['y'], axis=1), stacked_data['y'], alpha=0.001, n_estimators=360, max_iter=55, max_depth=5, class_weight=None)
    
        st.success('✔️ Final learning done')
 

#%%
#       
#%% [markdown]
## Streamlit kod startar här
#%%
top = st.container()
buttons = st.container()
scraping = st.container()

def scrape(full=True):
    scraping.write('Starta web-scraping för ny data')
    with st.spinner('Ta det lugnt!'):
        st.image('winning_horse.png')  # ,use_column_width=True)
        #####################
        # start v75_scraping as a thread
        #####################
        i=0.0
        placeholder = st.empty()
        seconds=0
        my_bar = st.progress(i) 
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(v75_scraping )
            while future.running():
                time.sleep(1)
                seconds+=1
                placeholder.write(f"⏳ {seconds} sekunder")
                i+=1/65
                if i<0.99:
                    my_bar.progress(i)
            my_bar.progress(1.0)        
            df = future.result()

            df.to_csv('sparad_scrape.csv', index=False)
        
        st.balloons()
        my_bar.empty()
        placeholder.empty()
            
        st.session_state.df = df


models = [typ6, typ1, typ9]


############################################
#   Init session_state                     #
############################################
with top:
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        
    if 'datum' not in st.session_state:
        omg_df = pd.read_csv('omg_att_spela_link.csv' )
        urlen=omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state.datum = datum 
        
    st.header(f'Omgång:  {st.session_state.datum}')    

###########################################
# control flow with buttons               #
###########################################
with buttons:
    if st.sidebar.button('scrape'):
        scrape()
        del st.session_state.datum  # säkra att datum är samma som i scraping
    
    if st.sidebar.button('load data'):
        del st.session_state.datum  # säkra att datum är samma som i scraping
        try:
            df=pd.read_csv('sparad_scrape.csv')
            st.session_state.df=df
        except:
            # write error message
            st.error('Ingen data sparad')

    if st.session_state.df is not None:
        if st.sidebar.button('Learn TimeSeries'):        
            df = st.session_state.df
            stacked_data = TimeSeries_learning(df, typer, n_splits=3, meta_fraction=0.2, meta='rf', save=True, learn_models=True)
            st.success('✔️ TimeSeries learning done')
    
    if st.session_state.df is not None:
        if st.sidebar.button('Validate'):
            validate()      
            
    if st.session_state.df is not None:
        if st.sidebar.button('Final learning'):
            final_learning(typer)      
    
   
    
