# moduler
import streamlit as st
import sys
import time
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool, cv
from IPython.display import display
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')

# sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs

import typ as tp

pref=''   # '../'

###################################################################################
# %%
################################################
#              Web scraping                    #
################################################

def v75_scraping():
    df, strukna = vs.v75_scraping(history=True, resultat=True, headless=True)
    
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
def TimeSeries_learning(df_ny_, typer, n_splits=5,meta_fraction=0.2,learn_models=True):
    # learn models and produce the stacked data
    # learn_models=True betyder att vi både gör en learning och skapar en stack
    # learn_models=False betyder att vi bara skapar en stack
    
    df_all = pd.read_csv(pref+'all_data.csv')
        
    if df_ny_ is not None:
        df_ny = df_ny_.copy()
        df_all = concat_data(df_all.copy(), df_ny, save=True)
        
    print('sista datum',df_all.datum.iloc[-1])
    
    X, y, _, _ = förbered(df_all, meta_fraction=meta_fraction)
    # print('X shape',X.shape,'y shape',y.shape)
    # print('X_val shape',X_val.shape,'y_val shape',y_val.shape)

    ts = TimeSeriesSplit(n_splits=n_splits)
    stacked_data=pd.DataFrame(columns=['proba6', 'proba1', 'proba9', 'proba16', 'kelly6', 'kelly1', 'kelly9', 'kelly16','y'])
        
    my_bar = st.progress(0)   
    step=1/(n_splits*len(typer))-0.0000001
    steps=0.0
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
                typ.learn(X_train, y_train, X_test, y_test)
                
            nr = typ.name[3:]
            this_proba=typ.predict(X_test)
            
            temp_df['proba'+nr] = this_proba

            this_kelly=kelly(this_proba, X_test[['streck']], None)
            temp_df['kelly' + nr] = this_kelly
        stacked_data = pd.concat([stacked_data, temp_df],ignore_index=True)
        stacked_data.y = stacked_data.y.astype(int)
    my_bar.progress(1.0)
    my_bar.empty()
    return stacked_data    

def skapa_stack_learning(X_, y, save=True):
    X=X_.copy()
    stacked_data = pd.DataFrame()
    for typ in typer:
            nr = typ.name[3:]
            stacked_data['proba'+nr] = typ.predict(X)
            stacked_data['kelly' + nr] = kelly(stacked_data['proba' + nr], X[['streck']], None)

    print(stacked_data.columns)
    assert len(stacked_data) == len(y), f'stacked_data {len(stacked_data)} and y {len(y)} should have same length'
    return stacked_data,y   # enbart stack-info

##############################################################
#                   RidgeClassifier (meta model)             #
##############################################################
def learn_meta_ridge_model(X, y,  class_weight='balanced', save=True):
    from sklearn.linear_model import RidgeClassifier
    
    ridge_model = RidgeClassifier(class_weight=class_weight, random_state=2022)
    ridge_model.fit(X,y)
    # pickle save stacking
    if save:
        with open(pref+'modeller/meta_ridge_model.model', 'wb') as f:
            pickle.dump(ridge_model, f)
    
    return ridge_model

##############################################################
#              learning steps for TimeSeries                 #
##############################################################
def all_TS_learning(df_ny,typer, n_splits=5,learn_models=True):
    placeholder = st.empty()

    with st.empty():
        
        # step 1: learn models and produce the stacked data
        placeholder.info('Step 1: learn models and produce the stacked data')
        stacked_data = TimeSeries_learning(df_ny,typer, n_splits=n_splits,learn_models=learn_models)
        
        # step 2: learn meta model
        placeholder.info('Step 2: learn meta model')
        meta_model=learn_meta_ridge_model(stacked_data.drop(['y'], axis=1), stacked_data['y'], class_weight=None) # use RidgeClassifier
    
        st.success('✔️ Learning done')
    placeholder.empty()
    
#%% 
##############################################################
#                     VALIDATE                               #
##############################################################

def predict_meta_ridge_model(X,ridge_model=None):
    if ridge_model is None:
        with open(pref+'modeller/meta_ridge_model.model', 'rb') as f:
            ridge_model = pickle.load(f)
            
    return ridge_model._predict_proba_lr(X)

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
    
def scores(y_true, y_pred):    
    # what is the AUC score?
    from sklearn.metrics import roc_auc_score
    st.write('AUC',roc_auc_score(y_true, y_pred),'  ')
    # and the F1 score
    from sklearn.metrics import f1_score
    st.write('F1',f1_score(y_true, y_pred),'  ')
    #accuracy
    from sklearn.metrics import accuracy_score
    st.write('Acc',accuracy_score(y_true, y_pred), '  ')
    
def validate(ridge_model=None):
    st.info('Only accurate directly after "Learn TimeSeries"')
    df_all = pd.read_csv(pref+'all_data.csv')
    print('sista datum',df_all.datum.iloc[-1])
    
    _, _, X_val, y_val = förbered(df_all, meta_fraction=0.2)
    
    #create the stack from validation data
    stacked_meta_val, y_val = skapa_stack_learning(X_val, y_val)
    stacked_meta_val['meta'] = predict_meta_ridge_model(stacked_meta_val, ridge_model=meta_model)[:,1]
    stacked_meta_val['y'] = y_val.values
    stacked_meta_val['avd'] = X_val.avd.values
    ##############################################################
    #                          Meta model                        #
    ##############################################################
    typ='meta'
    for tresh in np.arange(0.1,0.5,0.0001):
        cost=12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val)
        if cost<2.5:
            break
    tresh=round(tresh,4)
    # print(f'Treshold: {tresh}')
    confusion_matrix_graph(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh).astype(int), f'{typ} treshold={tresh}')
    scores(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh))
    st.write('spelade per lopp:',12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val))
   
    ################################################################
    #                         proba6                               #
    ################################################################ 
    typ='proba6'
    for tresh in np.arange(0.1,0.5,0.001):
        cost=12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val)
        if cost<2.5:
            break
        
    tresh = round(tresh,4)    
    # print(f'Treshold: {tresh}\n')
    confusion_matrix_graph(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh).astype(int), f'{typ} treshold={tresh}')
    scores(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh))
    st.write('spelade per lopp:',12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val))
    
    ################################################################
    #                         proba1                               #
    ################################################################
    typ='proba1'
    for tresh in np.arange(0.1,0.5,0.001):
        cost=12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val)
        if cost<2.5:
            break
        
    tresh = round(tresh,4)    
    # print(f'Treshold: {tresh}\n')
    confusion_matrix_graph(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh).astype(int), f'{typ} treshold={tresh}')
    scores(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh))
    st.write('spelade per lopp:',12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val))
    
    ################################################################
    #                         proba9                               #
    ################################################################
    typ='proba9'
    for tresh in np.arange(0.1,0.5,0.001):
        cost=12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val)
        if cost<2.5:
            break
        
    tresh = round(tresh,4)    
    # print(f'Treshold: {tresh}\n')
    confusion_matrix_graph(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh).astype(int), f'{typ} treshold={tresh}')
    scores(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh))
    st.write('spelade per lopp:',12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val))
    
    ################################################################
    #                         proba16                              #
    ################################################################
    typ='proba16'
    for tresh in np.arange(0.1,0.5,0.001):
        cost=12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val)
        if cost<2.5:
            break
        
    tresh = round(tresh,4)    
    # print(f'Treshold: {tresh}\n')
    confusion_matrix_graph(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh).astype(int), f'{typ} treshold={tresh}')
    scores(stacked_meta_val['y'], (stacked_meta_val[typ]>tresh))
    st.write('spelade per lopp:',12*sum(stacked_meta_val[typ]>tresh)/len(stacked_meta_val))
 
#%% 
##############################################################
#            FINAL LEARNING steps                            #
##############################################################
def final_learning(typer, n_splits=5,learn_models=True):
    placeholder = st.empty()

    with st.empty():
        
        # step 1: learn models and produce the stacked data
        placeholder.info('Step 1: Final learn models and produce the stacked data')
        stacked_data = TimeSeries_learning(None, typer, n_splits=n_splits,meta_fraction=0,learn_models=learn_models)
        
        # step 2: learn meta model
        placeholder.info('Step 2: Final learn meta model')
        meta_model=learn_meta_ridge_model(stacked_data.drop(['y'], axis=1), stacked_data['y'], class_weight=None) # use RidgeClassifier
    
        st.success('✔️ Final learning done')
    placeholder.empty()
 
    
#%%
# skapa modeller
#           name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck
typ6 = tp.Typ('typ6', True,       True, False,     0,          False,          0,            False,    True)
typ1 = tp.Typ('typ1', False,      True, False,     2,          True,           2,            True,     False)
typ9 = tp.Typ('typ9', True,       True, True,      2,          True,           2,            True,     True)
typ16= tp.Typ('typ16', True,      True, True,      2,          True,           2,            False,    True)

typer = [typ6, typ1, typ9, typ16]  # load a file with pickl


with open('modeller\\meta_ridge_model.model', 'rb') as f:
    meta_model = pickle.load(f)

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
        df=v75_scraping()
        df.to_csv('sparad_scrape.csv', index=False)
        
        st.balloons()
        
        st.session_state.df = df


models = [typ6, typ1, typ9, typ16]


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
            all_TS_learning(st.session_state.df,typer, n_splits=5,learn_models=True)
    
    if st.session_state.df is not None:
        if st.sidebar.button('Validate'):
            validate(st.session_state.df)      
            
    if st.session_state.df is not None:
        if st.sidebar.button('Final learning'):
            final_learning(typer)      
    
   
    
