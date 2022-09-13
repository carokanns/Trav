import streamlit as st
from logging import PlaceHolder
from category_encoders import TargetEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
import streamlit as st
import sys
import time, datetime
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

st.set_page_config(page_title="V75 Learning", page_icon="🏫")

st.markdown("# 🏫 V75 Learning")
st.sidebar.header("🏫 V75 Learning")
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

def förbered(df,meta_fraction=None):
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
def TimeSeries_learning(df_ny_, typer, n_splits=5,meta_fraction=None, save=True, learn_models=True):
    """
    Skapar en stack av {1-meta_fraction} av X från alla typer. Används som input till meta_model.
        - learn_models=True betyder att vi både gör en learning och skapar en stack
        - learn_models=False betyder att vi bara skapar en stack och då har save ingen funktion
    """
    df_all = pd.read_csv(pref+'all_data.csv')
        
    if df_ny_ is not None:
        df_ny = df_ny_.copy()
        df_all = concat_data(df_all.copy(), df_ny, save=True)
    
    X, y, X_val, _ = förbered(df_all, meta_fraction=meta_fraction)

    validation_text = ""

    if X_val is not None:
        validation_text = f', Validation: _{X_val.datum.iloc[0]} - -{X_val.datum.iloc[-1]}'
        
    st.info(
        f'Train: {X.datum.iloc[0]} --{X.datum.iloc[-1]}{validation_text}')
    
    ts = TimeSeriesSplit(n_splits=n_splits)
    # läs in meta_features
    with open(pref+'META_FEATURES.txt', 'r',encoding='utf-8') as f:    
        meta_features = f.read().splitlines()
        
    stacked_data=pd.DataFrame(columns=meta_features + ['y'])
    
    ###############################################################################
    #         Step 1: Learn the models on ts.split X_train and predict on X_test  #
    ###############################################################################
    st.write('Skapar stacked_data till meta')
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
                with open('optimera/params_'+typ.name+'.json', 'r') as f:
                    params = json.load(f)
                    params = params['params']
                # learn på X_train-delen
                cbc = typ.learn(X_train, y_train, X_test, y_test,params=params,save=save)
                
            # predict the new fitted model on X_test-delen    
            nr = typ.name[3:]
            this_proba=typ.predict(X_test)
            
            # Bygg up meta-kolumnerna (proba och Kelly) för denns typ
            temp_df['proba'+nr] = this_proba

            this_kelly=kelly(this_proba, X_test[['streck']], None)
            temp_df['kelly' + nr] = this_kelly
        
        stacked_data = pd.concat([stacked_data, temp_df],ignore_index=True)
        stacked_data.y = stacked_data.y.astype(int)
        
    #  Make sure that the meta features are in the right order in stack 
    stacked_data = stacked_data[meta_features+['y']]

    my_bar.progress(1.0)
    
    ###############################################################################
    #         Step 2:       Learn the meta models                                 #
    ###############################################################################
    st.write('Learning meta models')
    _, _, _, _ = learn_meta_models(stacked_data[meta_features], stacked_data['y'])


    ###############################################################################
    #         Step 3: learn models on all of X - what iteration to use?           #
    ###############################################################################
    st.write('Learn models on all of Train')
    my_bar2 = st.progress(0)
    ant_meta_models = 4
    step = 1/(ant_meta_models) - 0.0000001
    steps = 0.0
    my_bar2.progress(steps)

    for typ in typer:
        steps += step
        my_bar2.progress(steps)
        if learn_models:
            with open('optimera/params_'+typ.name+'.json', 'r') as f:
                params = json.load(f)
                
            params = params['params']   
            cbc = typ.learn(X, y, None, None, iterations=500, params=params,save=save)
     
    my_bar2.progress(1.0)
    st.empty()
    return stacked_data[meta_features + ['y']]

def skapa_stack_learning(X_, y):
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
def learn_meta_ridge_model(X, y, save=True):
    from sklearn.linear_model import RidgeClassifier
    
    with open(pref+'optimera/params_ridge.json', 'r') as f:
        params = json.load(f)['params']
        # st.write(params)
        
    ridge_model = RidgeClassifier(**params,random_state=2022)
    
    ridge_model.fit(X, y)

    if save:
        with open(pref+'modeller/meta_ridge_model.model', 'wb') as f:
            pickle.dump(ridge_model, f)

    return ridge_model

##### RandomForestClassifier (meta model) #####
def learn_meta_rf_model(X, y, save=True):
    
    with open(pref+'optimera/params_rf.json', 'r') as f:
        params = json.load(f)
        params=params['params']
        # st.write(params)

    rf_model = RandomForestClassifier(**params, n_jobs=6, random_state=2022)
    rf_model.fit(X, y)
    
    ######################### for testing ###############################
    rf_train = X.copy(deep=True)
    rf_train['y'] = y
    rf_train.to_csv(pref+'rf_train.csv', index=False)
    #########################              ###############################
    
    if save:
        with open(pref+'modeller/meta_rf_model.model', 'wb') as f:
            pickle.dump(rf_model, f)

    return rf_model

##### LassoClassifier (meta model) #####
def learn_meta_lasso_model(X, y, save=True):
    from sklearn.linear_model import Lasso
    with open(pref+'optimera/params_lasso.json', 'r') as f:
        params = json.load(f)
        params=params['params']
        # st.write(params)
        
    lasso_model = Lasso(**params, random_state=2022)
    
    lasso_model.fit(X, y)

    if save:
        with open(pref+'modeller/meta_lasso_model.model', 'wb') as f:
            pickle.dump(lasso_model, f)

    return lasso_model

##### KNeighborsClassifier (meta model) #####
def learn_meta_knn_model(X, y, save=True):
    from sklearn.neighbors import KNeighborsClassifier
    with open(pref+'optimera/params_knn_meta.json', 'r') as f:
        params = json.load(f)
        params = params['params']
        # st.write(params)
        
    knn_model = KNeighborsClassifier(**params, n_jobs=6)
    knn_model.fit(X, y)
    
    if save:
        with open(pref+'modeller/meta_knn_model.model', 'wb') as f:
            pickle.dump(knn_model, f)

    return knn_model


def learn_meta_models(X, y, save=True):
    """ all meta models will be fitted on X and y """
    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
    assert meta_features == list(X.columns), f'X.columns {list(X.columns)} is not the same as meta_features {meta_features}' 
       
    Ridge_Classifier =  learn_meta_ridge_model(X, y,save=save)
    RandomForest_Classifier = learn_meta_rf_model(X, y, save=save)
    Lasso_model = learn_meta_lasso_model(X, y, save=save)
    Knn_model   = learn_meta_knn_model(X, y, save=save)
    
    return Ridge_Classifier, RandomForest_Classifier, Lasso_model, Knn_model
    
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
        X.to_csv(pref+'rf_validate.csv', index=False)
        y_pred = predict_meta_rf_model(X)
        # write y_pred to file for testing
        # first make y_pred a dataframe
        rf_y_pred = pd.DataFrame(y_pred, columns=['0', '1'])
        rf_y_pred.to_csv(pref+'rf_y_pred.csv', index=False)
        return y_pred[:, 1]
    elif meta_model == 'lasso':
        pred = predict_meta_lasso_model(X)
        return pred
    elif meta_model == 'knn':
        y_pred = predict_meta_knn_model(X)
        return y_pred[:, 1]
    elif meta_model == None:
        assert False, 'ingen meta_model angiven'
    else:
        return meta_model.predict(X)
    
# write the scores    
def display_scores(y_true, y_pred, spelade):    
    st.write('AUC',round(roc_auc_score(y_true, y_pred),5),'F1',round(f1_score(y_true, y_pred),5),'Acc',round(accuracy_score(y_true, y_pred),5),'MAE',round(mean_absolute_error(y_true, y_pred),5), '\n', spelade)
    return roc_auc_score(y_true, y_pred)

def find_threshold(y_pred,fr, to,margin):
    """ hitta threshold som ger 2.5 spelade per avdelning """
    thresh = 0
    cnt=0
    # make a binary search
    while cnt < 1000:
        thresh = (fr + to) / 2 
        antal_spelade_per_avd = 12 * sum(y_pred > thresh)/len(y_pred)
        if (antal_spelade_per_avd > (2.5 - margin)) and (antal_spelade_per_avd < (2.5 + margin)):
            break
        
        if antal_spelade_per_avd > 2.5:
            fr = thresh-0.00001
        else:
            to = thresh+0.00001
        cnt += 1
        
    print('ant',cnt, 'thresh',round(thresh,4))
    if cnt >= 1000:
        print('threshold not found', 'fr',round(fr,6), 'to',round(to,6))
        
    return thresh

def plot_confusion_matrix(y_true, y_pred, typ, fr=0.0, to=0.9, margin=0.001):

    #### Först:  hitta ett threshold som tippar ca 2.5 hästar per avd ####
    # thresh = 0
    # for thresh in np.arange(fr, to, step):
    #     cost = 12*sum(y_pred > thresh)/len(y_pred)
    #     if cost < 2.5:
    #         break
    thresh = round(find_threshold(y_pred,fr,to,margin), 4)
    print(f'Threshold: {thresh}\n')
    y_pred = (y_pred > thresh).astype(int)
    # confusion_matrix_graph(y_true, y_pred, f'{typ} threshold={thresh}')

    #### Sedan: confusion matrix graph ####
    title = f'{typ} threshold={thresh}'
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred,)
    fig, ax = plt.subplots()
    sns.set(font_scale=2.0)
    sns.heatmap(cm/np.sum(cm), annot=True, fmt=".2%", linewidths=.5,
                square=True, cmap='Blues_r')

    # increase font size
    plt.rcParams['font.size'] = 20
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    # plot fig
    
    st.write(fig)
    
    # read dict from disk
    try:
        with open(pref+'modeller/meta_scores.pkl', 'rb') as f:
            meta_scores = pickle.load(f)
    except:
        st.write('No meta_scores.pkl found')
        meta_scores = {'knn': 0, 'lasso': 0, 'rf': 0, 'ridge': 0}  

    #### print scores ####
    typ_AUC = display_scores(y_true, y_pred, f'spelade per lopp: {round(12 * sum(y_pred)/len(y_pred),4)}' )
    meta_scores[typ] = float(typ_AUC)
    #### save dict to disk ####
    with open(pref+'modeller/meta_scores.pkl', 'wb') as f:
        pickle.dump(meta_scores, f)
    
def validate(drop=[],fraction=None):
    st.info('skall endast  köras efter "Learn for Validation"')
    df_all = pd.read_csv(pref+'all_data.csv')
    
    _, _, X_val, y_val = förbered(df_all, meta_fraction=fraction)
    st.info(f'Validerar på:  {X_val.datum.iloc[0]} - -{X_val.datum.iloc[-1]}')
    
    # create the stack from validation data
    stacked_val, y_val = skapa_stack_learning(X_val, y_val)
    stacked_val = stacked_val.drop(drop, axis=1)
        
    ##############################################################
    #                          Meta models                       #
    ##############################################################
    
    y_true = y_val.values
    y_pred_rf = predict_meta_model(stacked_val, meta_model='rf') 
    
    ############## write y_true to file for testing ##############
    # first make y_true a dataframe
    rf_y_true = pd.DataFrame(y_true, columns=['y'])
    rf_y_true.to_csv(pref+'rf_y_true.csv', index=False)
    ##############################################################
    
    st.info('förbereder rf plot')
    plot_confusion_matrix(y_true, y_pred_rf, 
                        'rf', fr=0.0, to=1.0, margin=0.01)

    st.write('\n')
    st.info('förbereder knn plot')
    plot_confusion_matrix(y_true, predict_meta_model(stacked_val, meta_model='knn'),
                          'knn', fr=0.0, to=0.9)

    st.write('\n')
    st.info('förbereder ridge plot')
    plot_confusion_matrix(y_true, predict_meta_model(stacked_val, meta_model='ridge'), 
                        'ridge', fr=0.0, to=0.9)
    # placeholder.empty()
    
    st.write('\n')
    st.info('förbereder lasso plot')
    plot_confusion_matrix(y_true, predict_meta_model(stacked_val, meta_model='lasso'),
                        'lasso', fr=0.0, to=0.9)
    # placeholder.empty()
    
    stacked_val['meta'] = y_pred_rf # default
    stacked_val['y'] = y_true
    stacked_val['avd'] = X_val.avd.values
    
    
    ################################################################
    #                         proba 6, 1, 9, (16)                  #
    ################################################################
    st.write('\n')
    for typ in typer:
        st.write('\n')
        name = 'proba' + typ.name[3:]
        y_pred = stacked_val[name]
        plot_confusion_matrix(y_true, y_pred, name, fr=0.0, to=0.9)
    
#%% 
##############################################################
#            FINAL LEARNING                                  #
##############################################################
def final_learning(typer, n_splits=5):
    st.info('Final learning on all the data')
   
    _ = TimeSeries_learning(None, typer, n_splits=n_splits,meta_fraction=0, save=True)
    
    # st.info('Step 2: Final learn meta model')
    # _, _, _, _= learn_meta_models(stacked_data.drop(['y'], axis=1), stacked_data['y'])

    st.success('✔️ Final learning done')
 

#%%
def scrape(full=True):
    scraping.write('Starta web-scraping för ny data')
    with st.spinner('Ta det lugnt!'):
        # st.image('winning_horse.png')  # ,use_column_width=True)
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
            scraping.write('✔️ Scraping done, pls wait')      
            time.sleep(2)   
            df = future.result()
            scraping.write(f'✔️ {len(df)} rader hämtade')
            
            df.to_csv('sparad_scrape_learn.csv', index=False)
        
        st.balloons()
        my_bar.empty()
        placeholder.empty()
            
        st.session_state.df = df


models = [typ6, typ1, typ9]

#%%
top = st.container()
buttons = st.container()
scraping = st.container()

############################################
#   Init session_state                     #
############################################
with top:
    if 'fraction' not in st.session_state:
        st.session_state['fraction'] = 0.25
        
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        
    if 'datum' not in st.session_state:
        omg_df = pd.read_csv('omg_att_spela_link.csv' )
        urlen=omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state.datum = datum 
        
    if 'datum' in st.session_state:
        datum=st.session_state['datum']
        year=int(datum[:4])
        month=int(datum[5:7])
        day=int(datum[8:])
        datum = st.sidebar.date_input('Välj datum',datetime.date(year, month, day))
        datum = datum.strftime('%Y-%m-%d')

        if datum != st.session_state['datum']:
            st.session_state['datum'] = datum
            datum="https://www.atg.se/spel/"+datum+"/V75/"
            omg_df = pd.DataFrame([datum],columns=['Link'])
            omg_df.to_csv('omg_att_spela_link.csv', index=False)
            
    st.header(f'Omgång:  {st.session_state.datum}')            
    
###########################################
# control flow with buttons               #
###########################################
with buttons:
    if st.sidebar.button('scrape'):
        st.write(f'web scraping {st.session_state.datum}')
        try:
            scrape()
            del st.session_state.datum  # säkra att datum är samma som i scraping
        except:
            st.error("Fel i web scraping. Kolla att resultat finns för datum och internet är tillgängligt")
    
    if st.sidebar.button('reuse scrape'):
        # del st.session_state.datum  # säkra att datum är samma som i scraping
        try:
            df=pd.read_csv('sparad_scrape_learn.csv')
            st.session_state.df=df
            if df.datum.iloc[0] != st.session_state.datum:
                st.error(f'Datum i data = {df.datum.iloc[0]} \n\n är inte samma som i omgång') 
            else:    
                st.success(f'inläst data med datum = {df.datum.iloc[0]}') 
        except:
            # write error message
            st.error('Ingen data sparad')

    if st.session_state.df is not None:
        if st.sidebar.button('Learn for validation'):  
            st.write('TimeSeries learning for validation')
            fraction = st.session_state.fraction
            df = st.session_state.df
            st.write(f'learn models and meta models on first {(1-fraction)*100} % of the data')    
            
            stacked_data = TimeSeries_learning(df, typer, n_splits=5, meta_fraction=fraction, save=True, learn_models=True)
            st.success('✔️ TimeSeries learning done')
    
        if st.sidebar.button('Validate'):
            validate(fraction=st.session_state.fraction)
       
        if st.sidebar.button('Final learning'):
            final_learning(typer)  

        if st.sidebar.button('Clear'):
            st.empty()
            
    
