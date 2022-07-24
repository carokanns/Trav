# %%
# from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from IPython.display import display
from catboost import CatBoostClassifier, Pool
import concurrent.futures
import time
import sklearn

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)
import streamlit as st

import sys
import pickle
from sklearn.ensemble import RandomForestRegressor

sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import typ as tp
pref = ''
import logging
logging.basicConfig(filename='app.log', filemode='w',
                    format='%(name)s - %(message)s', level=logging.INFO)

# %%

# %%
def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    df.drop(['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
            'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'], axis=1, inplace=True)
    if remove_mer:
        df.drop(remove_mer, axis=1, inplace=True)
    return df


def v75_scraping(full=True):
    if not full:
        # st.write("ANVÄNDER SPARAT")
        df = pd.read_csv('sparad_scrape_spela.csv')
        try:
            df.drop(['plac'], axis=1, inplace=True)
        except:
            pass
    else:
        print('start vs.v75_scraping')
        df = vs.v75_scraping(resultat=False, history=True, headless=True)
    
    for f in ['häst','bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
    return df

    # Alternativ metod
    # Ta fram rader för varje typ enligt test-resultaten innan
    # låt meta_model välja mellan typerna tills max insats, sorterat på meta_proba

# Funktioner för att prioritera mellan hästar
# Skapa ett Kelly-värde baserat på streck omvandlat till odds
def kelly(proba, streck, odds):  # proba = prob winning, streck i % = streck
    with open('rf_streck_odds.pkl', 'rb') as f:
        rf = pickle.load(f)

    if odds is None:
        o = rf.predict(streck.copy())
    else:
        o = rf.predict(streck.copy())

    # for each values > 40 in odds set to 1
    o[o > 40] = 1
    return (o*proba - (1-proba))/o

# för en omgång (ett datum) ta ut största diff för streck per avd
# om only_clear=True, enbart för diff >= 25
def lista_med_favoriter(df_, ant, only_clear):
    df = df_.copy()
    min_diff = 25 if only_clear else 0
    # sortera på avd,streck
    df = df.sort_values(['avd', 'streck'], ascending=[False, False])
    diff_list = []
    for avd in range(1, 8):
        diff = df.loc[df.avd == avd].streck.iloc[0] - \
            df.loc[df.avd == avd].streck.iloc[1]
        if diff >= min_diff:
            diff_list.append((avd, diff))

     # sortera på diff
    diff_list = sorted(diff_list, key=lambda x: x[1], reverse=True)
    return diff_list[:ant]

# temp is a list of tuples (avd, diff). check if avd is in the list
def check_avd(avd, temp):
    for t in temp:
        if t[0] == avd:
            return True
    return False


def compute_total_insats(df):
    summa = df.groupby('avd').avd.count().prod() / 2
    return summa

# feature med antal hästar per avdeling
def lägg_in_antal_hästar(df_):
    df = df_.copy()
    df['ant_per_lopp'] = None
    df['ant_per_lopp'] = df.groupby(['datum', 'avd'])['avd'].transform('count')
    return df

# räkna ut mest streck per avdeling
def mest_streck(X_, i, datum, avd):
    X = X_.copy()
    X.sort_values(by=['datum', 'avd', 'streck'], ascending=[True, True, False], inplace=True)
    return X.loc[(X.datum == datum) & (X.avd == avd), 'streck'].iloc[i]

# n flest streck per avd som features
def lägg_in_motståndare(X_, ant_motståndare):
    X = X_.copy()

    # set X['motståndare1'] to largest streck in every avd
    grouped = X.groupby(['datum', 'avd'])['streck']
    X['motståndare1'] = grouped.transform(max)

    for i in range(2, ant_motståndare+1):
        # set X['motståndare'+str(i)] to ith largest streck in every avd
        X['motståndare' +
            str(i)] = grouped.transform(lambda x: x.nlargest(i).min())

    return X

# som föregående men med diff istf faktiska värden
def lägg_in_diff_motståndare(X_, motståndare):
    X = X_.copy()

    # set X['motståndare1'] to largest streck in every avd
    grouped = X.groupby(['datum', 'avd'])['streck']
    X['diff1'] = grouped.transform(max) - X.streck

    for i in range(2, motståndare+1):
        # set X['motståndare'+str(i)] to ith largest streck in every avd
        X['diff' +
            str(i)] = grouped.transform(lambda x: x.nlargest(i).min()) - X.streck

    return X


#%%
# skapa modeller
#             name,  ant_hästar, proba, kelly, motst_ant,   motst_diff,  ant_favoriter,  only_clear, streck
typ6 = tp.Typ('typ6', True,       True, False,     0,          False,          0,            False,    True)
typ1 = tp.Typ('typ1', False,      True, False,     2,          True,           2,            True,     False)
typ9 = tp.Typ('typ9', True,       True, True,      2,          True,           2,            True,     True)
# typ16= tp.Typ('typ16', True,      True, True,      2,          True,          2,            False,    True)

typer = [typ6, typ1, typ9]  # load a file with pickl

with open('modeller\\meta_rf_model.model', 'rb') as f:
    meta_model = pickle.load(f)

# with open('modeller\\meta_ridge_model.model', 'rb') as f:
#    meta_model = pickle.load(f)

# with open('modeller\\meta_lasso_model.model', 'rb') as f:
#     meta_model = pickle.load(f)


#%%
# för stacking ta med alla hästar per typ och proba plus kelly
def build_stack_df(X_, typer):
    X = X_.copy()
    first_features = ['datum', 'avd', 'startnr', 'häst']
    stacked_data = X[first_features].copy()
    for typ in typer:
        nr = typ.name[3:]
        # print('stack '+typ.name)
        stacked_data['proba'+nr] = typ.predict(X)
        stacked_data['kelly'+nr] = kelly(stacked_data['proba'+nr], X[['streck']], None)
        
    with open(pref+'META_FEATURES.txt', 'r', encoding='utf-8') as f:
        meta_features = f.read().splitlines()
    stacked_data = stacked_data[first_features + meta_features]    
    
    return stacked_data


def meta_knn_predict(X_):
    # X_ innehåller även datum,startnr och avd
    first_features = ['datum', 'avd', 'startnr', 'häst']
    pred_columns = ['proba'+str(i) for i in [6, 1, 9]] + ['kelly'+str(i) for i in [6, 1, 9]]

    X = X_.copy()
    assert list(
        X.columns[:4]) == first_features, 'meta_model måste ha datum, avd och startnr, häst för att kunna välja'
    with open('modeller\\meta_knn_model.model', 'rb') as f:
        meta_model = pickle.load(f)

    # print(meta_model.predict_proba(X.iloc[:, -8:]))
    X['meta_predict'] = meta_model.predict_proba(X[pred_columns])[:, 1]
    my_columns = first_features + pred_columns + ['meta_predict']

    return X[my_columns]

def meta_rf_predict(X_):
    # X_ innehåller även datum,startnr och avd
    first_features = ['datum', 'avd', 'startnr', 'häst']
    pred_columns = ['proba'+str(i) for i in [6,1,9]] + ['kelly'+str(i) for i in [6,1,9]]
    
    X = X_.copy()
    assert list(X.columns[:4]) == first_features, 'meta_model måste ha datum, avd och startnr, häst för att kunna välja'
    with open('modeller\\meta_rf_model.model', 'rb') as f:
        meta_model = pickle.load(f)

    # print(meta_model.predict_proba(X.iloc[:, -8:]))
    X['meta_predict'] = meta_model.predict_proba(X[pred_columns])[:, 1]
    my_columns = first_features + pred_columns + ['meta_predict']

    return X[my_columns]

def meta_ridge_predict(X_):
    # X_ innehåller även datum,startnr och avd
    first_features = ['datum', 'avd', 'startnr', 'häst']
    pred_columns = ['proba'+str(i) for i in [6, 1, 9]] + ['kelly'+str(i) for i in [6, 1, 9]]

    assert list(
        X_.columns[:4]) == first_features, 'meta_model måste ha datum, avd och startnr, häst för att kunna välja'
    X = X_.copy()
    with open('modeller\\meta_ridge_model.model', 'rb') as f:
        meta_model = pickle.load(f)
    
    # print(meta_model.predict_proba(X.iloc[:, -8:]))
    X['meta_predict'] = meta_model._predict_proba_lr(X[pred_columns])[:, 1]
    my_columns = first_features + pred_columns + ['meta_predict']

    return X[my_columns]


def meta_lasso_predict(X_):
    # X_ innehåller även datum,startnr och avd
    first_features = ['datum', 'avd', 'startnr', 'häst']
    pred_columns = ['proba'+str(i) for i in [6, 1, 9]] + ['kelly'+str(i) for i in [6, 1, 9]]

    assert list(
        X_.columns[:4]) == first_features, 'meta_model måste ha datum, avd och startnr, häst för att kunna välja'
    X = X_.copy()
    with open('modeller\\meta_lasso_model.model', 'rb') as f:
        meta_model = pickle.load(f)

    # print(meta_model.predict_proba(X.iloc[:, -8:]))
    X['meta_predict'] = meta_model.predict(X[pred_columns])
    my_columns = first_features + pred_columns + ['meta_predict']

    return X[my_columns]


def mesta_diff_per_avd(X_):
    sm = X_.copy()
    # select the highest meta_predict per avd
    sm['first'] = sm.groupby('avd')['meta_predict'].transform(lambda x: x.nlargest(2).reset_index(drop=True)[0])
    sm['second'] = sm.groupby('avd')['meta_predict'].transform(lambda x: x.nlargest(2).reset_index(drop=True)[1])
    
    sm=sm.query("(first==meta_predict or second==meta_predict)").copy()
    sm['diff'] = sm['first'] - sm['second']
        
    # drop duplicates per avd 
    sm = sm.drop_duplicates(subset='avd', keep='first')
    
    sm.sort_values(by='diff', ascending=False, inplace=True)
    # sm.to_csv('mesta_diff_per_avd.csv')
    return sm

def välj_rad(df_meta_predict, max_insats=301):
    veckans_rad = df_meta_predict.copy()
    veckans_rad['välj'] = False   # inga rader valda ännu

    # first of all: select one horse per avd
    for avd in veckans_rad.avd.unique():
        max_pred = veckans_rad[veckans_rad.avd == avd]['meta_predict'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (veckans_rad.meta_predict == max_pred), 'välj'] = True
    # veckans_rad.query("välj==True").to_csv('veckans_basrad.csv')
    veckans_rad = veckans_rad.sort_values(by=['meta_predict'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)
    
    mest_diff = mesta_diff_per_avd(veckans_rad)
    
    cost = compute_total_insats(veckans_rad[veckans_rad.välj])
    
    # now select the rest of the horses one by one sorted by meta_predict
    for i, row in veckans_rad.iterrows():
        if row.avd == mest_diff.avd.iloc[0]: 
            continue
        if row.avd == mest_diff.avd.iloc[1]: 
            continue
        # print('i',i)
        veckans_rad.loc[i, 'välj'] = True
        cost_before = cost
        cost = compute_total_insats(veckans_rad[veckans_rad.välj])
        # print('cost',cost)
        if cost > max_insats:
            # veckans_rad.loc[i, 'välj'] = False
            break
        
    # print('cost', cost_before)
    veckans_rad.sort_values(by=['välj', 'avd'], ascending=[False, True], inplace=True)
    # display(veckans_rad[veckans_rad.välj])
    return veckans_rad


#%%
#############################
#### läs in meta_scores  ####
#############################
try:
    with open(pref+'modeller/meta_scores.pkl', 'rb') as f:
        meta_scores = pickle.load(f)
except:
    st.write('No meta_scores.pkl found')
    print('No meta_scores.pkl found')
    meta_scores = {'knn':0.6, 'rf':0.4,'ridge':0.7,'lasso':0.8}
# print('meta_scores:', meta_scores)    

#%%
def sort_list_of_meta(m):
    try:
        if meta_scores[m] == None:
            print(f'No score for {m} found')
            return 0
        return meta_scores[m]
    except:
        st.write(f'{m} not found')
        print(f'{m} not found')
        return -1

#%%
## Streamlit kod startar här
v75 = st.container()
scraping = st.container()
avd = st.container()
sortera = st.container()

if 'datum' in st.session_state:
    datum = st.sidebar.date_input('Välj datum')
    datum = datum.strftime('%Y-%m-%d')

    if datum != st.session_state['datum']:
        st.session_state['datum'] = datum
        datum="https://www.atg.se/spel/"+datum+"/V75/"
        omg_df = pd.DataFrame([datum],columns=['Link'])
        omg_df.to_csv('omg_att_spela_link.csv', index=False)
    


models = [typ6, typ1, typ9]   # typ16 och typ9 är samma förutom hur man väljer rader

def use_meta(df_stack,meta):
    if meta == 'knn':
        df_meta = meta_knn_predict(df_stack)
    elif meta == 'rf':
        df_meta = meta_rf_predict(df_stack)
    elif meta=='lasso':
        df_meta = meta_lasso_predict(df_stack)
    elif meta=='ridge':
        df_meta = meta_ridge_predict(df_stack)
    else:
        st.error(f'meta={meta} finns inte - avänder RandomForestClassifier')
        df_meta = meta_rf_predict(df_stack)       
    
    df_meta.reset_index(drop=True, inplace=True)
    df = välj_rad(df_meta)
    st.session_state.df = df
    st.experimental_rerun()
    
    
# define st.state
if 'df' not in st.session_state:
    st.session_state['df'] = None
    print("sklearn version", sklearn.__version__)
    
if 'meta' not in st.session_state: 
    st.session_state['meta'] = 'rf'

with scraping:
    def scrape(full=True, meta='rf'):
        scraping.write('web-scraping för ny data')
        with st.spinner('Ta det lugnt!'):
            st.image('winning_horse.png')  # ,use_column_width=True)
            
            #####################
            # start v75_scraping as a thread
            #####################
            
            i=0.0
            seconds=0
            placeholder = st.empty()

            my_bar = st.progress(i)   
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(v75_scraping , full)
                while future.running():
                    time.sleep(1)
                    seconds+=1
                    placeholder.write(f"⏳ {seconds} sekunder")
                    i+=1/65
                    if i<0.99:
                        my_bar.progress(i)
                my_bar.progress(1.0)        
                df_scraped = future.result()

                df_scraped.to_csv('sparad_scrape_spela.csv', index=False)
            
            st.balloons()
            my_bar.empty()
            placeholder.empty()
            # print(df_scraped.datum.unique())
            df_stack = build_stack_df(df_scraped, typer)
            df_stack.to_csv('sparad_stack.csv', index=False)
            use_meta(df_stack, meta)
            
    if st.button('scrape'):
        scrape(meta=st.session_state['meta'])
        del st.session_state.datum  # säkra att datum är samma som i scraping
        
    if st.sidebar.button('reuse scrape'):
        scrape(False, meta=st.session_state['meta'])
        del st.session_state.datum  # säkra att datum är samma som i scraping
    scraping.empty()
    
with v75:
    if 'datum' not in st.session_state:
        omg_df = pd.read_csv('omg_att_spela_link.csv' )
        urlen=omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state.datum = datum
        
    st.title('v75 -  ' +st.session_state.datum)

    
with avd:
    if st.session_state.df is not None:
        use = avd.radio('Välj avdelning', ('Avd 1 och 2','Avd 3 och 4','Avd 5 och 6','Avd 7','clear'))
        avd.subheader(use)
        st.write('TA BORT OUTLIERS')
        col1, col2 = st.columns(2)
        # print(df.iloc[0].häst)
        dfi=st.session_state.df
        dfi.rename(columns={'startnr': 'nr', 'meta_predict': 'Meta'}, inplace=True)
        dfi['kelly'] = (dfi[['kelly1','kelly6','kelly9']]).max(axis=1)
        # print(dfi[dfi.välj][['avd','nr','häst','kelly1','kelly6','kelly9','kelly16','Meta']])
        # CSS to inject contained in a string
        hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        
        if use == 'Avd 1 och 2':
            col1.table(dfi[(dfi.avd == 1) & dfi.välj].sort_values(by=['Meta'],ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
            col2.table(dfi[(dfi.avd == 2) & dfi.välj].sort_values(by=['Meta'],ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
        elif use=='Avd 3 och 4':
            col1.table(dfi[(dfi.avd == 3) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
            col2.table(dfi[(dfi.avd == 4) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
        elif use=='Avd 5 och 6':
            col1.table(dfi[(dfi.avd == 5) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
            col2.table(dfi[(dfi.avd == 6) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
        elif use=='Avd 7':
            col1.table(dfi[(dfi.avd == 7) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst', 'Meta', 'kelly']])
        elif use=='clear':
            st.stop()    
        else:
            st.write('ej klart')
            
        st.write(compute_total_insats(dfi[dfi.välj]))

with sortera:   
    if st.sidebar.checkbox('se data'):
        dfr = st.session_state.df
        sort=st.sidebar.radio('sortera på',['Meta','kelly','avd'])
        if sort:
            if sort=='kelly':
                st.write(dfr[['avd','nr','häst','Meta','kelly']].sort_values(by=['kelly','avd'],ascending=[False,False]))
            elif sort=='Meta':
                st.write(dfr[['avd', 'nr', 'häst','Meta', 'kelly']].sort_values(by=['Meta','avd','nr'], ascending=[False, False,False]))
            else:
                dfra  = dfr[['avd','nr','häst','proba6','proba9','proba1', 'kelly6','kelly9','kelly1','Meta','välj','kelly']]
                st.write(dfra.sort_values(by=['avd', 'nr'], ascending=[True, True]))
                
meta_list = ['rf', 'knn','ridge', 'lasso']
meta_list.sort(reverse=True, key=lambda x: sort_list_of_meta(x))
meta = st.sidebar.radio('välj meta_model',meta_list)

if meta != st.session_state.meta:
    st.session_state.meta = meta
    st.write('meta_model:', meta)
    df_scraped = pd.read_csv('sparad_scrape_spela.csv')
    try:
        df_scraped.drop(['plac'], axis=1, inplace=True)
        st.info('this file is not up to date - a scrape is needed')
    except:
        pass
    df_stack = build_stack_df(df_scraped, typer)
    df_stack.to_csv('sparad_stack.csv', index=False)
    use_meta(df_stack, meta)
