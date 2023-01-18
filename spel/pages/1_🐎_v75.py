
#%%
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from IPython.display import display
# from catboost import CatBoostClassifier, Pool
import concurrent.futures
import time
import datetime
import sklearn
import logging

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)
import streamlit as st

import sys
import pickle
import json
# from sklearn.ensemble import RandomForestRegressor

sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import typ as tp
import travdata as td
import skapa_modeller as mod
pref = ''
import logging
    
    
# %%

logging.basicConfig(level=logging.DEBUG, filemode='w', filename='v75.log', force=True,
                    encoding='utf-8', format='v75:' '%(asctime)s - %(levelname)s - %(message)s - %(lineno)d')
logging.info('Startar')

#%%
st.set_page_config(page_title="v75 Spel", page_icon="🐎")

# st.markdown("# 🐎 V75 Spel")
st.sidebar.header("🐎 V75 Spel")


# %%
def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    df.drop([ 'vodds', 'podds', 'bins', 'h1_dat',
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
    # Ta fram rader för varje modell enligt test-resultaten innan
    # låt meta_model välja mellan modellerna tills max insats, sorterat på meta_proba


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
# TODO: Kopiera Layer1 och Layer2 kod från Hyperparms/Learn
# TODO: Ta bort alla gamla meta-modeller och dess funktioner
# TODO: Ta bort valet av meta-modeller i streamlit
# TODO: Inför val mellan matematiskt eller geometriskt medelvärde 

L1_modeller, L2_modeller = mod.skapa_modeller()
logging.info('Modeller skapade')

def förbered_data(df_ny):
    class v75_ny(td.v75):  # used for newly scraped data in order to reuse functions in v75
        def __init__(self, df_ny, pref=''):
            self.pref = pref
            self.df = df_ny
            self.work_df = self.df.copy()   # arbetskopia

    ny = v75_ny(df_ny, pref=pref)
    ny.förbered_data(missing_num=False)
    ny = ny.test_lägg_till_kolumner()
        
    return ny
    
    
def add_to_stack(model, name, X, stack_data):
    """Bygg på stack_data inklusive ev Kelly
        X är en omgång scrapad data
        stack_data fylls på med nya data
    """
    # assert 'y' in stack_data.columns, 'y is missing in stack_data'
    this_proba = model.predict(X)
    # print(f'X_meta.shape = {X_meta.shape} this_proba.shape={this_proba.shape}')

    # Bygg up meta-kolumnerna (proba och Kelly) för denns typ
    nr = name[3:]
    stack_data['proba'+nr] = this_proba
    # stack_data['kelly'+nr] = kelly(this_proba, X_meta[['streck']], None)

    return stack_data

#%%
# för stacking ta med alla hästar per modell och proba plus ev kelly
def build_stack_df(X_, modeller):
    X = förbered_data(X_)
    stack_data = X.copy()
    st.dataframe(X)
    for modell in modeller:
        stack_data = add_to_stack(modell, modell.name, X, stack_data)
           
    return stack_data



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

def välj_rad(df_predicted, max_insats=300):
    veckans_rad = df_predicted.copy()
    veckans_rad['välj'] = False   # inga rader valda ännu

    # first of all: select one horse per avd
    for avd in veckans_rad.avd.unique():
        max_pred = veckans_rad[veckans_rad.avd == avd]['meta_predict'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (veckans_rad.meta_predict == max_pred), 'välj'] = True
    # veckans_rad.query("välj==True").to_csv('veckans_basrad.csv')
    veckans_rad = veckans_rad.sort_values(by=['meta_predict'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)
    
    mest_diff = mesta_diff_per_avd(veckans_rad)
    
    cost = 0.5 # 1 rad
    
    # now select the rest of the horses one by one sorted by meta_predict
    for i, row in veckans_rad.iterrows():
        if row.avd == mest_diff.avd.iloc[0]: 
            continue
        if row.avd == mest_diff.avd.iloc[1]: 
            continue
        # print('i',i)
        veckans_rad.loc[i, 'välj'] = True
        cost = compute_total_insats(veckans_rad[veckans_rad.välj])
        if cost > max_insats:
            veckans_rad.loc[i, 'välj'] = False
            break
        
    veckans_rad.sort_values(by=['välj', 'avd'], ascending=[False, True], inplace=True)

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
    meta_scores = {'knn':0.6, 'rf':0.4,'ridge':0.7,'et':0.5}
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
    


# models = [typ6, typ1, typ9]   # typ16 och typ9 är samma förutom hur man väljer rader

def prepare_stack_data(stack_data_):
    """Hantera missing values, NaN, etc för meta-modellerna"""
    assert 'y' not in stack_data_.columns, "y shouldn't be in stack_data"
    
    if 'y' in stack_data_.columns:  # fejkad test!!
        raise Exception('uppdatera/tabort prepare_stack_data')

    stack_data = stack_data_.copy()
        
    # use the existing encoder - y is not used
    assert 'y' not in stack_data, "y shouldn't be in stack_data when encode"

    """ Fyll i saknade numeriska värden med 0 """
    numericals = stack_data.select_dtypes(exclude=['object']).columns
    stack_data[numericals] = stack_data[numericals].fillna(0)

    """ Fyll i saknade kategoriska värden med 'missing' """
    categoricals = stack_data.select_dtypes(include=['object']).columns
    stack_data[categoricals] = stack_data[categoricals].fillna('missing')

    # """ Hantera high cardinality """
    # cardinality_list=['häst','kusk','h1_kusk','h2_kusk','h3_kusk','h4_kusk','h5_kusk']
    
    # läs in encoder till ENC
    with open(pref+'modeller/xgb_encoder.pkl', 'rb') as f:
        ENC = pickle.load(f)

    df = stack_data.drop(columns=['startnr', 'datum', 'avd'])
    extra_columns = list(set(df.columns).difference(set(ENC.get_feature_names())))
    assert len(extra_columns) == 0, f"extra_columns i stack_data={extra_columns}"
    df = ENC.transform(df)
    stack_data = pd.concat([stack_data[['startnr','datum','avd']],df],axis=1)
    return stack_data

 
    
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
            # st.image('winning_horse.png')  # ,use_column_width=True)
            
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
    
            df_stack = build_stack_df(df_scraped, None)  # modeller )
            df_stack.to_csv('sparad_stack.csv', index=False)
            # use_meta(df_stack, meta)

    col1, col2 = st.columns([1,4])

    do_scraping=False
    with col1:          
        if st.button('scrape'):
            do_scraping=True
    with col2:  
        if st.button('reuse scrape'):
            st.session_state.df = None  # säkra att df blir ny
            try:
                df=pd.read_csv('sparad_scrape_spela.csv')
                
                if df.datum.iloc[0] != st.session_state.datum:
                    st.error(f'Datum i data = {df.datum.iloc[0]} \n\n är inte samma som i omgång') 
                else:    
                    # st.success(f'inläst data med datum = {temp_df.datum.iloc[0]}') 
                    st.info(f'inläst data med datum = {df.datum.iloc[0]} kör nu scrape med full=False')
                    try:
                        df.drop(['plac'], axis=1, inplace=True)
                    except:
                        pass
                    # scrape(False, meta=st.session_state['meta'])
                    # st.info('scrape klar')
                    del st.session_state.datum  # säkra att datum är samma som i scraping
            except:
                # write error message
                st.error('Det finns ingen sparad data') 
            
    
    if do_scraping:
        scrape(meta=st.session_state['meta'])
        del st.session_state.datum  # säkra att datum är samma som i scraping

    scraping.empty()
    

    if 'datum' not in st.session_state:
        omg_df = pd.read_csv('omg_att_spela_link.csv' )
        urlen=omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state.datum = datum
        
    st.title('🐎 v75 -  ' +st.session_state.datum)

    
with avd:
    if st.session_state.df is not None:
        # st.write(st.session_state.df    )
        use = avd.radio('Välj avdelning', ('Avd 1 och 2','Avd 3 och 4','Avd 5 och 6','Avd 7','clear'))
        avd.subheader(use)
        st.write('TA BORT OUTLIERS')
        col1, col2 = st.columns(2)
        # print(df.iloc[0].häst)
        dfi=st.session_state.df.copy()
        
        assert 'startnr' in dfi.columns, f"startnr finns inte i dfi"
        assert 'meta_predict' in dfi.columns, f"meta_predict finns inte i dfi"
        dfi.rename(columns={'startnr': 'nr', 'meta_predict': 'Meta'}, inplace=True)
        
    
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
                       ['nr', 'häst_', 'Meta', 'streck']])
            col2.table(dfi[(dfi.avd == 2) & dfi.välj].sort_values(by=['Meta'],ascending=False)[
                       ['nr', 'häst_', 'Meta', 'streck']])
        elif use=='Avd 3 och 4':
            col1.table(dfi[(dfi.avd == 3) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst_', 'Meta', 'streck']])
            col2.table(dfi[(dfi.avd == 4) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst_', 'Meta', 'streck']])
        elif use=='Avd 5 och 6':
            col1.table(dfi[(dfi.avd == 5) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst_', 'Meta', 'streck']])
            col2.table(dfi[(dfi.avd == 6) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst_', 'Meta', 'streck']])
        elif use=='Avd 7':
            col1.table(dfi[(dfi.avd == 7) & dfi.välj].sort_values(by=['Meta'], ascending=False)[
                       ['nr', 'häst_','häst', 'Meta', 'streck']])
        elif use=='clear':
            st.stop()    
        else:
            st.write('ej klart')
            
        st.write(compute_total_insats(dfi[dfi.välj]))

with sortera:   
    if st.sidebar.checkbox('se data'):
        dfr = st.session_state.df.copy()
        dfr.rename(columns={'startnr': 'nr', 'meta_predict': 'Meta'}, inplace=True)
        
        sort=st.sidebar.radio('sortera på',['Meta','avd'])
        if sort:
            if sort=='Meta':
                st.write(dfr[['avd', 'nr', 'häst_', 'häst', 'Meta', 'streck', 'välj']].sort_values(
                    by=['Meta', 'avd', 'nr'], ascending=[False, False, False]))
            else:
                dfra = dfr[['avd', 'nr', 'häst_', 'häst', 'streck', 'Meta', 'välj']]
                st.write(dfra.sort_values(by=['avd', 'nr'], ascending=[True, True]))
                
meta_list = ['rf', 'et', 'knn','ridge']
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
        st.info('Datum: ' + df_scraped.datum.iloc[0])
        pass
  
    df_stack = build_stack_df(df_scraped, L1_modeller)
    df_stack.to_csv('sparad_stack.csv', index=False)
    # use_meta(df_stack, meta)
