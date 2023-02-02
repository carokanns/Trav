
#%%

import numpy as np
import pandas as pd
from IPython.display import display

import concurrent.futures
import time
import datetime

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)
import streamlit as st

import sys
# import pickle
# import json

sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import typ as tp
import travdata as td
import skapa_modeller as mod
pref = ''
import logging
    
    
# %%

logging.basicConfig(level=logging.INFO, filemode='a', filename='v75.log', force=True,
                    encoding='utf-8', format='v75:' '%(asctime)s - %(levelname)s - %(lineno)d - %(message)s')
logging.info('V75.py Startar')

#%%
st.set_page_config(page_title="v75 Spel", page_icon="🐎")
st.sidebar.header("🐎 V75 Spel")
 
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

#%%
def v75_scraping():
    log_print('vs.v75_scraping: startar')
    print('start vs.v75_scraping')
    df = vs.v75_scraping(resultat=True, history=True, headless=True)
    log_print(f'vs.v75_scraping: klar: df.shape = ' + str(df.shape))
    for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
        
    if 'plac' in df.columns:
        log_print(f'HITTADE PLAC i df.columns')
    else:
        log_print(f'plac finns inte i df.columns')  
    return df

#%%
def do_scraping(datum):
    log_print(f'scrape() - startar ny scraping','i')
    
    st.write('web-scraping för ny data')
    placeholder = st.empty()
    i = 0.0
    seconds = 0
    my_bar = st.progress(i)
    with st.spinner('Ta det lugnt!'):

        #####################
        # start v75_scraping as a thread
        #####################

        with concurrent.futures.ThreadPoolExecutor() as executor:
            logging.info('scrape() - startar threaded v75_scraping')
            future = executor.submit(v75_scraping)
            while future.running():
                time.sleep(1)
                seconds += 1
                placeholder.write(f"⏳ {seconds} sekunder")
                i += 1/65
                if i < 0.99:
                    my_bar.progress(i)
            my_bar.progress(1.0)
            df_scraped = future.result()

            df_scraped.to_csv('sparad_scrape_spela.csv', index=False)

            st.balloons()
            my_bar.empty()
            placeholder.empty()
            logging.info(f'scrape() - klar med threaded v75_scraping - df_scraped.shape = {df_scraped.shape}')
    return df_scraped

#%%
def date_handling():  
    # Read date from CSV file or use current date
    try:
        omg_df = pd.read_csv('omg_att_spela_link.csv')
        urlen = omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state['datum'] = datum
    except FileNotFoundError:
        logging.info("File 'omg_att_spela_link.csv' not found, using current date")
        datum = datetime.date.today().strftime("%Y-%m-%d")
        st.session_state['datum'] = datum

    # Get date from user input
    year, month, day = map(int, datum.split("-"))
    datum = st.sidebar.date_input('Välj datum', datetime.date(year, month, day))
    datum = datum.strftime('%Y-%m-%d')

    # Check if date has changed and save it to CSV file
    if datum != st.session_state['datum']:
        logging.info(f"New date set to {datum}")
        st.session_state['datum'] = datum
        datum_url = "https://www.atg.se/spel/" + datum + "/V75/"
        omg_df = pd.DataFrame([datum_url], columns=['Link'])
        omg_df.to_csv('omg_att_spela_link.csv', index=False)
    return datum

#%%
def get_scrape_data(datum): 
    # Get data from CSV file or scrape it
    col1, col2 = st.columns([1, 4])
    st.session_state['scrape_type'] = None

    with col1:
        scrape_button = st.button('scrape')
    if scrape_button:
        st.session_state['scrape_type'] = 'scrape'
        st.session_state['df'] = do_scraping(datum)
 
    with col2:
        if st.button('Resue scrape'):
            st.session_state['scrape_type'] = 'resue scrape'
            log_print('reuse Scraping','i')
            try:
                df = pd.read_csv('sparad_scrape_spela.csv')

                if df.datum.iloc[0] != st.session_state['datum']:
                    logging.error(f'Datum i data = {df.datum.iloc[0]} är inte samma som i omgång')
                    st.error(f'Datum i data = {df.datum.iloc[0]} är inte samma som i omgång')
                else:
                    logging.info(f'inläst data med datum = {df.datum.iloc[0]} kör nu med sparad_scrape')
                    st.info(f'inläst data med datum = {df.datum.iloc[0]} kör nu med sparad_scrape')
                    st.session_state['df'] = df
            except FileNotFoundError:
                log_print('Det finns ingen sparad data','e')
                st.error('Det finns ingen sparad data')
                return False    # raise FileNotFoundError('Det finns ingen sparad data')
    return True

def mesta_diff_per_avd(X_):
    """Räknar ut den största och näst största diffen av meta per avd
    Args:
        X_ (DataFrame): veckans_rad (ej färdig)

    Returns:
        DataFrame: 1 rad per avd med kolumnerna [first, second, diff] tillagda
    """
    log_print(f'räknar ut mesta_diff_per_avd')
    df = X_.copy()
    # select the highest meta per avd
    df['first'] = df.groupby('avd')['meta'].transform(
        lambda x: x.nlargest(2).iloc[0])
    df['second'] = df.groupby('avd')['meta'].transform(
        lambda x: x.nlargest(2).iloc[1])

    df = df.dropna(subset=['first', 'second'])  # behåll endast first och second med värde. De andra är NaN
    df['diff'] = df['first'] - df['second']

    # drop duplicates per avd
    df = df.drop_duplicates(subset='avd', keep='first')

    df.sort_values(by='diff', ascending=False, inplace=True)
    # st.write(f'kolumnerna i df = {df.columns}')
    return df

def compute_total_insats(df):
    summa = df.groupby('avd').avd.count().prod() / 2
    return summa

def compute_weights(scores):
    """ Compute normalized weights to use to get the meta column. 
    Args:
        (list) scores: list of (i.e. F1 or AUC) scores from L2 models on validation set
    Returns:
        (list) weights: weights to use to get the meta column    
    """
    weights = scores / np.sum(scores)
    return weights

def välj_rad(df, max_insats=300):
    """_summary_
    Args:   df_ (dataframe): Predicted med Layer2. meta-kolumnen är den som ska användas
            max_insats (int): Max insats per omgång
    Returns:
        DataFrame: Kolumnen välj är True för de rader som ska spelas
        int: total insats
    """
    logging.info('Väljer rad')
    veckans_rad = df.copy()
    veckans_rad['välj'] = False   # inga rader valda ännu

    # first of all: select one horse per avd
    for avd in veckans_rad.avd.unique():
        max_pred = veckans_rad[veckans_rad.avd == avd]['meta'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (veckans_rad.meta == max_pred), 'välj'] = True
    
    veckans_rad = veckans_rad.sort_values(by=['meta'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)
    
    mest_diff = mesta_diff_per_avd(veckans_rad)
    # TODO: Kolla att mest_diff har exakt samma ordning som veckans rad
    assert len(mest_diff) == 7, \
            log_print(f'len(mest_diff) {len(mest_diff)} != 7 (antal lopp) {mest_diff[["avd","diff"]]}','e')
    
    assert set(mest_diff['avd'].unique()).issubset(veckans_rad['avd'].unique()), \
            log_print(f"Alla avd i mest_diff måste finnas i veckans_rad")
    assert len(mest_diff['avd'].unique()) == len(veckans_rad['avd'].unique()), \
            log_print(f"Antalet unique avd i mest_diff och i veckans_rad skall vara samma")
            
    cost = 0.5 # 1 rad
    
    # now select the rest of the horses one by one sorted by meta
    for i, row in veckans_rad.iterrows():
        if row.avd == mest_diff.avd.iloc[0]: 
            continue
        if row.avd == mest_diff.avd.iloc[1]: 
            continue
        
        veckans_rad.loc[i, 'välj'] = True
        cost = compute_total_insats(veckans_rad[veckans_rad.välj])
        if cost > max_insats:
            veckans_rad.loc[i, 'välj'] = False        
            break
    
    cost = compute_total_insats(veckans_rad[veckans_rad.välj])
    veckans_rad.sort_values(by=['välj', 'avd'], ascending=[False, True], inplace=True)

    return veckans_rad, cost           
            
            
#############################################
#         streamlit kod följer              #
#############################################

v75 = st.container()
scraping = st.container()
avd = st.container()
    
logging.info('Startar med att sätta alla saknade st.session_state till None')

if 'use' not in st.session_state:
    st.session_state['use'] = None
if 'rätta' not in st.session_state:
    st.session_state['rätta'] = False    
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'datum' not in st.session_state:
    st.session_state['datum'] = None
if 'scrape_type' not in st.session_state:
    st.session_state['scrape_type'] = None
if 'mean_type' not in st.session_state:
    st.session_state['mean_type'] = None    # ['arithmetic', 'geometric']
if 'L1_modeller' not in st.session_state:
    st.session_state['L1_modeller'], st.session_state['L2_modeller'] = mod.skapa_modeller()
        
with scraping:
    logging.info('Nu kör vi "with scraping"')
    datum = date_handling()
    st.title('🐎 v75 -  ' + st.session_state.datum)
    if get_scrape_data(datum) == False:
        st.session_state['scrape_type'] = None

if st.session_state['scrape_type'] is not None: 
    logging.info('Nu bygger vi df_stack_L1')
    df_stack_L1 = st.session_state['df'].copy()
    L1_features, cat_features, num_features = mod.read_in_features()
    
    # Layer 1
    df_stack_L1 = mod.lägg_till_extra_kolumner(df_stack_L1)
    df_stack_L1 = mod.fix_history_bana(df_stack_L1)
    
    df_stack_L1, use_features = mod.create_L2_input(df_stack_L1, st.session_state.L1_modeller, L1_features, with_y=False)
    logging.info('sparar df_stack_L1 till csv')
    df_stack_L1.to_csv('sparad_stack_L1.csv', index=False)
    
    # Layer 2
    logging.info('Nu bygger vi df_stack_L2')
    df_stack_L2 = mod.predict_med_L2_modeller(  \
        st.session_state.L2_modeller, df_stack_L1, use_features, mean_type='geometric', with_y=False)
    logging.info(
        f'Vi har en df_stack_L2 med meta_kolumn. shape = {df_stack_L2.shape}')
    df_stack_L2.to_csv('sparad_stack_L2.csv', index=False)
    
with avd:     
    try:
        df_stack_L2 = pd.read_csv('sparad_stack_L2.csv')
    except:
        df_stack_L2 = None    
    
    if df_stack_L2 is None :
        st.warning('Ingen df_stack_L2 hittades')
        st.stop()
    else:    
        if df_stack_L2.iloc[0].datum == st.session_state['datum']:    
            use = avd.radio('Välj avdelning', ('Avd 1 och 2', 'Avd 3 och 4', 'Avd 5 och 6', 'Avd 7', 'clear'))
            avd.subheader(use)
            col1, col2 = st.columns(2)
            if 'plac' in df_stack_L2.columns:
                if st.session_state['rätta'] == False:
                    st.session_state['rätta'] = avd.button('Rätta raden')
            else:
                st.session_state['rätta'] = False
                
            veckans_rad, kostnad = välj_rad(df_stack_L2, max_insats=375)
            assert 'meta' in veckans_rad.columns, f"meta finns inte i veckans_rad"
            veckans_rad.rename(columns={'startnr': 'nr', 'meta': 'Meta'}, inplace=True)
            
            st.info(f'Kostnad för veckans rad: {kostnad:.0f} kr')
            
            log_print(f'Visar veckans rad för {use}','i')
            
            header_list = ['nr', 'häst', 'Meta', 'streck']
            if st.session_state['rätta']:
                veckans_rad.to_csv('veckans_rad.csv', index=False)
                header_list = ['nr', 'häst', 'Meta', 'streck', 'plac']
                sju, sex, fem, utd = mod.rätta_rad(veckans_rad, datum)
                vinst = utd - kostnad
                vinst_text= "vinst"
                if vinst<0:
                    vinst_text = 'förlust'

                st.info(f'utdelning={utd:.0f}, {vinst_text}={(vinst):.0f} kr, {sju} sjuor {sex} sexor {fem} femmor')
            ################################################
            #    Denna kod döljer radindex i dataframe     #
            #    När man t.ex. gör en st.write(df)         #
            ################################################
            # CSS to inject contained in a strin
            hide_dataframe_row_index = """
                    <style>
                    .row_heading.level0 {display:none}
                    .blank {display:none}
                    </style>
                    """
            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            ############ CSS klart ###########################
            if use == 'Avd 1 och 2':
                col1.table(veckans_rad[(veckans_rad.avd == 1) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
                col2.table(veckans_rad[(veckans_rad.avd == 2) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
            elif use == 'Avd 3 och 4':
                col1.table(veckans_rad[(veckans_rad.avd == 3) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
                col2.table(veckans_rad[(veckans_rad.avd == 4) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
            elif use == 'Avd 5 och 6':
                col1.table(veckans_rad[(veckans_rad.avd == 5) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
                col2.table(veckans_rad[(veckans_rad.avd == 6) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
            elif use == 'Avd 7':
                col1.table(veckans_rad[(veckans_rad.avd == 7) & veckans_rad.välj].sort_values(
                    by=['Meta'], ascending=False)[header_list])
            elif use == 'clear':
                st.stop()
            else:
                st.write('ej klart')
        else:
            st.warning(f"datum i session_state={st.session_state['datum']} är inte samma som i sparad_stack_L2={df_stack_L2.iloc[0].datum}")
            st.stop()
