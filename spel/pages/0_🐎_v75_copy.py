
#%%

import numpy as np
import pandas as pd
from IPython.display import display

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

sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import typ as tp
import travdata as td
import skapa_modeller as mod
pref = ''
import logging
    
    
# %%

logging.basicConfig(level=logging.DEBUG, filemode='w', filename='v75.log', force=True,
                    encoding='utf-8', format='v75:' '%(asctime)s - %(levelname)s - %(lineno)d - %(message)s')
logging.info('Startar')

#%%
st.set_page_config(page_title="v75 Spel copy", page_icon="🐎")
st.sidebar.header("🐎 V75 Spel copy")

logging.info('Startar med att sätta alla saknade st.session_state till None')

#%%
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
#%%
def v75_scraping():
    logging.info('vs.v75_scraping: startar')
    print('start vs.v75_scraping')
    df = vs.v75_scraping(resultat=False, history=True, headless=True)
    logging.info('vs.v75_scraping: klar')
    for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
    return df

#%%
def do_scraping(datum):
    logging.info(f'scrape() - startar ny scraping')
    st.write('web-scraping för ny data')
    with st.spinner('Ta det lugnt!'):

        #####################
        # start v75_scraping as a thread
        #####################

        i = 0.0
        seconds = 0
        placeholder = st.empty()

        my_bar = st.progress(i)
        with concurrent.futures.ThreadPoolExecutor() as executor:
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

            # use_meta(df_stack, meta)
def build_stack_df(df):
    """ Bygg stack data som skall bli input till L2-modellerna
    Args:
        df (DataFram): Innehåller 'alla' features men endast L1_features kommer att användas 

    Returns:
        df_stack (DataFrame): df nu kompletterad med L1-modellernas prediktioner
        use_featuress (list): lista med features som används i L2-modellerna (L1_features + L1-modellernas prediktioner)
    """
    logging.info('build_stack_df(df) - startar')
    df_stack = df.copy()
    L1_features, cat_features, num_features = mod.read_in_features()
    
    df_stack, use_features = mod.create_L2_input(df_stack,st.session_state.L1_modeller, L1_features, with_y=False)
        
    logging.info('build_stack_df(df) - klar')   
    return df_stack, use_features

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
        if st.button('scrape'):
            st.session_state['scrape_type'] = 'scrape'
            st.session_state['df'] = do_scraping(datum)
    with col2:
        if st.button('Resue scrape'):
            st.session_state['scrape_type'] = 'resue scrape'
            logging.info('reuse Scraping')
            try:
                df = pd.read_csv('sparad_scrape_spela.csv')

                if df.datum.iloc[0] != st.session_state['datum']:
                    logging.error(f'Datum i data = {df.datum.iloc[0]} är inte samma som i omgång')
                    st.error(f'Datum i data = {df.datum.iloc[0]} är inte samma som i omgång')
                else:
                    logging.info(f'inläst data med datum = {df.datum.iloc[0]} kör nu med sparad_scrape')
                    st.info(f'inläst data med datum = {df.datum.iloc[0]} kör nu med sparad_scrape')
                    try:
                        df.drop(['plac'], axis=1, inplace=True)
                    except:
                        pass

                    st.session_state['df'] = df
                    # logging.info(f'Tar bort session_state["datum"]   - OK???' )
                    # del st.session_state['datum']  # säkrar att datum sätts samma som i df
            except FileNotFoundError:
                st.error('Det finns ingen sparad data')
                raise FileNotFoundError('Det finns ingen sparad data')
#############################################
#         streamlit kod följer              #
#############################################
v75 = st.container()
scraping = st.container()
avd = st.container()

with scraping:
    datum = date_handling()
    st.title('🐎 v75 -  ' + st.session_state.datum)
    get_scrape_data(datum)

if st.session_state['scrape_type'] is not None: 
    logging.info('build_stack_df(df) - startar')
    df_stack = st.session_state['df'].copy()
    L1_features, cat_features, num_features = mod.read_in_features()
    
    # Layer 1
    df_stack = mod.lägg_till_extra_kolumner(df_stack)
    df_stack = mod.fix_history_bana(df_stack)
    
    df_stack, use_features = mod.create_L2_input(df_stack,st.session_state.L1_modeller, L1_features, with_y=False)    
    logging.info('sparar df_stack till csv')
    df_stack.to_csv('sparad_stack.csv', index=False)
    
    # Layer 2
    df_stack = mod.predict_med_L2_modeller(st.session_state.L2_modeller, df_stack, use_features, mean_type='geometric', with_y = False)
    logging.info(f'Vi har en df_stack med meta_kolumn. shape = {df_stack.shape}')
    df_stack.to_csv('sparad_stack_meta.csv', index=False)
    
    st.info('veckans_rad är ännu inte klar - bara fejk')
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

    with avd:
        logging.info('välj avd')
        use = avd.radio('Välj avdelning', ('Avd 1 och 2', 'Avd 3 och 4', 'Avd 5 och 6', 'Avd 7', 'clear'))
        avd.subheader(use)
        col1, col2 = st.columns(2)

        assert 'meta' in df_stack.columns, f"meta_predict finns inte i df_stack"
        df_stack.rename(columns={'startnr': 'nr', 'meta': 'Meta'}, inplace=True)

        logging.info(f'display veckans rad för avd {use}')
        if use == 'Avd 1 och 2':
            col1.table(df_stack[(df_stack.avd == 1)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
            col2.table(df_stack[(df_stack.avd == 2)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
            # col1.table(df_stack[(df_stack.avd == 1) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'Meta', 'streck']])
            # col2.table(df_stack[(df_stack.avd == 2) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'Meta', 'streck']])
        elif use == 'Avd 3 och 4':
            col1.table(df_stack[(df_stack.avd == 3)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
            col2.table(df_stack[(df_stack.avd == 4)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
            # col1.table(df_stack[(df_stack.avd == 3) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'Meta', 'streck']])
            # col2.table(df_stack[(df_stack.avd == 4) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'Meta', 'streck']])
        elif use == 'Avd 5 och 6':
            col1.table(df_stack[(df_stack.avd == 5)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
            col2.table(df_stack[(df_stack.avd == 6)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
            # col1.table(df_stack[(df_stack.avd == 5) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'Meta', 'streck']])
            # col2.table(df_stack[(df_stack.avd == 6) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'Meta', 'streck']])
        elif use == 'Avd 7':
            col1.table(df_stack[(df_stack.avd == 7)].sort_values(by=['Meta'], ascending=False)[['nr', 'häst', 'Meta', 'streck']].head(3))
 
            # col1.table(df_stack[(df_stack.avd == 7) & df_stack.välj].sort_values(by=['Meta'], ascending=False)[
            #            ['nr', 'häst_', 'häst', 'Meta', 'streck']])
        elif use == 'clear':
            st.stop()
        else:
            st.write('ej klart')
