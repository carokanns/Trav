#%%
import logging
import numpy as np
import pandas as pd
import json
import datetime
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from IPython.display import display
import concurrent.futures
import V75_scraping as vs
import travdata as td
import typ as tp
import streamlit as st
import skapa_modeller as mod

import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')


# %%

logging.basicConfig(level=logging.INFO, filemode='w', filename='v75.log', force=True,
                    encoding='utf-8', format='Learn: %(asctime)s - %(levelname)s - %(lineno)d - %(message)s ')

logging.info('Startar')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

# sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')

pref = ''   # '../'

###################################################################################

st.set_page_config(page_title="V75 Learning", page_icon="🏫")

st.markdown("# 🏫 V75 Learning")
st.sidebar.header("🏫 V75 Learning")

#%%


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


L1_modeller, L2_modeller = mod.skapa_modeller()


# %%
################################################
#              Web scraping                    #
################################################

def v75_scraping():
    df = vs.v75_scraping(history=True, resultat=True, headless=True)
    logging.info(f'Antal hästar shape: {df.shape}')
    logging.info(f'plac finns i df: {"plac" in df.columns}')
    for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()

    return df

###############################################
#              LEARNING                       #
###############################################

# TODO: egen funktion i skapa_modeller.py


# %%

def normal_learn_meta_models(meta_modeller, L2_input_data, save=True):

    assert 'y' in L2_input_data.columns, 'y is missing in stack_data'

    y = L2_input_data.y.astype(int)
    meta_features = L2_input_data.drop(
        ['datum', 'avd', 'y'], axis=1).columns.to_list()
    
    assert 'y' not in L2_input_data.columns, "y shouldn't be in stack_data"
    assert len([item for item in L2_input_data.columns if 'proba' in item]) == 4, "4 proba should be in stack_data"
    X_meta = L2_input_data[meta_features].copy(deep=True)

    y_meta = y
    for key, items in meta_modeller.items():
        meta_model = items['model']

        items['model'] = meta_model.fit(X_meta, y_meta)
        meta_modeller[key] = items
        meta_modeller[key]['model'] = meta_model

        if save:
            # Save the model to a pckle file
            with open(pref+'modeller/'+key+'.model', 'wb') as f:
                pickle.dump(meta_model, f)

            # Save the list of column names to a text file
            with open(pref+'modeller/'+key+'_columns.txt', "w", encoding="utf-8") as f:
                for col in X_meta.columns.tolist():
                    f.write(col + '\n')

    return meta_modeller
# %%

def skapa_data_för_datum(df_, curr_datum_ix, frac=0.5):
    df = df_.copy()
    datumar = df.datum.unique()
    curr_datum = datumar[curr_datum_ix]
    base_datum_ix = int(len(datumar[:curr_datum_ix]) * frac)  # base models

    base_datum = datumar[base_datum_ix]
    X_train = df.query(f'datum < @base_datum')
    y_train = X_train.y
    X_train = X_train.drop('y', axis=1)

    X_meta = df.query(f'datum >= @base_datum and datum < @curr_datum')
    y_meta = X_meta.y
    X_meta = X_meta.drop('y', axis=1)

    X_curr = df.query(f'datum == @curr_datum')
    y_curr = X_curr.y
    X_curr = X_curr.drop(['y'], axis=1)

    return X_train, y_train, X_meta, y_meta, X_curr, y_curr


def normal_skapa_stack_data(model, name, X_meta, stack_data, use_features):
    """Skapa stack_data"""
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    this_proba = model.predict(X_meta, use_features)

    # Bygg up meta-kolumnerna (proba) för denns modell
    nr = name[3:]
    stack_data['proba'+nr] = this_proba
    return stack_data


def hold_out_val_data(df_work, val_fraction):
    """ Spara validation data till senare för Layer2"""
    y = df_work.y
    X = df_work.drop('y', axis=1)
    if val_fraction == 0:
        # no validation data
        X_val, y_val = None, None
    else:
        # use a fraction for meta data
        datumar = df_work.datum.unique()
        val_antal = int(len(datumar)*val_fraction)
        val_datum = datumar[-val_antal:]

        X_val = X.loc[X.datum.isin(val_datum)]
        y_val = y[X_val.index]
        X = X.loc[~X.datum.isin(val_datum)]
        y = y.loc[X.index]
    return X, y, X_val, y_val

# %%


def normal_learn_modeller(modeller, X_train, y_train, X_meta, y_meta):
    """ Normal Learning (ej TimeSeriesCV) av modeller"""
    ############################################################################################################
    #                        Här görs en första learn av modeller och sedan skapas stack_data
    #                        - Learn modeller på X,y
    #                        - Ha en egen skapa_stack_funktion (som också används längre ner)
    #                           - Skapa stack_data med predict X_meta med nya modellerna
    #                           - Spara även X_meta, y_meta i stack_data
    ############################################################################################################
    stack_data = X_meta.copy()
    stack_data['y'] = y_meta
    assert 'y' in stack_data.columns, '1. y is missing in stack_data'
    for model in modeller:
        name = model.name
        print(
            f'first Learn {name} {X_train.datum.min()} -{X_train.datum.max()}')

        model.learn(X_train, y_train, params=None, save=True)

        stack_data = normal_skapa_stack_data(model, name, X_meta, stack_data)

    assert 'y' in stack_data.columns, '3. y is missing in stack_data'
    # stack_data, enc = prepare_stack_data(stack_data)

    return stack_data


def normal_learning(modeller, meta_modeller, X_train, y_train, X_meta, y_meta):
    stack_data = normal_learn_modeller(
        modeller, X_train, y_train, X_meta, y_meta)
    assert 'y' in stack_data.columns, 'y is missing in stack_data'
    stack_data.to_csv('first_stack_data.csv', index=False)

    use_features,_,_ = mod.read_in_features()

    meta_modeller = normal_learn_meta_models(meta_modeller, stacked_data, use_features)

    return stack_data

# %%
# TimeSeriesSplit learning models


def TimeSeries_learning(df_ny_, L1_modeller, L2_modeller, n_splits=5, val_fraction=0.25, save=True):
    """
    Skapar en stack med {1 - val_fraction} av X från Layer1. Används som input till Layer2.
        - learn_models=True betyder att vi både gör en learning och skapar en stack
        - learn_models=False betyder att vi bara skapar en stack och då har param save ingen funktion
    """

    # Skapa v75-instans
    v75 = td.v75(pref=pref)

    base_features = v75.get_df().columns.to_list() # all features in alla_data.csv

    if df_ny_ is not None:  # Har vi en ny omgång?
        df_ny = df_ny_[base_features].copy()
        v75.concat(df_ny, update_work=True, save=True)

    use_features,_,_ = mod.read_in_features()  # De features som används i modellerna
    # Hämta data från v75
    df, enc = v75.förbered_data(missing_num=False)  # num hanteras av catboost
    df_work = v75.test_lägg_till_kolumner(df)

    X, y, X_val, y_val = hold_out_val_data(df_work, val_fraction)

    validation_text = ""

    if X_val is not None:
        validation_text = f', Validation: {X_val.datum.iloc[0]} - {X_val.datum.iloc[-1]}'
        print('start datum i X_val:', X_val.datum.min())

    st.info(f'Train: {X.datum.iloc[0]} - {X.datum.iloc[-1]} {validation_text}')

    ts = TimeSeriesSplit(n_splits=n_splits)

    L1_output_data = pd.DataFrame()

    ########################################################################################
    #         Step 1: Learn the Layer1 on ts X_train and predict on ts X_test              #
    ########################################################################################
    st.write('Skapar stacked_data till Layer2')
    my_bar = st.progress(0)

    step = 1/(n_splits*len(L1_modeller))-0.0000001
    steps = 0.0

    for enum, (train_index, test_index) in enumerate(ts.split(X, y)):
        print(f"enum: {enum}, shape of X: {X.shape}, shape of X_train: {X.iloc[train_index].shape}, shape of X_test: {X.iloc[test_index].shape}")
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        temp_stack = X_test.copy()
        temp_stack['y'] = y_test

        print('start datum i X_train:', X_train.datum.min())
        print('start datum i X_test:', X_test.datum.min())

        ###### Learn L1-modeller på X_train från ts.split#######

        for model_name, model in L1_modeller.items():
            steps += step
            # progress bar continues to complete from 0 to 100

            my_bar.progress(steps)

            with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
                params = json.load(f)
                params = params['params']

            logging.info(f'# learn {model_name} Layer1 på X_train-delen')

            my_model = model.learn(X_train, y_train, X_test,
                                   y_test, params=params, save=save)

        # Alla L1_modeller är nu tränade på X_train från ts.split 
        # och vi kan nu göra predict av L!_modellerna på X_test från ts.split
        
        assert 'streck' in X.columns, f'streck is missing in X före create_L2_data med L1_modeller'
        Xy = X_test.copy()
        Xy['y'] = y_test
        L1_output_data, use_L2features = mod.create_L2_input(Xy, L1_modeller, use_features)
        
        assert 'streck' in L1_output_data.columns, f'streck is missing in L1_output_data efter predict med {model_name}'
        
    # Nu har vi en stack från alla L1_modeller som ska användas som input till L2_modeller
    
    # Kolla att vi har korrekta proba-kolumner i L1_output_data
    proba_features = [col for col in L1_output_data.columns if 'proba_' in col]
    assert len(proba_features) == 4, f'proba-kolumner saknas i L1_output_data'
    
    my_bar.progress(1.0)

    ###############################################################################
    #         Step 2:       Learn Layer2                                          #
    ###############################################################################
    st.write('Learning L2 models')
    # use_L2features = use_features + proba_features   - use_L2features är redan satt i create_L2_input
    assert 'streck' in use_L2features, f'streck is missing in use_L2features innan learn L2_modeller'
    assert 'datum' in L1_output_data, 'datum is missing in L1_output_data'
    assert len([col for col in use_L2features if 'proba_' in col]) == 4,f' proba-kolumner saknas för L2-modell {model_name}'
    
    L2_modeller = mod.learn_L2_modeller(L2_modeller, L1_output_data, use_L2features)
    

    ###############################################################################
    #         Step 3: learn L1-modeller on all of X - what iteration to use?      #
    ###############################################################################
    st.write('Learn models on all of X - why not plus X_test as well?')
    print('Starting to learn models on all of X')
    my_bar2 = st.progress(0)
    ant_meta_models = 4
    step = 1/(ant_meta_models) - 0.0000001
    steps = 0.0
    my_bar2.progress(steps)

    for model_name, model in L1_modeller.items():
        steps += step
        my_bar2.progress(steps)
        with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
            params = json.load(f)
            params = params['params']

        assert 'datum' in X, 'datum is missing in X before learn L1_model on all of X'
        tot_mod = model.learn(X, y, None, None,
                              iterations=500,
                              params=params,
                              save=save)

    my_bar2.progress(1.0)
    st.empty()

    assert 'datum' in L1_output_data, 'datum is missing in L1_output_data that will be returned'
    return L1_output_data


def validate_skapa_stack_learning(X_, y, use_features):
    # För validate
    X = X_.copy()
   
    assert len(set(use_features)) == len(use_features), f' 0a use_features has doubles: {use_features}'
    temp_use_features = use_features.copy()
    stacked_data = X.copy()
    for model_name, model in L1_modeller.items():
        part = model_name[2:]
        stacked_data['proba'+part] = model.predict(X, use_features)
        temp_use_features.append('proba'+part)
        
    use_features = temp_use_features.copy()
    assert len(set(use_features)) == len(use_features), f' 0b use_features has doubles: {use_features}'

    missing_items = [item for item in use_features if item not in stacked_data.columns]
    assert not missing_items, f"The following items in 'use_features' are not found in 'std_columns': {missing_items}"

    assert len(stacked_data) == len(y), f'stacked_data {len(stacked_data)} and y {len(y)} should have same length'
    return stacked_data, use_features, y


# %%


### write the scores ###
def display_scores(y_true, y_pred, spelade):
    st.write('AUC', round(roc_auc_score(y_true, y_pred), 5), 'F1', round(f1_score(y_true, y_pred), 5), 'Acc', round(
        accuracy_score(y_true, y_pred), 5), 'MAE', round(mean_absolute_error(y_true, y_pred), 5), '\n', spelade)
    return roc_auc_score(y_true, y_pred)


def find_threshold(y_pred, fr, to, margin):
    """ hitta threshold som ger 2.5 spelade per avdelning """
    thresh = 0
    cnt = 0
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

    print('ant', cnt, 'thresh', round(thresh, 4))
    if cnt >= 1000:
        print('threshold not found', 'fr', round(fr, 6), 'to', round(to, 6))

    return thresh


def plot_confusion_matrix(y_true, y_pred, typ, fr=0.0, to=0.9, margin=0.001):

    #### Först:  hitta ett threshold som tippar ca 2.5 hästar per avd ####
    # thresh = 0
    # for thresh in np.arange(fr, to, step):
    #     cost = 12*sum(y_pred > thresh)/len(y_pred)
    #     if cost < 2.5:
    #         break
    thresh = round(find_threshold(y_pred, fr, to, margin), 4)
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
        meta_scores = {'knn': 0, 'lasso': 0, 'rf': 0, 'ridge': 0, 'et': 0}

    #### print scores ####
    typ_AUC = display_scores(
        y_true, y_pred, f'spelade per lopp: {round(12 * sum(y_pred)/len(y_pred),4)}')
    meta_scores[typ] = float(typ_AUC)
    #### save dict to disk ####
    with open(pref+'modeller/meta_scores.pkl', 'wb') as f:
        pickle.dump(meta_scores, f)


def get_data_for_validate(fraction):
    v75 = td.v75(pref=pref)

    # Hämta data från v75
    df_work,enc = v75.förbered_data(missing_num=False)  # num hanteras av catboost
    df_work = v75.test_lägg_till_kolumner(df_work)

    _, _, X_val, y_val = hold_out_val_data(
        df_work, fraction)  # validera mot detta
    st.info(f'Validerar på:  {X_val.datum.iloc[0]} - -{X_val.datum.iloc[-1]}')

    # X och y i samma dataframe
    Xy_val = X_val.copy(deep=True)
    Xy_val['y'] = y_val
    
    return Xy_val


def validate(L1_modeller, L2_modeller, fraction=None):
    # TODO: Varför blir xgb så dåliga och varför blir L1 bättre än L2?
    logging.info('startar validate-funktionen')
    st.info('skall endast  köras efter "Learn for Validation"')    
    
    logging.info(f'L1_modeller till validate {L1_modeller.keys()}')
    logging.info(f'L2_modeller till validate {L2_modeller.keys()}')
    
    Xy_val = get_data_for_validate(fraction)
    use_features,_,_ = mod.read_in_features()
    
    logging.info(f'validate use_features:\n {use_features}')
    assert len(set(use_features)) == len(use_features), f'use_features shoudnt have doubles innan create_L2_input: {use_features}'

    L2_input, L2_features = mod.create_L2_input(Xy_val, L1_modeller, use_features)
    
    L2_output = mod.predict_med_L2_modeller(L2_modeller, L2_input, L2_features, mean_type=st.session_state['mean_type'])
    
    y_true = L2_output['y']
    y_preds = L2_output['meta']
    
    logging.info('plot confusion matrix för meta-predictions')
    st.info('plot confusion matrix för meta-predictions')
    plot_confusion_matrix(y_true, y_preds, 'meta', fr=0.0, to=0.9)

    logging.info(f'plot confusion matrix for L2 output kolumner: {L2_output.columns}')
    logging.info(f'L2-modeller:    {L2_modeller.keys()}')
    st.info('plot confusion matrix for L2-modeller')
    
    # filtrera ut kolumner som börjar med proba_ och slutar med L2
    # Bara för att stila! Kunde lika gärna göra som med L!-modeller nedan
    for model in L2_output.filter(regex="^proba_.*L2$", axis=1).columns:
        st.write('\n')
        st.info(f'förbereder {model} plot')
        plot_confusion_matrix(y_true, L2_output[model], model, fr=0.0, to=0.9)

    st.write('\n')
    logging.info(f'plot confusion matrix for L1-modeller {L1_modeller.keys()}')
    st.info('plot confusion matrix for L1-modeller')

    st.write('\n')
    for model_name in L1_modeller:
        st.write('\n')
        proba_col = 'proba_' + model_name
        y_pred = L2_input[proba_col]
        plot_confusion_matrix(y_true, y_pred, proba_col, fr=0.0, to=0.9)

# %%
##############################################################
#            FINAL LEARNING                                  #
##############################################################


def final_learning(modeller, meta_modeller, n_splits=5):
    st.info('Final learning on all the data')

    _ = TimeSeries_learning(None,
                            modeller,
                            meta_modeller,
                            n_splits=n_splits,
                            val_fraction=0,
                            save=True)

    # st.info('Step 2: Final learn meta model')
    # ENC = learn_meta_models(stacked_data.drop(['y'], axis=1), stacked_data['y'])

    st.success('✔️ Final learning done')


# %%
def scrape(full=True):
    scraping.write('Starta web-scraping för ny data')
    with st.spinner('Ta det lugnt!'):
        # st.image('winning_horse.png')  # ,use_column_width=True)
        #####################
        # start v75_scraping as a thread
        #####################
        i = 0.0
        placeholder = st.empty()
        seconds = 0
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
            scraping.write('✔️ Scraping done, pls wait')
            time.sleep(2)
            df = future.result()
            scraping.write(f'✔️ {len(df)} rader hämtade')

            df.to_csv('sparad_scrape_learn.csv', index=False)

        st.balloons()
        my_bar.empty()
        placeholder.empty()

        st.session_state.df_ny = df


def position_to_top():
    st.write("""
        <script>
            window.scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)
    

st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
# %%
top = st.container()
buttons = st.container()
scraping = st.container()

############################################
#   Init session_state                     #
############################################
with top:
    if 'fraction' not in st.session_state:
        st.session_state['fraction'] = 0.25

    if 'df_ny' not in st.session_state:
        st.session_state['df_ny'] = None
    if 'mean_type' not in st.session_state:
        st.session_state['mean_type'] = 'geometric'
    if 'datum' not in st.session_state:
        omg_df = pd.read_csv('omg_att_spela_link.csv')
        urlen = omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state.datum = datum

    if 'datum' in st.session_state:
        datum = st.session_state['datum']
        year = int(datum[:4])
        month = int(datum[5:7])
        day = int(datum[8:])
        datum = st.sidebar.date_input(
            'Välj datum', datetime.date(year, month, day))
        datum = datum.strftime('%Y-%m-%d')

        if datum != st.session_state['datum']:
            st.session_state['datum'] = datum
            datum = "https://www.atg.se/spel/"+datum+"/V75/"
            omg_df = pd.DataFrame([datum], columns=['Link'])
            omg_df.to_csv('omg_att_spela_link.csv', index=False)

    st.header(f'Omgång:  {st.session_state.datum}')

###########################################
# control flow with buttons               #
###########################################
position_to_top()

with buttons:
    # TODO: Inför två streamlit-val att köra learn.py antingen med TimeSeriesSplit eller Train_Test_Split
    # TODO: använd sparad_scrape_spela.csv om den finns och om den innehåller plac
    if st.sidebar.button('scrape'):
        st.write(f'web scraping {st.session_state.datum}')
        try:
            scrape()
            del st.session_state.datum  # säkra att datum är samma som i scraping
        except:
            st.error(f"Fel i web scraping. Kolla att resultat finns för datum och internet är tillgängligt")

    if st.sidebar.button('reuse scrape'):
        try:
            df_ny = pd.read_csv('sparad_scrape_learn.csv')
            st.session_state.df_ny = df_ny
            if df_ny.datum.iloc[0] != st.session_state.datum:
                logging.error(f'Datum i data = {df_ny.datum.iloc[0]} \n\n är inte samma som i omgång')
                st.error(f'Datum i data = {df_ny.datum.iloc[0]} \n\n är inte samma som i omgång')
            else:
                logging.info(f'inläst data med datum = {df_ny.datum.iloc[0]}')
                st.success(f'inläst data med datum = {df_ny.datum.iloc[0]}')
        except:
            # write error message
            logging.error('Ingen data sparad')
            st.error('Ingen data sparad')

    if st.session_state.df_ny is not None:
        if st.sidebar.button('Learn for validation'):    
            st.write('TimeSeries learning for validation')
            fraction = st.session_state.fraction
            df_ny = st.session_state.df_ny
            st.write(
                f'learn models and meta models on first {(1-fraction)*100} % of the data')

            stacked_data = TimeSeries_learning(df_ny,
                                               L1_modeller, L2_modeller,
                                               n_splits=5,
                                               val_fraction=fraction,
                                               save=True)

            assert 'datum' in stacked_data.columns, 'datum saknas i stacked_data'
            assert 'streck' in stacked_data.columns, 'datum saknas i stacked_data'
            assert len([col for col in stacked_data.columns if 'proba' in col]
                       ) == 4, 'proba saknas i stacked_data'
            st.success('✔️ TimeSeries learning done')

        st.session_state['mean_type'] = st.sidebar.selectbox('Välj typ av medelvärde', ['arithmetic', 'geometric'], index=1)

        if st.sidebar.button('Validate'):
            validate(L1_modeller, L2_modeller, fraction=st.session_state.fraction)
            
        if st.sidebar.button('Final learning'): 
            st.write("""
                <script>
                    setTimeout(function() {
                        window.scrollTo(0, 0);
                    }, 10);
                </script>
            """, unsafe_allow_html=True)
                    
            final_learning(L1_modeller, L2_modeller)

        if st.sidebar.button('Clear'):
            st.empty()
    st.markdown("<a href='#linkto_top'>Top of page</a>", unsafe_allow_html=True)