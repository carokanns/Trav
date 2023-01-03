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

import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')


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
# %%
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

# %%
################################################
#              Web scraping                    #
################################################


def v75_scraping():
    df = vs.v75_scraping(history=True, resultat=True, headless=True)

    for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
    return df

###############################################
#              LEARNING                       #
###############################################


def prepare_L2_input_data(L2_input_data_, y, use_L2features):
    L2_input_data = data = L2_input_data_.copy(deep=True)
    assert 'y' not in L2_input_data.columns, "y shouldn't be in stack_data"
    assert len([item for item in L2_input_data.columns if 'proba' in item]) == 4, "4 proba should be in stack_data"

    print([item for item in L2_input_data.columns if 'proba' in item])
    return L2_input_data


def learn_L2_modeller(L2_modeller, L2_input_data, use_L2features, save=True):
    assert 'y' in L2_input_data.columns, 'y is missing in L2_input_data'
    
    y_meta = L2_input_data.pop('y').astype(int)
    X_meta = prepare_L2_input_data(L2_input_data, y_meta, use_L2features)  # gör inget särskilt just nu

    for model_name, model in L2_modeller.items():

        with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
            params = json.load(f)
            params = params['params']

        print(f'# learn {model_name} Layer2 på L2_input_data (stack-data)')

        assert 'streck' in use_L2features, f'streck is missing in use_features innan Learn med {model.name}'
        my_meta = model.learn(X_meta[use_L2features], y_meta, params=params, save=save)
        assert 'streck' in use_L2features, f'streck is missing in use_features efter Learn med {model.name}'

        L2_modeller[model_name] = my_meta
        
        if save:
            # Save the model to a pckle file
            with open(pref+'modeller/'+model_name+'.model', 'wb') as f:
                pickle.dump(model, f)

            # Save the list of column names to a JSON file
            with open(pref+'modeller/'+key+'_columns.json', "w") as f:
                json.dump(X_meta.columns.tolist(), f)

    return L2_modeller


# def learn_meta_models(Layer2_modeller, stack_data, use_L2features, save = True):

#     assert 'y' in stack_data.columns, 'y is missing in stack_data'

#     y_meta=stack_data.pop('y').astype(int)
#     X_meta, ENC=prepare_stack_data(
#         stack_data, y_meta, use_L2features)  # gör inget just nu

#     with open(pref+'modeller/meta_encoder.pkl', 'wb') as f:
#         pickle.dump(ENC, f)

#     for model_name, model in L1_modeller.items():
#     # for key, items in Layer2_modeller.items():
#         Layer2_model = items['model']

#         items['model'] = Layer2_model.fit(X_meta, y_meta)
#         Layer2_modeller[key] = items
#         Layer2_modeller[key]['model'] = Layer2_model

#         if save:
#             # Save the model to a pckle file
#             with open(pref+'modeller/'+key+'.model', 'wb') as f:
#                 pickle.dump(Layer2_model, f)

#             # Save the list of column names to a JSON file
#             with open(pref+'modeller/'+key+'_columns.json', "w") as f:
#                 json.dump(X_meta.columns.tolist(), f)

#     return Layer2_modeller
# %%


def normal_learn_meta_models(meta_modeller, L2_input_data, save=True):

    assert 'y' in L2_input_data.columns, 'y is missing in stack_data'

    y = L2_input_data.y.astype(int)
    meta_features = L2_input_data.drop(
        ['datum', 'avd', 'y'], axis=1).columns.to_list()
    X_meta, ENC = prepare_L2_input_data(L2_input_data[meta_features], y)

    with open(pref+'modeller/meta_encoder.pkl', 'wb') as f:
        pickle.dump(ENC, f)

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

            # Save the list of column names to a JSON file
            with open(pref+'modeller/'+key+'_columns.json', "w") as f:
                json.dump(X_meta.columns.tolist(), f)

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
    # print(f'X_meta.shape = {X_meta.shape} this_proba.shape={this_proba.shape}')

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

    # Läs in NUM_FEATURES.txt och CAT_FEATURES.txt
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()

    # läs in NUM_FEATURES.txt till num_features
    with open(pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
        num_features = f.read().split()
    use_features = cat_features + num_features

    meta_modeller = normal_learn_meta_models(
        meta_modeller, stacked_data, use_features)

    return stack_data

# %%
# TimeSeriesSplit learning models


def TimeSeries_learning(df_ny_, L1_modeller, L2_modeller, n_splits=5, val_fraction=0.25, save=True):
    """
    Skapar en stack med {1 - meta_fraction} av X från Layer1. Används som input till Layer2.
        - learn_models=True betyder att vi både gör en learning och skapar en stack
        - learn_models=False betyder att vi bara skapar en stack och då har param save ingen funktion
    """

    # Skapa v75-instans
    v75 = td.v75(pref=pref)

    base_features = v75.get_df().columns.to_list()

    if df_ny_ is not None:  # Har vi en ny omgång?
        df_ny = df_ny_[base_features].copy()
        v75.concat(df_ny, update_work=True, save=True)

    # Läs in NUM_FEATURES.txt och CAT_FEATURES.txt
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()

    # läs in NUM_FEATURES.txt till num_features
    with open(pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
        num_features = f.read().split()
        
    use_features = cat_features + num_features

    # Hämta data från v75
    _ = v75.förbered_data(missing_num=False)  # num hanteras av catboost
    df_work = v75.test_lägg_till_kolumner()

    X, y, X_val, y_val = hold_out_val_data(df_work, val_fraction)

    validation_text = ""

    if X_val is not None:
        validation_text = f', Validation: {X_val.datum.iloc[0]} - {X_val.datum.iloc[-1]}'

    st.info(f'Train: {X.datum.iloc[0]} - {X.datum.iloc[-1]} {validation_text}')
    # print(f"Train: {X.datum.iloc[0]} - {X.datum.iloc[-1]} {validation_text}")

    ts = TimeSeriesSplit(n_splits=n_splits)

    L1_output_data = pd.DataFrame()

    ########################################################################################
    #         Step 1: Learn the Layer1 on splitted X_train and predict on splitted X_test  #
    ########################################################################################
    st.write('Skapar stacked_data till Layer2')
    my_bar = st.progress(0)

    step = 1/(n_splits*len(L1_modeller))-0.0000001
    steps = 0.0

    for enum, (train_index, test_index) in enumerate(ts.split(X, y)):
        print('shape of X', X.shape, 'shape of X_train',
              X.iloc[train_index].shape, 'shape of X_test', X.iloc[test_index].shape)
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        temp_stack = X_test.copy()
        temp_stack['y'] = y_test

        ###### Learn L1-modeller #######

        for model_name, model in L1_modeller.items():
            steps += step
            # progress bar continues to complete from 0 to 100

            my_bar.progress(steps)

            with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
                params = json.load(f)
                params = params['params']

            print(f'# learn {model_name} Layer1 på X_train-delen')
            assert 'streck' in use_features, f'streck is missing in use_features innan Learn med {model.name}'
            
            my_model = model.learn(X_train, y_train, X_test,
                                   y_test, params=params, save=save)
            assert 'streck' in use_features, f'streck is missing in use_features efter Learn med {model.name}'

            print(f'# predict the new fitted {model_name} Layer1 on X_test-delen')
            nr = model.name[2:]

            assert 'streck' in use_features, f'streck is missing in use_features innan predict med {nr}'

            this_proba = model.predict(X_test, use_features, verbose=False)
            assert 'streck' in use_features, f'streck is missing in use_features efter predict med {model_name}'
            # Bygg up meta-kolumnen proba för denns modell
            temp_stack['proba'+nr] = this_proba

        if L1_output_data.empty:
            L1_output_data = temp_stack.copy()
        else:
            L1_output_data = pd.concat([L1_output_data, temp_stack], ignore_index=True)

        L1_output_data.y = L1_output_data.y.astype(int)


    
    L1_output_data.head(10).to_csv('L1_output_data.csv', index=False)

    # create a list with all column names that includes 'proba'
    proba_features = [col for col in L1_output_data.columns if 'proba' in col]

    my_bar.progress(1.0)

    ###############################################################################
    #         Step 2:       Learn Layer2                                          #
    ###############################################################################
    st.write('Learning L2 models')
    use_L2features = use_features + proba_features

    L2_modeller = learn_L2_modeller(L2_modeller, L1_output_data, use_L2features)

    ###############################################################################
    #         Step 3: learn models on all of X - what iteration to use?           #
    ###############################################################################
    st.write('Learn models on all of Train')

    my_bar2 = st.progress(0)
    ant_meta_models = 4
    step = 1/(ant_meta_models) - 0.0000001
    steps = 0.0
    my_bar2.progress(steps)

    for model in L1_modeller:
        steps += step
        my_bar2.progress(steps)
        with open(pref+'optimera/params_'+model.name+'.json', 'r') as f:
            params = json.load(f)

        params = params['params']
        tot_mod = model.learn(X, y, None, None,
                                iterations=500,
                                params=params,
                                save=save)

    my_bar2.progress(1.0)
    st.empty()

    return L1_output_data


def validate_skapa_stack_learning(X_, y, use_features):
    # För validate
    X = X_.copy()
    # print(X.shape)
    # print(len(meta_features))
    stacked_data = X[meta_features].copy()
    for model in L1_modeller:
        part = model.name[3:]
        stacked_data['proba'+part] = model.predict(X, use_features)
        meta_features += ['proba'+part]

    assert list(
        stacked_data.columns) == meta_features, f'columns in stacked_data is wrong {list(stacked_data.columns)} \n {meta_features}'
    assert len(stacked_data) == len(
        y), f'stacked_data {len(stacked_data)} and y {len(y)} should have same length'
    return stacked_data[meta_features], meta_features, y


# %%
##############################################################
#                     VALIDATE                               #
##############################################################


def predict_meta_models(meta_modeller, stack_data, meta_features, mean_type='geometric'):
    """
    TODO: Ändra till Layer2 predict
    Predicts the meta models på stack_data och beräknar meta_proba med mean_type

    Args:
        meta_modeller (_type_): Dict definierad från start
        stack_data (_type_): Dataframe från modellers predict_proba plus deras input data
        meta_features (_type_): Allt utom datum, avd och y
        mean_type (str, optional): 'arithmetic' or 'geometric'. Defaults to 'geometric'.

    Returns:
        preds: Dataframe med meta_modellers prediktioner
    """

    assert 'y' not in stack_data.columns, f'y skall inte finnas i stack_data'
    preds = pd.DataFrame(columns=list(meta_modeller.keys())+['meta'])

    # load ENC
    with open(pref+'modeller/meta_encoder.pkl', 'rb') as f:
        ENC = pickle.load(f)

    stack_data, _ = prepare_L2_input_data(stack_data, None,  meta_features)
    temp = stack_data.copy()

    # dirty trick to init preds.meta with the number of rows
    preds['meta'] = temp.iloc[:, 0]
    if mean_type == 'arithmetic':
        preds['meta'] = 0
    else:
        preds['meta'] = 1

    for key, values in meta_modeller.items():
        print(f'{key} predicts')
        # meta_model = values['model']

        fn = f'predict_{key}_model(temp)'

        preds[key] = eval(fn)[:, 1]

        if mean_type == 'arithmetic':
            preds['meta'] += preds[key]
        else:
            preds['meta'] *= preds[key]

    if mean_type == 'arithmetic':
        # aritmetisk medelvärde
        preds['meta'] /= len(meta_modeller)
    else:
        # geometriskt medelvärde
        preds['meta'] = preds['meta'] ** (1/len(meta_modeller))

    return preds


# write the scores
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


def validate(meta_modeller, fraction=None):
    # Skapa v75-instans
    v75 = td.v75(pref=pref)

    base_features = v75.get_df().columns.to_list()

    # Hämta data från v75
    _ = v75.förbered_data(missing_num=False)  # num hanteras av catboost
    df_work = v75.test_lägg_till_kolumner()

    st.info('skall endast  köras efter "Learn for Validation"')

    _, _, X_val, y_val = hold_out_val_data(df_work, fraction)
    st.info(f'Validerar på:  {X_val.datum.iloc[0]} - -{X_val.datum.iloc[-1]}')
    # print(f'Validerar på:  {X_val.datum.iloc[0]} - -{X_val.datum.iloc[-1]}')

    # # create the stack from validation data

    # Läs in NUM_FEATURES.txt och CAT_FEATURES.txt
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()

    # läs in NUM_FEATURES.txt till num_features
    with open(pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
        num_features = f.read().split()
    use_features = cat_features + num_features

    stacked_val, meta_features, y_val = validate_skapa_stack_learning(
        X_val, y_val, use_features)

    ##############################################################
    #                          Meta models                       #
    ##############################################################

    y_true = y_val.values

    y_preds = predict_meta_models(meta_modeller, stacked_val, use_features)

    ############## write y_true to file for testing ##############
    # first make y_true a dataframe
    # rf_y_true = pd.DataFrame(y_true, columns=['y'])
    # rf_y_true.to_csv(pref+'rf_y_true.csv', index=False)
    ##############################################################

    st.info('förbereder meta plot')

    plot_confusion_matrix(y_true, y_preds.meta, 'meta', fr=0.0, to=0.9)

    for model in y_preds.columns:
        if model == 'meta':
            continue

        st.write('\n')
        st.info(f'förbereder {model} plot')
        plot_confusion_matrix(y_true, y_preds[model], model, fr=0.0, to=0.9)

    # st.write('\n')
    # st.info('förbereder et plot')
    # plot_confusion_matrix(y_true, y_preds.et, 'et',
    #                       fr=0.0, to=1.0, margin=0.01)

    # st.write('\n')
    # st.info('förbereder knn plot')
    # plot_confusion_matrix(y_true, y_preds.knn, 'knn', fr=0.0, to=0.9)

    # st.write('\n')
    # st.info('förbereder ridge plot')
    # plot_confusion_matrix(y_true, y_preds.ridge, 'ridge', fr=0.0, to=0.9)

    st.write('\n')

    stacked_val['y'] = y_true
    stacked_val['avd'] = X_val.avd.values

    ################################################################
    #                         proba bas-modeller                   #
    ################################################################
    st.write('\n')
    for typ in L1_modeller:
        st.write('\n')
        name = 'proba' + typ.name[3:]
        y_pred = stacked_val[name]
        plot_confusion_matrix(y_true, y_pred, name, fr=0.0, to=0.9)

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

        st.session_state.df_ny = df_ny


# modeller = [test1,test2,test3,test4]

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
with buttons:
    if st.sidebar.button('scrape'):
        st.write(f'web scraping {st.session_state.datum}')
        try:
            scrape()
            del st.session_state.datum  # säkra att datum är samma som i scraping
        except:
            st.error(
                "Fel i web scraping. Kolla att resultat finns för datum och internet är tillgängligt")

    if st.sidebar.button('reuse scrape'):
        # del st.session_state.datum  # säkra att datum är samma som i scraping
        try:
            df_ny = pd.read_csv('sparad_scrape_learn.csv')
            st.session_state.df_ny = df_ny
            if df_ny.datum.iloc[0] != st.session_state.datum:
                st.error(
                    f'Datum i data = {df_ny.datum.iloc[0]} \n\n är inte samma som i omgång')
            else:
                st.success(f'inläst data med datum = {df_ny.datum.iloc[0]}')
        except:
            # write error message
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

            st.success('✔️ TimeSeries learning done')

        if st.sidebar.button('Validate'):
            validate(L2_modeller, fraction=st.session_state.fraction)

        if st.sidebar.button('Final learning'):
            final_learning(L1_modeller, L2_modeller)

        if st.sidebar.button('Clear'):
            st.empty()
