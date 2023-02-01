#%%
import logging
import numpy as np
import pandas as pd
from IPython.display import display
import V75_scraping as vs
import travdata as td
import typ as tp
import json
import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')

#%%

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

pref = ''   # '../'

###################################################################################

def skapa_modeller():  

    logging.info('skapa_modeller: Initierar dict med modeller')

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
        model = tp.Typ(L1_key, value['#hästar'], value['#motst'], value['motst_diff'], value['streck'])
        L1_modeller[L1_key] = model

        L2_key = key + 'L2'
        model = tp.Typ(L2_key, value['#hästar'], value['#motst'], value['motst_diff'], value['streck'])
        L2_modeller[L2_key] = model

    print('keys and names i modeller')
    # print keys in dict modeller
    for key, value in L1_modeller.items():
        assert key == value.name, "key and value.name should be the same in modeller"
        logging.info(f'skapa_modeller: {key} klar')

    print('keys and names i meta_modeller')
    for key, value in L2_modeller.items():
        assert key == value.name, "key and value.name should be the same in meta_modeller"
        logging.info(f'skapa_modeller: {key} klar')
    
    return L1_modeller, L2_modeller    


def read_in_features():
    # läs in NUM_FEATURES.txt till num_features
    with open(pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
        num_features = f.read().split()

    # läs in CAT_FEATURES.txt till cat_features
    with open(pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
        cat_features = f.read().split()

    use_features = cat_features + num_features
    return use_features, cat_features, num_features

def create_L2_input(df_,L1_modeller, L1_features,with_y=True):
    """ Använder L1_modeller för att skapa input till L2_modeller
    Args:
        df_ (DataFrame): All tvättad input från scraping men endast L1_features används
        L1_modeller (Dict): av typen {'model_namn': model (instans av Typ)}
        L1_features (List): De features som används för att träna L1_modeller
        with_y (bool, optional): y är med om vi har en Learning-situation annars inte. Defaults to True.

    Returns:
        df (DataFrame): df kompletterad med proba-data från L1_modeller
        List: L1_features + proba-data från L1_modeller, dvs L2_features
    """
    logging.info('create_L2_input: Startar create_L2_input')
    
    if with_y:
        #  assert that 'y' is in Xy_
        assert 'y' in df_.columns, f'y skall finnas i Xy_ '
    
    df = df_.copy()

    df = df.reset_index(drop=True)
    proba_data = pd.DataFrame()
    for model_name, typ in L1_modeller.items():
        logging.info(f'create_L2_input: predict med {model_name}')
        proba_data['proba_'+model_name] = typ.predict(df, L1_features)
    
    proba_data = proba_data.reset_index(drop=True)

    ####### kolla om det finns NaNs i Xy eller proba_data
    Xy_na = df.isna()
    Xy_missing = df[Xy_na.any(axis=1)]
    proba_data_na = proba_data.isna()
    proba_data_missing = proba_data[proba_data_na.any(axis=1)]

    if Xy_missing.shape[0] > 0:
        logging.warning(f'create_L2_input: rader med NaNs i Xy {Xy_missing.shape[0]}')
        print('rader med NaNs i Xy', Xy_missing.shape[0])

    if proba_data_missing.shape[0] > 0:
        print(f'rader med NaNs i proba_data_missing {proba_data_missing.shape[0]}')
        logging.warning(f'create_L2_input: rader med NaNs i proba_data_missing {proba_data_missing.shape[0]}')
    ####### slutkollat

    assert df.shape[0] == proba_data.shape[0], f'Xy.shape[0] != proba_data.shape[0] {df.shape[0]} != {proba_data.shape[0]}'

    assert len(proba_data) == len(df), f'proba_data {len(proba_data)} is not the same length as Xy {len(df)} innan concat'
    assert 'bana' in df.columns, f'bana not in Xy.columns {df.columns} innan concat'
    
    logging.info('create_L2_input: concat Xy and proba_data')
    df = pd.concat([df, proba_data], axis=1, ignore_index=False) # eftersom index är kolumn-namn (axis=1)
    assert len(proba_data) == len(df), f'proba_data {len(proba_data)} is not the same length as Xy {len(df)} efter concat'
    assert 'bana' in df.columns, f'bana not in Xy.columns {df.columns} efter concat'

    assert df.shape[0] == proba_data.shape[0], f'Xy.shape[0] != proba_data.shape[0] {df.shape[0]} != {proba_data.shape[0]}'
    # df.y = df.y.astype(int)
    
    proba_columns = df.filter(like='proba').columns
    assert proba_columns.size == 4, f"4 proba_ columns should be in stack_data. We have {proba_columns}"
    assert proba_data.columns.size == 4, f"4 items should be in proba_data.columns. We have {proba_data.columns}"
    
    logging.info(f'create_L2_input: Är klar med {proba_columns.size} proba_ columns')
    return df, L1_features+proba_data.columns.tolist()

#%%

def learn_L2_modeller(L2_modeller, L2_input_data, use_L2features, save=True):
    logging.info('Starting "learn_L2_modeller"')
    
    assert 'streck' in use_L2features, f'streck is missing in use_L2features direkt i början'

    assert 'y' in L2_input_data.columns, 'y is missing in L2_input_data'
    y_meta = L2_input_data.pop('y').astype(int)

    assert len([item for item in L2_input_data.columns if 'proba_' in item]) == 4, "4 proba_ should be in stack_data"

    X_meta = L2_input_data.copy(deep=True)
    assert 'datum' in X_meta.columns, f'datum is missing in X_meta efter prepare_L2_input_data'
    assert 'streck' in X_meta.columns, f'streck is missing in X_meta efter prepare_L2_input_data'
    assert 'streck' in use_L2features, f'streck is missing in use_L2features efter prepare_L2_input_data'

    for enum, (model_name, model) in enumerate(L2_modeller.items()):
        display(f'#### learn {model_name} Layer2 på L2_input_data (stack-data)')
        logging.info(f'Learn_L2: {model_name} Layer2 på L2_input_data (stack-data)')
        with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
            params = json.load(f)
            params = params['params']

        assert 'streck' in use_L2features, f'{enum} streck is missing in use_L2features innan learn för {model_name}'
        my_meta = model.learn(X_meta, y_meta, use_L2_features_=use_L2features, params=params, save=save)

        L2_modeller[model_name] = my_meta

        if save:
            # Save the list of column names to a text file
            with open(pref+'modeller/'+model_name+'_columns.txt', "w", encoding="utf-8") as f:
                for col in X_meta[use_L2features].columns.tolist():
                    f.write(col + '\n')

    return L2_modeller


def predict_med_L2_modeller(L2_modeller, L2_input, use_features, weights=[0.25, 0.25, 0.25, 0.25],mean_type='geometric', with_y = True):
    """
    Predicts med L2_modeller på stack_data och beräknar meta_proba med mean_type

    Args:
        L2_modeller (Dict): Definierad från start
        L2_input (DataFrame): Skapad av L1-modeller. Är input-data till L1 plus deras resp predict_proba  
        use_features (List): Lista med kolumner som används i L2-modeller
        weights (List, optional): Vikter för L2-modeller. [cat1L2, cat2L2, cat3L2, cat4L2].
        mean_type (str, optional): 'arithmetic' or 'geometric'. Defaults to 'geometric'.

    Returns:
        temp: DataFrame L2_input kompletterat med L2_modellers prediktioner
    """
    df = L2_input.copy(deep=True)
    
    # Check for the presence of 4 'proba-' columns in L2_input
    proba_columns = df.filter(like='proba').columns
    assert proba_columns.size == 4, f"4 proba_ columns should be in stack_data. We have {proba_columns}"

    assert len([col for col in use_features if 'proba_' in col]) == 4, f'use_features saknar proba-kolumner till L2-modeller\n{use_features}'
    if with_y:
        assert 'y' in df.columns, f'y skall finnas i stack_data'

    proba_names = list(L2_modeller.keys())
    proba_names = ['proba_'+name for name in proba_names]
    proba_names.append('meta')

    temp = df.copy()
    # extend temp.columns with column_names with None values
    temp = temp.reindex(columns=temp.columns.tolist() +
                        proba_names, fill_value=np.nan)

    for model_name, model in L2_modeller.items():
        print(f'{model_name} predicts for validate')

        missing_items = set(use_features) - set(temp.columns)
        assert not missing_items, f'{missing_items} in use_features not in temp.columns {temp.columns}'

        temp['proba_'+model_name] = model.predict(temp, use_features)

    proba_cols = ['proba_cat1L2', 'proba_cat2L2', 'proba_xgb1L2', 'proba_xgb2L2']
    # multiply each column with its corresponding weight
    # weights = pd.Series(weights * len(temp), index=temp.index)
    # weights_matrix = np.repeat(weights, len(temp), axis=0).reshape(-1, len(weights))

    # weighted_cols = temp[proba_cols].mul(weights, axis=0)
    # weighted_cols = temp[proba_cols].values * weights_matrix
    for col in proba_cols:
        temp[col] = temp[col] * weights[proba_cols.index(col)]
    
    if mean_type == 'arithmetic':
        # aritmetisk medelvärde
        temp['meta'] = temp[proba_cols].mean(axis=1)

        # temp['meta'] = temp.filter(like='proba_').mean(axis=1)
    else:
        # geometriskt medelvärde
        temp['meta'] = (temp[proba_cols].prod(axis=1)) ** (1/len(weights))
        
        # temp['meta'] = temp.filter(like='proba_').prod(
        #     axis=1) ** (1/len(L2_modeller))

    return temp

def lägg_till_extra_kolumner(df_):
    """
    beräknar och  till nya kolumner
    OBS: Denna funktion kanske också skall anropas av travdata-funktionen med samma namn
    """
    df = df_.copy()
    ##### kr/total_kr_avd ******
    sum_kr = df.groupby(['datum', 'avd']).kr.transform(lambda x: x.sum())
    df['rel_kr'] = df.kr/sum_kr
    df.drop(['kr'], axis=1, inplace=True)
    
    ##### avst till ettan (streck) ******
    df['max_streck'] = df.groupby(['datum', 'avd']).streck.transform(lambda x: x.max())
    df['streck_avst'] = df.max_streck - df.streck
    df.drop(['max_streck'], axis=1, inplace=True)
    
    ##### ranking per avd / ant_startande ******
    rank_per_avd = df.groupby(['datum', 'avd'])['streck'].rank(
        ascending=False, method='dense')
    count_per_avd = df.groupby(['datum', 'avd']).streck.transform(lambda x: x.count())
    df['rel_rank'] = rank_per_avd/count_per_avd
    
    ##### hx samma bana (h1-h3)
    df['h1_samma_bana'] = df.bana == df.h1_bana
    df['h2_samma_bana'] = df.bana == df.h2_bana
    df['h3_samma_bana'] = df.bana == df.h3_bana

    ##### hx samma kusk (h1-h3)
    df['h1_samma_kusk'] = df.kusk == df.h1_kusk
    df['h2_samma_kusk'] = df.kusk == df.h2_kusk
    df['h3_samma_kusk'] = df.kusk == df.h3_kusk

        
    return df

def fix_history_bana(df_):
    """ ta bort suffix-nummer från travbana i history (i.e Åby-1 -> Åby, etc)
    OBS: Denna funktion kanske också skall anropas av travdata-funktionen med samma namn
    """
    
    df = df_.copy()
    df.loc[:, 'h1_bana'] = df.h1_bana.str.split('-').str[0]
    df.loc[:, 'h2_bana'] = df.h2_bana.str.split('-').str[0]
    df.loc[:, 'h3_bana'] = df.h3_bana.str.split('-').str[0]
    df.loc[:, 'h4_bana'] = df.h4_bana.str.split('-').str[0]
    df.loc[:, 'h5_bana'] = df.h5_bana.str.split('-').str[0]
    return df


def beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning):
    
    min_utdelning = df_utdelning.loc[df_utdelning.datum == datum, [
        '7rätt', '6rätt', '5rätt']]

    tot_utdelning = (min_utdelning['7rätt'] * sjuor + min_utdelning['6rätt']
                     * sexor + min_utdelning['5rätt'] * femmor).values[0]

    print('utdelning', tot_utdelning)

    return tot_utdelning

def rätta_rad(veckans_rad, datum, df_utdelning=None) :
    """Räkna ut antal 5:or, 6:or resp. 7:or
    Beräkna ev utdelning
    
    Args:
        veckans_rad: omgångens df ev inkluderat plac
        datum: datum för veckans rad i string-format (datum = datum.strftime('%Y-%m-%d'))
        df_utdelning: df med utdelning för alla datum. om None så hämtar vi filen här
    
    
    Returns:  (int) sjuor, sexor, femmor, utdelning
    """
    df=veckans_rad.copy()
    df['y'] = (df.plac==1).astype(int)
    
    if df_utdelning is None:
        df_utdelning = pd.read_csv(pref+'utdelning.csv')   
    
    sjuor, sexor, femmor, utdelning = 0, 0, 0, 0

    min_tabell = df[['y', 'avd', 'häst', 'rel_rank', 'välj']].copy()
    min_tabell.sort_values(by=['avd', 'y'], ascending=False, inplace=True)

    print('Antal rätt', min_tabell.query('välj==True and y==1').y.sum())

    # 1. om jag har max 7 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 7:
        sjuor = 1
        sexor = (min_tabell.groupby('avd').välj.sum()).sum()-7
        # antal femmor
        ant1 = min_tabell.query('avd==1 and välj==True').välj.sum()-1
        ant2 = min_tabell.query('avd==2 and välj==True').välj.sum()-1
        ant3 = min_tabell.query('avd==3 and välj==True').välj.sum()-1
        ant4 = min_tabell.query('avd==4 and välj==True').välj.sum()-1
        ant5 = min_tabell.query('avd==5 and välj==True').välj.sum()-1
        ant6 = min_tabell.query('avd==6 and välj==True').välj.sum()-1
        ant7 = min_tabell.query('avd==7 and välj==True').välj.sum()-1
        femmor = ant1*ant2+ant1*ant2+ant1*ant3+ant1*ant4+ant1*ant5+ant1*ant6+ant1*ant7 +\
            ant2*ant3+ant2*ant4+ant2*ant5+ant2*ant6+ant2*ant7 + \
            ant3*ant4+ant3*ant5+ant3*ant6+ant3*ant7 + \
            ant4*ant5+ant4*ant6+ant4*ant7 + \
            ant5*ant6+ant5*ant7 + \
            ant6*ant7

    # 2. om jag har max 6 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 6:
        avd_fel = min_tabell.loc[((min_tabell.välj == False) & (
            min_tabell.y == 1)), 'avd'].values[0]
        # print(min_tabell.query('avd== @avd_fel').välj.sum())
        sexor = min_tabell.query('avd==@avd_fel').välj.sum()
        # antal femmor
        femmor_fel, femmor_rätt = 0, 0
        for avd in range(1, 8):
            if avd == avd_fel:
                femmor_fel += min_tabell.loc[min_tabell.avd ==
                                             avd_fel].välj.sum()

            femmor_rätt += min_tabell.query(
                'avd==@avd and välj==True').välj.sum()-1
        # print(f'femmor_rätt = {femmor_rätt} femmor_fel = {femmor_fel}')
        femmor = femmor_fel * femmor_rätt

    # 3. om jag har max 5 rätt
    if min_tabell.query('välj==True and y==1').y.sum() == 5:
        avd_fel = min_tabell.loc[((min_tabell.välj == False) & (
            min_tabell.y == 1)), 'avd'].values
        femmor = min_tabell.loc[min_tabell.avd == avd_fel[0]].välj.sum(
        ) * min_tabell.loc[min_tabell.avd == avd_fel[1]].välj.sum()

    utdelning = beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning)
   
    return sjuor, sexor, femmor, utdelning
        
        
        
#%%
##########################
####### UNIT TESTS #######
##########################

def test_skapa_modeller(): 
    logging.basicConfig(level=logging.DEBUG, filemode='a', filename='v75.log', force=True,
                    encoding='utf-8', format='DBG:' '%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Startar test_skapa_modeller')
    L1_modeller, L2_modeller = skapa_modeller()
    assert len(L1_modeller) == 4, "L1_modeller should have 4 elements"
    assert len(L2_modeller) == 4, "L2_modeller should have 4 elements"
    for key, value in L1_modeller.items():
        assert key == value.name, "key and value.name should be the same in modeller"
    for key, value in L2_modeller.items():
        assert key == value.name, "key and value.name should be the same in meta_modeller"
test_skapa_modeller()

