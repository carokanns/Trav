#%%
import logging
import numpy as np
import pandas as pd
from IPython.display import display
import V75_scraping as vs
import travdata as td
import json
import sys
sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')

#%%

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

pref = ''   # '../'

###################################################################################


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

def skapa_modeller():  
    """Skapar dict med modeller och returnerar dict med modeller"""
    import typ as tp
    log_print('skapa_modeller: Initierar dict med modeller','i')

    # skapar dict med modeller
    modell_dict = { 'cat1': {'#hästar': False, '#motst': 3, 'motst_diff': True, 'streck': False},
                    'cat2': {'#hästar': False,  '#motst': 3, 'motst_diff': True, 'streck': True},
                    'xgb1': {'#hästar': False, '#motst': 3, 'motst_diff': True, 'streck': False},
                    'xgb2': {'#hästar': False,  '#motst': 3, 'motst_diff': True, 'streck': True}
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
    for key, value in L1_modeller.items():
        assert key == value.name, "key and value.name should be the same in modeller"
        log_print(f'skapa_modellen: {key} klar')

    log_print('keys and names i meta_modeller')
    for key, value in L2_modeller.items():
        assert key == value.name, "key and value.name should be the same in meta_modeller"
        logging.info(f'skapa_modellen: {key} klar')
    
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
        L1_features (List): De features som används för predict med L1_modeller
        with_y (bool, optional): y är med om vi har en Learning-situation annars inte. Defaults to True.

    Returns:
        df (DataFrame): df kompletterad med proba-data från L1_modeller
        List: L1_features + proba-data från L1_modeller, dvs L2_features
    """
    log_print(f'create_L2_input: Startar create_L2_input')
    
    # TODO: skriv om denna funktion så att vi använder df med use_features och inte proba_data
    if with_y:
        #  assert that 'y' is in Xy_
        assert 'y' in df_.columns, log_print(f'y skall finnas i Xy_ ', 'e')
    
    df = df_.copy()

    df = df.reset_index(drop=True)
    proba_data = pd.DataFrame()
    for model_name, typ in L1_modeller.items():
        log_print(f'create_L2_input: predict med {model_name}')
        proba_data['proba_'+model_name] = typ.predict(df, L1_features)
    
    proba_data = proba_data.reset_index(drop=True)

    ####### kolla om det finns NaNs i Xy eller proba_data
    Xy_na = df.isna()
    Xy_missing = df[Xy_na.any(axis=1)]
    proba_data_na = proba_data.isna()
    proba_data_missing = proba_data[proba_data_na.any(axis=1)]

    if Xy_missing.shape[0] > 0:
        log_print(f'create_L2_input: rader med NaNs i Xy {Xy_missing.shape[0]}', 'w')

    if proba_data_missing.shape[0] > 0:
        log_print(f'create_L2_input: rader med NaNs i proba_data_missing {proba_data_missing.shape[0]}')
    ####### slutkollat

    assert df.shape[0] == proba_data.shape[0], log_print(
        f'Xy.shape[0] != proba_data.shape[0] {df.shape[0]} != {proba_data.shape[0]}','e')

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

def learn_L2_modeller(L2_modeller, L2_input_data, L2_features, save=True):
    log_print('Starting "learn_L2_modeller"')
    
    assert 'streck' in L2_features, f'streck is missing in use_L2features direkt i början'

    assert 'y' in L2_input_data.columns, 'y is missing in L2_input_data'
    y_meta = L2_input_data.pop('y').astype(int)

    # assert len([item for item in L2_input_data.columns if 'proba_' in item]) == 4, "4 proba_ should be in stack_data"
    assert len(L2_input_data.filter(like='proba').columns) == 4, "4 proba_ should be in stack_data"

    X_meta = L2_input_data.copy(deep=True)
    assert 'datum' in X_meta.columns, f'datum is missing in X_meta efter prepare_L2_input_data'
    assert 'streck' in X_meta.columns, f'streck is missing in X_meta efter prepare_L2_input_data'
    assert 'streck' in L2_features, f'streck is missing in use_L2features efter prepare_L2_input_data'

    for enum, (model_name, model) in enumerate(L2_modeller.items()):
        log_print(f'Learn_L2: {model_name} Layer2 på L2_input_data (stack-data)')
        with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
            params = json.load(f)
            params = params['params']

        assert 'streck' in L2_features, log_print(f'{enum} streck is missing in use_L2features innan learn för {model_name}','d')
        my_meta = model.learn(X_meta, y_meta, use_L2_features_=L2_features, params=params, save=save)

        # L2_modeller[model_name] = my_meta

        if save:
            # Save the list of column names to a text file
            with open(pref+'modeller/'+model_name+'_columns.txt', "w", encoding="utf-8") as f:
                for col in X_meta[L2_features].columns.tolist():
                    f.write(col + '\n')

    return L2_modeller

def learn_L1_modeller(L1_modeller, L1_input_df, L1_features, L1_test_df=None, save=True):
    """ Lär upp L1_modeller på L1_input_df

    Args:
        L1_modeller (List): En lista av L1_modeller (Typ-instanser)
        L1_input_df (DataFrame): Den data som vi ska lära modellerna på
        L1_features (List): Det urval av features som vi ska använda
        L1_test_df (DataFrame, opt.): En test-data som vi kan använda för early stopping. Defaults to None.
        save (bool, optional): Spara modellerna eller inte. Defaults to True.
    """
    
    if 'y' in L1_input_df.columns:
        y_train = L1_input_df['y'].astype(int)
    elif 'plac' in L1_input_df.columns:
        y_train = (L1_input_df['plac']==1).astype(int)
    else:
        y_train = None   
            
    for model_name, model in L1_modeller.items():
        with open(pref+'optimera/params_'+model_name+'.json', 'r') as f:
                params = json.load(f)
                params = params['params']

        logging.info(f'# learn {model_name} Layer1 på X_train-delen')

        my_model = model.learn(L1_input_df, y_train, params=params, save=save)
    return L1_modeller

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
    assert 'datum' in temp.columns, f'datum saknas i temp.columns {temp.columns}'
    # extend temp.columns with column_names with None values
    temp = temp.reindex(columns=temp.columns.tolist() +
                        proba_names, fill_value=np.nan)
    assert 'datum' in temp.columns, f'datum saknas i temp.columns {temp.columns}'
    # temp.to_csv(pref+'temp1.csv', index=False)
    for model_name, model in L2_modeller.items():
        log_print(f'{model_name} predicts for validate','i')

        missing_items = set(use_features) - set(temp.columns)
        assert not missing_items, f'{missing_items} in use_features not in temp.columns {temp.columns}'
        
        # temp_describe = temp.describe(include='all').T
        # log_print(f'temp.describe() = {temp_describe}','i')
        # temp_describe = temp[use_features].describe(include='all').T
        # log_print(f'temp[use_features].describe() = {temp_describe}','i')
        
        
        # # skriv ut use_features som textfil
        # with open(pref+'temp_'+model_name+'_columns.txt', "w", encoding="utf-8") as f:
        #     for col in use_features:
        #         f.write(col + '\n') 
                
        # temp.to_csv(pref+'temp2.csv', index=False)
        log_print(f'predict_med_L2_modeller: {model_name} {type(model)}','i')
        temp['proba_' + model_name] = model.predict(temp, use_features_=use_features)
        log_print(f'model.predict klar','i')
        
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

    # behåll endast first och second med värde. De andra är NaN
    df = df.dropna(subset=['first', 'second'])
    df['diff'] = df['first'] - df['second']

    # drop duplicates per avd
    df = df.drop_duplicates(subset='avd', keep='first')

    df.sort_values(by='diff', ascending=False, inplace=True)
    # st.write(f'kolumnerna i df = {df.columns}')
    return df

def compute_total_insats(df):
    summa = df.groupby('avd').avd.count().prod() / 2
    return summa

def välj_rad(df, max_insats=360):
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
        veckans_rad.loc[(veckans_rad.avd == avd) & (
            veckans_rad.meta == max_pred), 'välj'] = True

    veckans_rad = veckans_rad.sort_values(by=['meta'], ascending=False)
    veckans_rad = veckans_rad.reset_index(drop=True)

    mest_diff = mesta_diff_per_avd(veckans_rad)
    # TODO: Kolla att mest_diff stämmer med veckans rad
    assert len(mest_diff) == 7, \
        log_print(
            f'len(mest_diff) {len(mest_diff)} != 7 (antal lopp) {mest_diff[["avd","diff"]]}', 'e')

    assert set(mest_diff['avd'].unique()).issubset(veckans_rad['avd'].unique()), \
        log_print(f"Alla avd i mest_diff måste finnas i veckans_rad")
    assert len(mest_diff['avd'].unique()) == len(veckans_rad['avd'].unique()), \
        log_print(
            f"Antalet unique avd i mest_diff och i veckans_rad skall vara samma")

    cost = 0.5  # 1 rad

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
    veckans_rad.sort_values(by=['välj', 'avd'], ascending=[
                            False, True], inplace=True)

    return veckans_rad, cost

def beräkna_utdelning(datum, sjuor, sexor, femmor, df_utdelning):
    # kolla om datum finns i df_utdelning
    log_print(f'finns datum {datum}, typ={type(datum)}, {(df_utdelning.datum==datum).sum()}', 'i')
    log_print(df_utdelning.query('datum == "2017-09-03"'), 'i')
    assert len(df_utdelning.query('datum == @datum')) > 0, log_print(f'datum {datum} finns inte i df_utdelning', 'e')
    
    min_utdelning = df_utdelning.loc[df_utdelning.datum == datum, [
        '7rätt', '6rätt', '5rätt']]

    tot_utdelning = (min_utdelning['7rätt'] * sjuor + min_utdelning['6rätt']
                     * sexor + min_utdelning['5rätt'] * femmor).values[0]

    log_print(f'utdelning: {tot_utdelning}', 'i')

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
    assert 'y' in df.columns, log_print(f'Kolumnen y saknas i df')
    
    if df_utdelning is None:
        df_utdelning = pd.read_csv(pref+'utdelning.csv')   
    
    sjuor, sexor, femmor, utdelning = 0, 0, 0, 0

    min_tabell = df[['y', 'avd', 'häst', 'rel_rank', 'välj']].copy()
    min_tabell.sort_values(by=['avd', 'y'], ascending=False, inplace=True)

    log_print(f'rätta_rad: Antal rätt={min_tabell.query("välj==True and y==1").y.sum()}', 'i')

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
# test_skapa_modeller()

