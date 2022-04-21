# %%
# from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from IPython.display import display
from catboost import CatBoostClassifier, Pool

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)
import streamlit as st
import sys
import pickle
from sklearn.ensemble import RandomForestRegressor

sys.path.append('C:\\Users\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import time
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

def v75_scraping():
    # st.write("TILLFÄLLIGT AVSTÄNGD. ANVÄNDER SPARAT")
    # df = pd.read_csv('sparad_scrape.csv')

    df, strukna = vs.v75_scraping(history=True, resultat=False)
    for f in ['häst','bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
    return df

    # Alternativ metod
    # Ta fram rader för varje typ enligt test-resultaten innan
    # låt meta_model välja mellan typerna tills max insats, sorterat på meta_proba

# remove NaN for cat_features in X and return (X, cat_features)
# ta bort alla features som inte används innan call


def prepare_for_catboost(X_, features=[]):
    X = X_.copy()
    Xtemp = remove_features(X, remove_mer=['avd', 'datum'])

    if len(features) > 0:
      Xtemp = Xtemp[features]
    # get numerical features and cat_features
    num_features = list(Xtemp.select_dtypes(include=[np.number]).columns)
    cat_features = list(Xtemp.select_dtypes(include=['object']).columns)

    # check cat_features isna
    print('Number of NaN in cat before:', X[cat_features].isna().sum()[
          X[cat_features].isna().sum() > 0].sort_values(ascending=False).sum())

    # impute 'missing' for all NaN in cat_features
    X[cat_features] = X[cat_features].fillna('missing')
    print('Number of NaN in cat after:', X[cat_features].isna().sum().sum())
    return X, cat_features

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
    insats = 0
    # group by avd
    summa = df.groupby('avd').avd.count().prod() / 2
    return summa

# antal hästar per avdeling


def lägg_in_antal_hästar(df_):
    df = df_.copy()
    df['ant_per_lopp'] = None
    df['ant_per_lopp'] = df.groupby(['datum', 'avd'])['avd'].transform('count')
    return df

# mest streck per avdeling


def mest_streck(X_, i, datum, avd):
    X = X_.copy()
    X.sort_values(by=['datum', 'avd', 'streck'], ascending=[
                  True, True, False], inplace=True)
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


class Typ():
    def __init__(self, name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck):
        assert (motst_diff == False and motst_ant == 0) or (motst_ant > 0)
        assert (ant_favoriter == 0 and only_clear ==
                False) or (ant_favoriter > 0)
        self.name = name                # string för filnamn mm

        # inkludera features eller ej
        self.ant_hästar = ant_hästar    # int feature med antal hästar per avdelning
        # int inkludera n features med bästa motståndare (streck)
        self.motst_ant = motst_ant
        self.motst_diff = motst_diff    # bool ovanstående med diff istf fasta värden
        self.streck = streck            # bool inkludera feature med streck

        # urval av rader
        self.proba = proba              # bool för prioritering vid urval av rader
        self.kelly = kelly              # bool för prioritering vid urval av rader
        # int för hur många favoriter (avd med en häst) som ska användas
        self.ant_favoriter = ant_favoriter
        self.only_clear = only_clear    # bool för att bara avvända klara favoriter

    def load_model(self):
        with open('modeller/'+self.name+'.model', 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model):
        with open('modeller/'+self.name+'.model', 'wb') as f:
            pickle.dump(model, f)

    def prepare_for_model(self, X_):
        # X_ måste ha datum och avd
        X = X_.copy()
        print(self.name, end=', ')
        if self.ant_hästar:
            print('Lägg in ant_hästar', end=', ')
            X = lägg_in_antal_hästar(X)
        if self.motst_diff:
            print('Lägg in diff motståndare', end=', ')
            X = lägg_in_diff_motståndare(X, self.motst_ant)
        elif self.motst_ant > 0:
            print('Lägg in motståndare', end=', ')
            X = lägg_in_motståndare(X, self.motst_ant)
        # Behåll streck ända tills learn och predict (används för prioritera rader)
        print()
        return X

    def learn(self, X_, y, features, iterations=1000, save=True, verbose=False):
        # X_ måste ha datum och avd

        cbc = CatBoostClassifier(
            iterations=iterations, loss_function='Logloss', eval_metric='AUC', verbose=verbose)

        X = self.prepare_for_model(X_)
        if not self.streck:
            X.drop('streck', axis=1, inplace=True)

        X, cat_features = prepare_for_catboost(X)

        X = remove_features(X, remove_mer=['datum', 'avd'])
        cbc.fit(X, y, cat_features, use_best_model=False)

        print('best score', cbc.best_score_)
        if save:
            self.save_model(cbc)
        return cbc

    def predict(self, X_):
        # X_ måste ha datum och avd
        X = self.prepare_for_model(X_)
        model = self.load_model()
        if not self.streck:
            # print('drop streck')
            X.drop('streck', axis=1, inplace=True)

        X, cat_features = prepare_for_catboost(X, model.feature_names_)

        # all features in model
        X = remove_features(X, remove_mer=['datum', 'avd'])
        # print(len(X.columns), len(model.feature_names_))
        # print('Diff', set(X.columns) - set(model.feature_names_))
        # print('X.columns\n',X.columns)
        # print('model features names\n',model.feature_names_)

        assert len(X.columns) == len(
            model.feature_names_), f'len(X.columns)  != len(model.feature_names_) in predict {self.name}'
        assert set(X.columns) == set(
            model.feature_names_), 'features in model and in X not equal'
        # assert list(X.columns) == list(model.feature_names_), f'features in model {self.name} and X not in same order'
        X = X[model.feature_names_]
        print('predict '+self.name)
        # print(model.get_feature_importance(prettified=True)[:3])

        return model.predict_proba(X)[:, 1]


#%%
# skapa modeller
#           name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck
typ6 = Typ('typ6', True,       True, False,     0,          False,          0,            False,    True)
typ1 = Typ('typ1', False,      True, False,     2,          True,           2,            True,     False)
typ9 = Typ('typ9', True,       True, True,      2,          True,           2,            True,     True)
typ16 = Typ('typ16', True,      True, True,      2,           True,           2,            False,    True)

typer = [typ6, typ1, typ9, typ16]  # load a file with pickl


with open('modeller\\meta.model', 'rb') as f:
    meta_model = pickle.load(f)

#%%
# för stacking ta med alla hästar per typ och proba plus kelly
def build_stack_df(X_, typer):
    X = X_.copy()
    stacked_data = X[['datum', 'avd', 'startnr', 'häst']].copy()
    for typ in typer:
        nr = typ.name[3:]
        stacked_data['proba'+nr] = typ.predict(X)
        stacked_data['kelly' +
                     nr] = kelly(stacked_data['proba'+nr], X[['streck']], None)
    return stacked_data


def meta_predict(X_):
    # X_ innehåller även datum,startnr och avd
    extra = ['datum', 'avd', 'startnr', 'häst']
    assert list(
        X_.columns[:4]) == extra, 'meta_model måste ha datum, avd och startnr, häst för att kunna välja'
    X = X_.copy()
    with open('modeller\\meta.model', 'rb') as f:
        meta_model = pickle.load(f)

    # print(meta_model.predict_proba(X.iloc[:, -8:]))
    X['meta_predict'] = meta_model.predict_proba(X.iloc[:, -8:])[:, 1]
    my_columns = extra + list(X.columns)[-9:]

    return X[my_columns]


def comp_cost(antal_rader):
    cost = (antal_rader**2)/2
    return antal_rader, cost

def välj_rad(X_):

    max_insats = 320
    veckans_rad = X_.copy()
    veckans_rad['välj'] = False

    for avd in veckans_rad.avd.unique():
        max_pred = veckans_rad[veckans_rad.avd == avd]['meta_predict'].max()
        veckans_rad.loc[(veckans_rad.avd == avd) & (
            veckans_rad.meta_predict == max_pred), 'välj'] = True
    antal_rader = 1
    veckans_rad = veckans_rad.sort_values(by=['meta_predict'], ascending=False)

    # 3. Använda ensam favorit för ett par avd? Kolla test-resultat
    # for each row in rad, välj=True if select_func(cost,avd) == True
    cost = antal_rader*0.5
    for i, row in veckans_rad.iterrows():
        new_antal, new_cost = comp_cost(antal_rader+1)
        # print(the_cost)
        if new_cost > max_insats:
            break

        antal_rader = new_antal
        cost = new_cost
        veckans_rad.loc[i, 'välj'] = True
        # print(cost)
    veckans_rad.sort_values(by=['välj', 'avd'], ascending=[
                            False, True], inplace=True)

    return veckans_rad

      
# add new columns for proba and kelly      
def stack_predictions(models, X_):      
    X=X_.copy()
    for model in models:
        X['proba'+model.name] = model.predict(X)
        X['kelly'+model.name] = kelly(X['proba'+model.name], X[['streck']], None)
    return X


        
#%% [markdown]
## Streamlit kod startar här
#%%

v75 = st.container()
scraping = st.container()
avd = st.container()
sortera = st.container()

models = [typ6, typ1, typ9, typ16]

# define st.state
if 'df' not in st.session_state:
    st.session_state['df'] = None
 
with scraping:
    def scrape():
        scraping.write('Starta web-scraping för ny data')
        with st.spinner('Ta det lugnt!'):
            st.image('winning_horse.png')  # ,use_column_width=True)
            
            #####################
            df_scraped=v75_scraping()
            df_scraped.to_csv('sparad_scrape.csv', index=False)
            #########################
            
            st.balloons()
            
            print(df_scraped.datum.unique())
            df_stack = build_stack_df(df_scraped, typer)
            df_meta = meta_predict(df_stack)
            df_meta.reset_index(drop=True, inplace=True)
            df = välj_rad(df_meta)
            st.session_state.df = df
            
        st.write('Scraping klar') 
        
    if st.button('scrape'):
        scrape()
        del st.session_state.datum  # säkra att datum är samma som i scraping
        
with v75:
    if 'datum' not in st.session_state:
        omg_df = pd.read_csv('omg_att_spela_link.csv' )
        urlen=omg_df.Link.values[0]
        datum = urlen.split('spel/')[1][0:10]
        st.session_state.datum = datum
        
    st.title('Veckans v75 -  ' +st.session_state.datum)

    
with avd:
    if st.session_state.df is not None:
        use = avd.radio('Välj avdelning', ('Avd 1 och 2','Avd 3 och 4','Avd 5 och 6','Avd 7','exit'))
        avd.subheader(use)
        col1, col2 = st.columns(2)
        # print(df.iloc[0].häst)
        dfi=st.session_state.df
        dfi.rename(columns={'startnr': 'nr', 'meta_predict': 'Meta'}, inplace=True)
        dfi['kelly'] = (dfi[['kelly1','kelly6','kelly9','kelly16']]).max(axis=1)
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
        elif use=='exit':
            st.stop()    
        else:
            st.write('ej klart')

with sortera:   
    if st.sidebar.checkbox('se data'):
        dfr = st.session_state.df
        sort=st.sidebar.radio('sortera på',['poäng','avd','insats'])
        if sort:
            if sort=='insats':
                st.write(dfr[['avd','häst','poäng','proba','insats','prob_order']].sort_values(by=['insats','proba'],ascending=[False,False]))
            else:
                st.write(dfr[['avd','häst','poäng','proba','insats','prob_order']].sort_values(by=[sort,'proba'],ascending=[True,False]))  
