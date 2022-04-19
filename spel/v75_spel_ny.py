# %%
# from sklearn.model_selection import TimeSeriesSplit
from IPython.display import display
from catboost import CatBoostClassifier
import pandas as pd 
pd.set_option('display.width', 200)

import numpy as np 
import streamlit as st
import sys
import pickle
from sklearn.ensemble import RandomForestRegressor

sys.path.append('C:\\Users\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import time
import logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(message)s',level=logging.INFO)

# %%
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

# mest streck per avd som fetures (n bästa)
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

# diff streck per avd som fetures (n största diffarna)
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

def compute_total_insats(df):
    insats = 0
    # group by avd
    summa = df.groupby('avd').avd.count().prod() / 2
    return summa

# för en omgång (ett datum) ta ut största diff för streck per avd men bara om diff >= 25 om only_clear
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
#%%

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
    df, strukna = vs.v75_scraping(history=True,resultat=False)
    for f in ['häst','bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
        df[f] = df[f].str.lower()
    return df, strukna  
  
    # df_scraped = pd.read_csv('sparad_scrape.csv')
    # for f in ['häst','bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
    #      df_scraped[f] = df_scraped[f].str.lower()
    # return df_scraped

# remove NaN for cat_features in X and return (X, cat_features)
# ta bort alla features som inte används innan call
def prepare_for_catboost(X_):
    X = X_.copy()
    X = remove_features(X, remove_mer=['avd', 'datum'])
    # get numerical features and cat_features
    num_features = list(X.select_dtypes(include=[np.number]).columns)
    cat_features = list(X.select_dtypes(include=['object']).columns)

    # check cat_features isna
    print('NaN in cat before:', X[cat_features].isna().sum()[
          X[cat_features].isna().sum() > 0].sort_values(ascending=False).sum())

    # impute 'missing' for all NaN in cat_features
    X[cat_features] = X[cat_features].fillna('missing')
    print('NaN in cat after:', X[cat_features].isna().sum().sum())
    return X, cat_features


def ta_fram_rad(df_, models):
    df = df_.copy()

    # en rad för varje modell
    rader = pd.DataFrame()
    for typ in models:  # modeller
        rad, insats = typ.spela(df)
        rad.sort_values(by=['avd', 'häst'], inplace=True)
        rader[['proba'+typ.name, 'kelly'+typ.name]
              ] = rad[['proba', 'kelly']].copy()
        print(typ.name, insats)

    ##### Stacking Predict #####
    # lägg in alla predict och kelly för alla modeller
    for model in models:
        nr = model.name[3:]
        df['proba'+nr] = model.predict(df)
        df['kelly'+nr] = kelly(df['proba'+nr], df[['streck']], None)

    rf = RandomForestRegressor()  # The meta model
    return(meta_model.predict_proba(df.iloc[:, -8:]))

#%%
class Typ():
    def __init__(self, name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck):
        assert (motst_diff == False and motst_ant == 0) or (motst_ant > 0)
        assert (ant_favoriter == 0 and only_clear == False) or (ant_favoriter > 0)
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
        with open('modeller\\'+self.name+'.model', 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model):
        with open('modeller\\'+self.name+'.model', 'wb') as f:
            pickle.dump(model, f)

    def prepare_for_model(self, X_):
        X = X_.copy()
        print(self.name)
        if self.ant_hästar:
            print('Lägg in ant_hästar')
            X = lägg_in_antal_hästar(X)
        if self.motst_diff:
            print('Lägg in diff motståndare')
            X = lägg_in_diff_motståndare(X, self.motst_ant)
        elif self.motst_ant > 0:
            print('Lägg in motståndare')
            X = lägg_in_motståndare(X, self.motst_ant)
        # Behåll streck ända tills learn och predict (används för prioritera rader)
        return X

    def learn(self, X_, y, iterations=1000, save=True, verbose=False):
        cbc = CatBoostClassifier(
            iterations=iterations, loss_function='Logloss', eval_metric='AUC', verbose=verbose)

        X = self.prepare_for_model(X_)
        if not self.streck:
            X.drop('streck', axis=1, inplace=True)

        X, cat_features = prepare_for_catboost(X)

        # X = remove_features(X, remove_mer=['avd', 'datum'])
        # cat_features = X.select_dtypes(include=['object']).columns.tolist()
        cbc.fit(X, y, cat_features, use_best_model=False)
    
        print('best score', cbc.best_score_)
        if save:
            self.save_model(cbc)

    def predict(self, X_):
        X = self.prepare_for_model(X_)
        X, cat_features = prepare_for_catboost(X)
        model = self.load_model()
        
        display(model.feature_names_)
        
        if not self.streck:
            print('drop streck')
            X.drop('streck', axis=1, inplace=True)
            
        # all features in model
        felmed = f'len(X.columns) {len(X.columns)} != len(model.feature_names_) {len(model.feature_names_)}'
        print(set(X.columns) - set(model.feature_names_))
        assert len(X.columns) == len(model.feature_names_), felmed
        assert set(X.columns) == set(model.feature_names_), 'X.columns != model.feature_names_'
        X = X[model.feature_names_]
        print('predict '+self.name)   
        print(model.get_feature_importance(prettified=True))
        return model.predict_proba(X)[:, 1]
    
    def spela(self, X_, max_insats=300, margin=1.2):
        print(f'Max insats={max_insats} Margin={margin}')

        X = stack_predictions( models, X_)  # add columns with all predictions to X
        
        # X['proba'] = self.predict(X)
        # X['kelly'] = kelly(X.proba, X[['streck']], None)

        dfSpel = pd.DataFrame()
        if self.proba:
            X = X.sort_values(by='proba', ascending=False)
            if self.kelly:
                X2 = X.sort_values(by='kelly', ascending=False)
        else:
            X = X.sort_values(by='kelly', ascending=False)  # must be kelly

        # se till att vi har minst en häst för alla avd. Välj den bästa per avd
        for avd in range(1, 8):
            dfSpel = dfSpel.append(X[X.avd == avd].iloc[0])

        favorit_list = lista_med_favoriter(
            X, self.ant_favoriter, self.only_clear)
        curr_insats = 0
        for cnt_rows, (_, row) in enumerate(X.iterrows()):
            if check_avd(row.avd, favorit_list):  # avd med en favorit - inga fler hästar
                continue

            dfSpel = dfSpel.append(row)

            curr_insats = compute_total_insats(dfSpel)
            if curr_insats > max_insats*margin:   # överstiger x% av max insats?
                dfSpel = dfSpel.iloc[:-1, :]    # ta bort sista hästen
                curr_insats = compute_total_insats(dfSpel)
                break

            if self.kelly & self.proba:
                row2 = X2.iloc[cnt_rows]
                dfSpel = dfSpel.append(row2)  # Addera en häst med bästa kelly
                # remove duplicates in dfSpel and keep the first

            dfSpel = dfSpel.drop_duplicates(
                subset=['avd', 'häst'], keep='first')
            curr_insats = compute_total_insats(dfSpel)  # kolla igen
            if curr_insats > max_insats*margin:   # överstiger x% av max insats?
                dfSpel = dfSpel.iloc[:-1, :]    # ta bort sista hästen
                curr_insats = compute_total_insats(dfSpel)
                break

        return dfSpel, curr_insats



# %%
# skapa modeller
#           name, ant_hästar, proba, kelly, motst_ant, motst_diff,  ant_favoriter, only_clear, streck
typ6 = Typ('typ6', True,       True, False,     0,      False,          0,            False,    True)
typ1 = Typ('typ1', False,      True, False,     2,      True,           2,            True,     False)
typ9 = Typ('typ9', True,       True, True,      2,      True,           2,            True,     True)
typ16 = Typ('typ16',True,      True, True,      2,      True,           2,            False,    True)
# load a file with pickl
with open('modeller\\meta.model', 'rb') as f:
    meta_model = pickle.load(f)
      
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
            df_scraped, strukna=v75_scraping()
            df_scraped.to_csv('sparad_scrape.csv', index=False)
            #########################
            
            st.balloons()
            df = ta_fram_rad(df_scraped, models)
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
        st.write(dfi)
        if use == 'Avd 1 och 2':
            # st.write(dfi.proba[0])
            col1.write(dfi[dfi.avd==1].sort_values(by=['proba'])[['nr', 'häst', 'proba', 'kelly']])
            col2.write(dfi[dfi.avd==2].sort_values(by=['proba'])[['nr', 'häst', 'proba','kelly']])
        elif use=='Avd 3 och 4':
            col1.write(dfi[dfi.avd==3].sort_values(by=['proba'])[['nr', 'häst', 'proba', 'kelly']])
            col2.write(dfi[dfi.avd==4].sort_values(by=['proba'])[['nr', 'häst', 'proba', 'kelly']])
        elif use=='Avd 5 och 6':
            col1.write(dfi[dfi.avd==5].sort_values(by=['proba'])[['nr', 'häst', 'proba', 'kelly']])
            col2.write(dfi[dfi.avd==6].sort_values(by=['proba'])[['nr', 'häst', 'proba', 'kelly']])
        elif use=='Avd 7':
            col1.write(dfi[dfi.avd==7].sort_values(by=['proba'])[['nr', 'häst', 'proba', 'kelly']])
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
