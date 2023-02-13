#%%
import matplotlib.pyplot as plt
import datetime as dt
import sys
sys.path.append('C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')
import typ as tp
from IPython.display import display
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier, Pool

import travdata as td
import concurrent.futures
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
pref = ''

import skapa_modeller as mod

plt.style.use('fivethirtyeight')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

import logging
logging.basicConfig(level=logging.INFO, filemode='w' , filename='v75.log', force=True, encoding='utf-8', format='Backtest: %(asctime)s - %(levelname)s - %(lineno)d - %(message)s ')
logging.info('***************** Startar *******************')
   
st.set_page_config(page_title="Backtesting av L1 + L2", page_icon="💪")

st.markdown("# 💪 Backtesting av modeller + meta")
st.sidebar.header("💪 Backtesting")

#%%
exp = st.expander('Beskrivning av testet')
exp.write("""
## Först L1-modellerna
Sätt start_datum tex start_datum = '2016-08-01.    

0. Varje modell har indivduella kolumner samt strategi för att ta fram raden.  
    a. Speca per modell i Typ-klassen.
1. Loop över veckor. Learn fram till aktuell vecka och spara modell.  
    a. Skit i cv tillsvidare
    b. Ev Spara modell med aktuell datum i namnet.
2. Predict nästa vecka (generera rad enl modellens strategi)
3. Rätta, Hämta ev priser och Spara resultat som en df i en csv-fil.
4. plot resultat
5. Repeat för varje modell och strategi 

## Gör detsamma för meta-modeller (rf, knn, ridge)
- Använd sparade L1-modeller och generara stack-data från start_datum till aktuell_datum  
- Lär upp L2-modeller på stack-data, använd strategins hyperparms 
- Hur vet vi hur länge meta-modellen skall köra?  
    - Kanske göra ett test innan på ganska stor test-data och spara som hyperparm
- Predict nästa vecka enl strategi för resp L2_modell skapa kolumnerna 'proba_cat1L2', etc  

- Beräkna viktade meta_värden från L2-modellernas output ('proba_cat1L2', etc)
- Ta medelvärdet av L2-modelerna aritmetiskt eller gemoetriskt
- Rätta och spara resultat plus ev priser
- plot resultat
- Repeat för varje meta-modell

"""
          )

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

def starta_upp(df, start_ix=220):
    import datetime

    startdatum = df.datum.unique()[start_ix]

    st.info(f'Startdatum = {startdatum}')

    # init resutat-tabell
    df_resultat = pd.DataFrame(columns=['datum', 't1_vinst', 't2_vinst', 't3_vinst', 't4_vinst',
                                                 't1_utd',   't2_utd',   't3_utd',   't4_utd',
                                                 't1_kostn', 't2_kostn', 't3_kostn', 't4_kostn',
                                                 't1_7', 't2_7', 't3_7', 't4_7',
                                                 't1_6', 't2_6', 't3_6', 't4_6',
                                                 't1_5', 't2_5', 't3_5', 't4_5'
                                        ])

    df_resultat.set_index('datum', drop=True, inplace=True)
    # print(df_resultat.head(0))
    df_resultat.loc[startdatum] = [0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0]
    # for all columns except datum make int
    for col in df_resultat.columns:
        if col != 'datum':
            df_resultat[col] = df_resultat[col].astype(int)

    return df.datum.unique(), df_resultat

def learn_modeller(df_, curr_datum, L2_datum, L1_modeller, L2_modeller, use_features):
    """ Skapar en DataFrame som innehåller input till L2-modellers predictioner
    Args:
        df_ (DataFrame): allt data rensat och tvättat
        curr_datum (string): aktuell datum som skall predikteras
        L2_datum (string): bryt-datum mellan Layer1 och Layer2
        L1_modeller (list): Layer1-modeller
        L2_modeller (list): Layer2-modeller
        use_features (list): Features för Layer1-modeller
    Returns:
        DataFrame: stack_för L2_predictions
        List: L2_features inkluderar proba_features från L1-modeller
    """
    log_print(f'learn_modeller: med curr_datum={curr_datum} och L2_datum={L2_datum}', 'i')
    log_print(f'learn_modeller: orginaldata={df_.shape}', 'i')

    ############# Learn L1-modeller på allt <= L2_datum ############################
    L1_learn_input_df = df_.query('datum < @L2_datum').copy()
    assert L1_learn_input_df.shape[0] > 0, log_print(f'L1_learn_input_df är tom', 'e')

    log_print(f'learn_modeller: L1_input_df shape={L1_learn_input_df.shape} min_dat= \
    {L1_learn_input_df.datum.min()}, max_dat={L1_learn_input_df.datum.max()}', 'i')

    L1_modeller = mod.learn_L1_modeller(L1_modeller, L1_learn_input_df, use_features, save=True)
    
    log_print(f'learn_modeller: L1_modeller är nu lärt {L1_modeller.keys()}', 'i')

    ############# Skapa data till L2 på datum >= L2_datum och datum < curr_datum #############
    # stack_df utökas med proba_xxxxxx från L1-modeller
    stack_df = df_.query('datum >= @L2_datum and datum < @curr_datum').copy()
    assert stack_df.shape[0] > 0, log_print(f'stack_df är tom', 'e')

    stack_df, L2_features = mod.create_L2_input(stack_df, L1_modeller, use_features, with_y=True)
    
    assert stack_df.shape[0] > 0, log_print(f'stack_df är tom', 'e')
    log_print(f'learn_modeller: L2_learn_input_df shape={stack_df.shape} min_dat=\
    {stack_df.datum.min()}, max_dat={stack_df.datum.max()}', 'i')

    assert len([x for x in L2_features if x.startswith('proba_')]) == 4, \
        log_print(f'Antal proba-kolumner skall vara 4', 'e')
    ############# Nu har vi adderat proba_xxxxxx till stack_df ############################

    ####################R Learn L2-modeller på stack_df ###################################
    L2_modeller = mod.learn_L2_modeller(L2_modeller, stack_df, L2_features, save=True)
    log_print(f'learn_modeller: L2_modeller är nu lärt {L2_modeller.keys()}', 'i')

    ############# Skapa data till L2 för curr_datum ############################
    curr_data_df = df_.query('datum == @curr_datum').copy()
    
    log_print(f'learn_modeller: curr_data_df shape={curr_data_df.shape} min_dat=\
    {curr_data_df.datum.min()}, max_dat={curr_data_df.datum.max()}', 'i')

    curr_stack_df, L2_features = mod.create_L2_input(curr_data_df, L1_modeller, use_features, with_y=True)
    assert curr_stack_df.shape[0] > 0, log_print(f'curr_stack_df är tom', 'e')

    log_print(f'learn_modeller: curr_stack_df shape={curr_stack_df.shape} min_dat=\
    {curr_stack_df.datum.min()}, max_dat={curr_stack_df.datum.max()}', 'i')

    assert len([col for col in curr_stack_df.columns if 'proba' in col]) == 4, \
        log_print(f'learn_modeller: curr_stack_df skall ha 4 proba-kolumner', 'e')
        
    ############# Nu ligger L2-proba_kolumner i curr_stack_df  ############################

    return curr_stack_df, L2_features

def välj_ut_hästar_för_spel(strategi, df_proba):
    
    print(f'df_proba {df_proba.shape}')
    if '100' in strategi:
        max_insats = 100
    else:
        max_insats = 360   
    return mod.välj_rad(df_proba, max_insats)

def plot_resultat(df_resultat_, placeholder2, placeholder3):
    import matplotlib.dates as mdates
    
    df_resultat = df_resultat_.copy()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df_resultat.to_csv('df_resultat.csv', index=False)
    
    #### Skapa en line plot för Kumulativ vinst_förlust per strategi över datum ####
    
    # Beräkna kumulativ vinst_förlust per strategi över datum
    df_resultat['cumulative_profit'] = df_resultat.groupby('strategi')[
        'vinst_förlust'].cumsum()
    
    
    # Skapa line plot
    plt.figure()
    plt.close()  # Stäng föregående plot# convert datum to date without time
    df_resultat['datum'] = pd.to_datetime(df_resultat['datum']).dt.date
    
    for strategi in df_resultat['strategi'].unique():
        strategi_data = df_resultat[df_resultat['strategi'] == strategi]
        plt.plot(strategi_data['datum'],
                strategi_data['cumulative_profit'], label=strategi)

    # Använd mdates för att sätta tidsaxeln
    # antal_unika = df_resultat['datum'].nunique()
    # interval = int(antal_unika/6)+1
    # log_print(f'antal_unika={antal_unika}, interval={interval}', 'i')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.title('Kumulativ vinst/förlust över tid')
    # plt.xlabel('Datum')
    plt.ylabel('Kumulativ vinst/förlust')
    plt.legend()
    placeholder2.pyplot()

    #### Skapa en bar plot för antal 5-7 rätt per strategi ####
   
    # skala om 5 och 6 rätt så att de blir i ung samma skala som 7 rätt
    df_resultat['6_rätt/10'] = df_resultat['6_rätt']/10
    df_resultat['5_rätt/100'] = df_resultat['5_rätt']/100
    df_resultat.drop(['5_rätt', '6_rätt'], axis=1, inplace=True)
    
    # Summera 7, 6 och 5 rätt per strategi
    df_sum = df_resultat.groupby('strategi').sum()
    df_sum = df_sum.drop(['vinst_förlust', 'cumulative_profit'], axis=1)
    
    # Skapa bar plot
    # Skapa bar plot
    
    plt.figure()  # Skapa en ny tom figure
    plt.close()  # Stäng föregående plot
    
    df_sum.plot(kind='bar', stacked=False)

    # Loop genom varje rad i df_sum
    for i in range(len(df_sum)):
        # Loop genom varje kolumn i df_sum
        positions = ['right' , 'center', 'left']
        for j in range(len(df_sum.columns)):
            # Skriv text med rundat värde för varje stapel
            plt.text(i, df_sum.iloc[i, j], str(round(df_sum.iloc[i, j], 2)), ha=positions[j], va='bottom')

    # plt.xlabel('Strategi')

    # Sätt övergripande titel för figuren
    plt.suptitle('Antal 5-7 rätt per strategi')

    # Sätt titel på y-axeln
    plt.ylabel('Antal 5-7 rätt')

    # Lägg till en legend
    plt.legend()

    # Visa plot
    placeholder3.pyplot()

def next_datum(df, curr_datum=None, step=1):
    """ Tar fram nästa datum för testet
    Args:
        df (dataframe): all_data.csv
        curr_datum (string, optional): aktuell datum för testet. None betyder första gången.
        step (int, optional): antal datum som ska stegas framåt. Defaults to 1.
        gap (int, optional): antal datum utöver nästa mellan learn och predict. Defaults to 0.
    Returns:
        tuple: L2_datum, ny curr_datum
    """
    
    alla_datum = df.datum.unique()                        ###.astype(str).unique().tolist()
    if curr_datum is None:
        curr_datum = alla_datum[200]  # första gången är det 200 veckor för rejäl start av training   
    
    log_print(f'next_datum 1: curr_datum: {curr_datum}, type: {type(curr_datum)}', 'i')
    log_print(f'next_datum 2: all_datum[0]: {alla_datum[0]}, type: {type(alla_datum[0])}', 'i')
    
    ix2 = [i for i, date in enumerate(alla_datum) if date == curr_datum][0]+step

    # ix2 = np.where(alla_datum == curr_datum)[0][0]+step   # index till ny curr_datum
    ix1 = int(round(ix2/2)+0.5) # index till L2_datum   (learn fr o m L2_datum till curr_datum)
    # learn L1_modeller på allt fram till L2_datum
    
    ############################### L1_datum    
    # Gör så att next_datum är L2_datum.index + 1
    # beräkan 50/50 mha next_datum för L1_datum och L2_datum
    
    if ix2 < len(alla_datum):
        curr_datum = alla_datum[ix2] # nästa omgång att testa på
        L2_datum = alla_datum[ix1]  # till_datum för att lära L1-modellern (L2-modellerna startar from L2_datum)
        curr_datum= pd.to_datetime(curr_datum, format='%Y-%m-%d').date()
        L2_datum= pd.to_datetime(L2_datum, format='%Y-%m-%d').date()
    else:
        curr_datum, L2_datum = None, None
    
    log_print(f'next_datum: curr_datum={curr_datum} type={type(curr_datum)}','i'    ) 
    return L2_datum, curr_datum

def set_up_for_restarting(strategier):
    try:
        df_resultat = pd.read_csv('df_resultat.csv')
    except: # Om det inte finns någon df_resultat.csv
        curr_datum = None
        df_resultat = pd.DataFrame(columns=['datum', 'strategi', 'vinst_förlust', '7_rätt', '6_rätt', '5_rätt'])
        df_resultat['datum'] = pd.to_datetime(df_resultat['datum'], format='%Y-%m-%d').dt.date    
        return df_resultat, curr_datum
    
    df_resultat['datum'] = pd.to_datetime(df_resultat['datum'], format='%Y-%m-%d').dt.date
    # sortera df_resultat på datum från lägsta värdet och uppåt
    df_resultat.sort_values(by=['datum'], inplace=True)
    df_resultat.reset_index(drop=True, inplace=True)
    df_resultat.to_csv('df_resultat.csv', index=False)
    curr_datum = df_resultat.datum.max() 
    print(f'setup curr_datum={curr_datum}')
      
    # Hur många strategier har vi för curr_datum?
    antal_strategier = len(df_resultat[df_resultat.datum == curr_datum])
    assert antal_strategier == len(strategier), log_print(f'antal_strategier={antal_strategier} != len(strategier)={len(strategier)}')
    # Är det samma strategieri df_resultat som vi har i dict strategier?
    strategier_i_df = df_resultat[df_resultat.datum == curr_datum].strategi.unique()
    assert set(strategier_i_df) == set(strategier), log_print(f'strategier_i_df={strategier_i_df} != strategier={strategier}')
    
    return df_resultat, curr_datum
    
def backtest(df, L1_modeller, L2_modeller, step=1, gap=0, proba_val=0.6):
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    
    use_features, cat_features, num_features = mod.read_in_features()
    
    # modeller, cat2mod och xgb2mod som körs rent och kan läras på allt innan curr_datum
    #                    Namn     #hästar  #motst  motst_diff  streck
    cat2mod = tp.Typ('cat2mod', False,      3,     True,      True)
    xgb2mod = tp.Typ('xgb2mod', False,      3,     True,      True)

    # Bygg strategier - ha alltid med minst en strategi med 'outer' och lägg den först i samma grupp
    # 'inner' används i predict och betyder att vi inte behöver köra learn_modeller
    strategier = {
        # 'geo100': ('inner', 'geometric','100','outer', 'layers'),
        # 'weights': ('inner', 'arithmetic', '100','weights'),
        'arit100': ('inner', 'arithmetic', '100', 'weights', 'outer', 'layers'),
        'cat2': ('inner', 'cat2mod'),
        'xgb2': ('inner', 'xgb2mod'),
        
    }
    
    df_resultat, curr_datum = set_up_for_restarting(strategier)
    df_utdelning = pd.read_csv('utdelning.csv')
    
    # Dela upp datan i lär- och test-delar
    L2_datum, curr_datum = next_datum(df, curr_datum=curr_datum, step=step)
    print(f'curr_datum: {curr_datum}, L2_datum: {L2_datum}')
    # För varje unikt datum i test-delen
    while curr_datum is not None: 
        # for strategi_name, (strategi, L1_modeller, L2_modeller, use_features) in strategier.items():
        for strategi_namn, strategi in strategier.items():
            log_print(f'curr_datum: {curr_datum}, {strategi_namn}: {strategi}', 'i')
            placeholder0.empty()
            placeholder0.info(f'Aktuell datum: {curr_datum} {"        "} \n {strategi_namn}: {strategi}')
        
            if 'outer' in strategi:
                # Lär modeller och predict aktuell datum
                curr_stack_df, L2_features = learn_modeller(df, curr_datum, L2_datum, L1_modeller, L2_modeller, use_features)
                log_print(f'curr_stack_df shape={curr_stack_df.shape} min_dat=\
                {curr_stack_df.datum.min()}, max_dat={curr_stack_df.datum.max()}', 'i')
                curr_stack_df.to_csv('backtest_curr_stack_df.csv', index=False)
            else:
                curr_stack_df = pd.read_csv('backtest_curr_stack_df.csv')
   

            if 'cat2mod' in strategi:  # Skit i stacking, använd bara cat2
                log_print(f'cat2mod i strategi', 'i')
                learn_input_df = df.query('datum < @curr_datum')
                _ = mod.learn_L1_modeller({'cat2mod':cat2mod}, learn_input_df, use_features, save=True)
                
                curr_df = df.query('datum == @curr_datum')
                curr_df['meta']= cat2mod.predict(curr_df.copy(), use_features)
                L2_output_df = curr_df    # HACK: använder L2_output_d2 även om L2 inte använts här
            elif 'xgb2mod' in strategi:  # Skit i stacking, använd bara xgb2
                log_print(f'xgb2mod i strategi', 'i')
                learn_input_df = df.query('datum < @curr_datum')
                _ = mod.learn_L1_modeller({'xgb2mod':xgb2mod}, learn_input_df, use_features, save=True)
                
                curr_df = df.query('datum == @curr_datum')
                curr_df['meta']= xgb2mod.predict(curr_df.copy(), use_features)
                L2_output_df = curr_df    # HACK: använder L2_output_d2 även om L2 inte använts här
            else:    
                ############# predict curr_stack_df med L2 dvs läggg till viktad meta #####################
                weights0 = [0.25, 0.25, 0.25, 0.25] 
                weights1 = [0.2702, 0.2712, 0.225, 0.2336]
                weights = weights1 if 'weights' in strategi else weights0
                mean_type = 'arithmetic' if 'arithmetic' in strategi else 'geometric'
                L2_output_df = mod.predict_med_L2_modeller(
                    L2_modeller, curr_stack_df, L2_features, mean_type=mean_type, weights=weights)
                assert L2_output_df.shape[0] > 0, log_print(f'L2_output_df är tom', 'e')
                assert 'meta' in L2_output_df.columns, log_print(f'L2_output_df saknar meta-kolumn', 'e')
                assert L2_output_df.shape[0] > 0, 'L2_output_df.shape[0] empty'

                ############# Nu ligger proba_kolumner plus viktad meta i L2_output_df ############################

            # Välj ut hästar för spel df_proba skall innehålla data för en omgång
            veckans_rad, kostnad = välj_ut_hästar_för_spel(strategi, L2_output_df)
            log_print(f'veckans_rad.shape: {veckans_rad.shape}  Kostnad: {kostnad}')
            
            # Beräkna resultat
            _7_rätt, _6_rätt, _5_rätt, utdelning = mod.rätta_rad(veckans_rad, curr_datum, df_utdelning)
            vinst_förlust =  utdelning - kostnad
            log_print(f'vinst_förlust: {vinst_förlust}  utdelning: {utdelning}  kostnad: {kostnad}','i')
            
            # make a string of the current date
            row = [curr_datum.strftime('%Y-%m-%d'), strategi_namn, vinst_förlust, _7_rätt, _6_rätt, _5_rätt]
            df_resultat.loc[len(df_resultat)] = row
            df_resultat['datum'] = pd.to_datetime(df_resultat['datum'])

            ant_head = df_resultat[df_resultat.datum == df_resultat.datum.max()].shape[0]
            print(df_resultat.tail(ant_head))
    
            # Plot resultat
            placeholder1.empty()
            placeholder2.empty()
            placeholder3.empty()
            plot_resultat(df_resultat, placeholder2, placeholder3)
            
        L2_datum, curr_datum = next_datum(df, curr_datum, step)
        
def kör(df, L1_modeller, L2_modeller, cv=False):

    base_ix = 100  # antal omgångar som vi startar bas-modellerna från i backtesting
    meta_ix = 150  # antal omgångar som vi startar meta-modellerna från i backtesting

    ##################################################################################
    # Bestäm i förväg vilka predictors som varje meta-model skall använda?           #
    # Bestäm också spik-strategi och kelly-strategi för varje meta-model             #
    # Kanske en dict är bra?                                                         #
    ##################################################################################

    datumar, df_resultat = starta_upp(df, base_ix)

    # backtesting
    assert 'y' in df,log_print(f'y saknas i df')  
    df_resultat = backtest(df, L1_modeller, L2_modeller, gap=0, proba_val=0.6, step=1)

    return df_resultat

def main():
    # Skapa v75-instans
    v75 = td.v75(pref='')
    # Hämta data från v75
    df,_ = v75.förbered_data(extra=True, missing_num=False)  # num hanteras av catboost
    # df = v75.lägg_till_kolumner()

    ###############################################################
    # Några idéer på nya kolumner:
    #  -   ❌ streck/sum(streck för avd) - fungerar inte bra. Nästan alla sum == 100 per avd
    #  a - ✔️ plats(streck)/ant_hästar_i_avd (antal startande hästar i avd)
    #  b - ❌ pris / tot_pris_i_avd - går inte att använda ju! pris är ju loppets 1.pris - samma för all i loppet
    #  c - ✔️ kr / tot_kr_i_avd     rel_kr
    #  d - ✔️ Avståndet till ettan (streck)
    #  e - ✔️ hx_bana samma som bana
    #  f - ✔️ hx_kusk samma som kusk

    # -------------- skapa modeller för backtesting
    L1_modeller, L2_modeller = mod.skapa_modeller()
    
    if st.button('kör'):
        df_resultat = kör(df, L1_modeller, L2_modeller, cv=False)

if __name__ == "__main__":
    if True:
        main()
    else: # Kör ren py     
            
        # Skapa v75-instans
        v75 = td.v75(pref='')
        # Hämta data från v75
        # num hanteras av catboost
        df, encoder = v75.förbered_data(extra=True, missing_num=False)

        # -------------- skapa modeller för backtesting
        L1_modeller, L2_modeller = mod.skapa_modeller()

        st.write('Startar "kör" från __name__ == "__main__"')
        df_resultat = kör(df, L1_modeller, L2_modeller, cv=False)

# %%
