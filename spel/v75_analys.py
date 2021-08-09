# %%
# !pip import catboost
import os
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np
import fixa_mer_features as ff2

# %%

def prepare_data(df):
    #strukna = df[df.vodds == 'EJ'][['datum', 'avd',  'häst', 'vodds']]
    df = df[df.vodds != 'EJ']  # Ta bort alla strukna

    # print(f"\n\ndf.shape före fixa_mer_features: {df.shape}\n")
    # df = ff2.fixa_mer_features(df, False)
    # print(f"\n\ndf.shape efter fixa_mer_features: {df.shape}")

    df.ålder = df.ålder.astype('float')

    # strecken
    #print(f'streck före med NaNs: {df.streck.isna().sum()}')
    # df.streck = df.streck.str.replace('%', '')
    # df.streck = df.streck.replace(' ', np.nan)
    df.streck = df.streck.astype('float')
    print(f'streck efter med NaNs: {df.streck.isna().sum()}')

    df.sort_values(by=['datum', 'bana', 'avd'], inplace=True)

    return df
# %%


def reserver(omg, ant_rader):

    #print(f"Antal prob_order>{ant_rader} med insats>0")

    by_prob3i = omg[(omg.prob_order == ant_rader+1)][['avd', 'häst', 'vodds', 'plac',
                                                      'prob_order', 'f', 'insats']].sort_values(by='insats', ascending=False)

    by_ins = omg[(omg.prob_order > ant_rader) & (omg.insats > 0)][['avd', 'häst', 'vodds',
                                                                   'plac', 'prob_order', 'f', 'insats']].sort_values(by='insats', ascending=False)

    by_prob3f = omg[(omg.prob_order == ant_rader + 1)][['avd', 'häst', 'vodds', 'plac',
                                                        'prob_order', 'f', 'insats']].sort_values(by=['prob_order', 'f'], ascending=False)

    return by_prob3i, by_prob3f, by_ins


# %%
# SKAPA V75 RADER


def v75(omgorg, modeller, model_name, selected_features, ant_rader=2, hist=False):
    omg=omgorg.copy(deep=True)
    ## ladda model ##
    model = CatBoostClassifier()
    if hist:
        model.load_model('modeller_history/model_senaste', format='cbm')  
    else:          
        model.load_model(modeller+'/'+model_name, format='cbm')
    
    # predict
    # print(model.feature_names_)
    # print(omg[selected_features].info())
    proba = model.predict_proba(omg[selected_features])

    # Ordna proba per avdelning samt beräkna Kelly
    kassa = 200
    omg['proba'] = proba[:, 1]
    omg['f'] = (omg.proba*omg.vodds - 1) / (omg.vodds-1)  # kelly formel
    omg['spela'] = omg.f > 0
    omg['insats'] = omg.spela * omg.f * kassa

    # Ta ut de n bästa per avd
    omg.sort_values(['datum', 'avd', 'proba'], ascending=[
                    True, True, False], inplace=True)
    proba_order = omg.groupby(['datum', 'avd']).proba.cumcount()

    omg['prob_order'] = proba_order+1

    return(omg)

# %%

def start_slut(modellen):
    path = 'modeller_'+modellen
    all_models = os.listdir(path)[0:-1]

    return path, all_models, all_models[0][6:], all_models[-1][6:]

# %%

# hist=True betyder att hist-modellen används men den använder en annan modell som hjälp
def analyze(modellen, selected_features, utdelningar, komplett,hist=False):
    print('analyze history' if hist else modellen)

    path, all_models, _, _ = start_slut(modellen)
    antal_rader = 2

    ## För varje omgång ##
    rätt7 = 0
    rätt6 = 0
    rätt5 = 0
    summaUtd = 0
    model_name = all_models[0]
    
    #print('stardatum:', model_name[6:])
    i = 0
    for datum in all_models[1:]:
        datum = pd.to_datetime(datum[6:])
        model_name = all_models[i]
        #print(datum, model_name)

        omg = komplett[komplett.datum == datum]
        omg = v75(omg, path, model_name, selected_features,hist=hist)

        #veckans_resultat = f"{datum}: {omg[(omg.plac==1)].prob_order.max()}"

        # plocka ut vinnande omgångar
        ant_rätt = (omg[(omg.plac == 1)].prob_order <= antal_rader).sum()

        if ant_rätt == 7:
            rätt7 += 1
            utdelning = utdelningar[utdelningar.datum == datum]['7rätt'].values
            summaUtd += utdelning

            print(
                f'{datum}: ant rätt={ant_rätt} max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders {utdelning}')

        elif ant_rätt == 6:
            by_prob3i, by_prob3f, by_ins = reserver(omg, antal_rader)

            if len(by_ins) > 0 and by_ins.iloc[0].plac == 1:
                rätt7 += 1
                utdelning = utdelningar[utdelningar.datum ==datum]['7rätt'].values
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 ins {utdelning}')
            if len(by_ins) > 1 and by_ins.iloc[1].plac == 1:
                rätt7 += 1
                utdelning = utdelningar[utdelningar.datum ==datum]['7rätt'].values
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 insb {utdelning}')
            elif len(by_prob3i) > 0 and by_prob3i.iloc[0].plac == 1:
                rätt7 += 1
                utdelning = utdelningar[utdelningar.datum ==datum]['7rätt'].values
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 3i {utdelning}')
            elif len(by_prob3i) > 1 and by_prob3i.iloc[1].plac == 1:
                rätt7 += 1
                utdelning = utdelningar[utdelningar.datum ==datum]['7rätt'].values
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 3ib {utdelning}')
            elif len(by_prob3f) > 0 and by_prob3f.iloc[0].plac == 1:
                rätt7 += 1
                utdelning = utdelningar[utdelningar.datum ==datum]['7rätt'].values
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 3f {utdelning}')
            elif len(by_prob3f) > 1 and by_prob3f.iloc[1].plac == 1:
                rätt7 += 1
                utdelning = utdelningar[utdelningar.datum ==datum]['7rätt'].values
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 3fb {utdelning}')
            else:
                rätt6 += 1
                # print(f'datum: {datum} summaUtd: {summaUtd}')
                utdelning = utdelningar[utdelningar.datum ==datum]['6rätt'].values
                # print(f'utdelning: {utdelning}')
                summaUtd += utdelning
                print(
                    f'{datum}: ant_rätt={ant_rätt} max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders')
        elif ant_rätt == 5:
            by_prob3i, by_prob3f, by_ins = reserver(omg, antal_rader)

            if len(by_ins) > 1 and (by_ins.iloc[0].plac == 1 and by_ins.iloc[1].plac == 1):
                rätt6 += 1
                summaUtd += utdelningar[utdelningar.datum == datum]['6rätt'].values
                print(
                    f'{datum}: ant_rätt={ant_rätt}+2 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+2 ins')
            elif len(by_prob3i) > 1 and by_prob3i.iloc[0].plac == 1 and by_prob3i.iloc[1].plac == 1:
                rätt6 += 1
                summaUtd += utdelningar[utdelningar.datum ==
                                        datum]['6rätt'].values
                print(
                    f'{datum}: ant_rätt={ant_rätt}+2 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+2 3i')
            elif len(by_prob3f) > 1 and by_prob3f.iloc[0].plac == 1 and by_prob3f.iloc[1].plac == 1:
                rätt6 += 1
                summaUtd += utdelningar[utdelningar.datum ==
                                        datum]['6rätt'].values
                print(
                    f'{datum}: ant_rätt={ant_rätt}+2 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+2 3f')
            elif len(by_ins) == 1 and (by_ins.iloc[0].plac == 1):
                rätt6 += 1
                summaUtd += utdelningar[utdelningar.datum ==
                                        datum]['6rätt'].values
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 ins')
            elif len(by_prob3i) == 1 and (by_prob3i.iloc[0].plac == 1):
                rätt6 += 1
                summaUtd += utdelningar[utdelningar.datum ==
                                        datum]['6rätt'].values
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 3i')
            elif len(by_prob3f) == 1 and (by_prob3f.iloc[0].plac == 1):
                rätt6 += 1
                summaUtd += utdelningar[utdelningar.datum ==
                                        datum]['6rätt'].values
                print(
                    f'{datum}: ant_rätt={ant_rätt}+1 max_prob={omg[omg.plac == 1].prob_order.max()}  2-raders+1 3f')
            else:
                summaUtd += utdelningar[utdelningar.datum ==
                                        datum]['5rätt'].values
            rätt5 += 1
        i += 1

    print(f"Antal veckor: {i}")
    
    modnamn=modellen
    if hist:
        modnamn = 'history'
        
    print(f'{modellen}: {rätt7} + {rätt6} + {rätt5}: {summaUtd[0]}')
    print()


# %%
## Läs in komplett ##
## och prepare data ##
komplett = prepare_data(pd.read_csv('mer_komplett.csv'))

# %%
def deltan(df):
    df['delta1'] = df.h1_dat-df.h2_dat
    df['delta2'] = df.h2_dat-df.h3_dat
    df['delta3'] = df.h3_dat-df.h4_dat
    df['delta4'] = df.h4_dat-df.h5_dat
    df['senast'] = df.datum-df.h1_dat
    return df 

def ber_perf(pris,plac):
    return np.sqrt(pris * np.exp(15-plac))

def perf(df):
    df['h1_perf'] = ber_perf(df.h1_pris,df.h1_plac)
    df['h2_perf'] = ber_perf(df.h2_pris,df.h2_plac)
    df['h3_perf'] = ber_perf(df.h3_pris,df.h3_plac)
    df['h4_perf'] = ber_perf(df.h4_pris,df.h4_plac)
    df['h5_perf'] = ber_perf(df.h5_pris,df.h5_plac)
    return df
  
    
# %%
#### Hela loopen med alla modeller #####
hist=True
utdelningar = pd.read_csv('utdelning.csv')
utdelningar.datum = pd.to_datetime(utdelningar.datum)
_, _, start, slut = start_slut('all_100')
print(f'datum: {start} - {slut}')
print()

if hist:
    # nya kolumner för datumavstånd
    komplett.datum=pd.to_datetime(komplett.datum)
    komplett.h1_dat=pd.to_datetime(komplett.h1_dat)
    komplett.h2_dat=pd.to_datetime(komplett.h2_dat)
    komplett.h3_dat=pd.to_datetime(komplett.h3_dat)
    komplett.h4_dat=pd.to_datetime(komplett.h4_dat)
    komplett.h5_dat=pd.to_datetime(komplett.h5_dat)
    komplett.senast = pd.to_timedelta(komplett.senast)
    komplett.delta1 = pd.to_timedelta(komplett.delta1)
    komplett.delta2 = pd.to_timedelta(komplett.delta2)
    komplett.delta3 = pd.to_timedelta(komplett.delta3)
    komplett.delta4 = pd.to_timedelta(komplett.delta4)
    
    # komplett=perf(komplett)
    # komplett=deltan(komplett)
    
    #### history test ###
    add_features=['senast', 'delta1','delta2', 'delta3', 'delta4',
                'h1_perf', 'h1_dist', 'h1_kmtid', 'h1_odds',   
                'h2_perf', 'h2_dist', 'h2_kmtid', 'h2_odds',
                'h3_perf', 'h3_dist', 'h3_kmtid', 'h3_odds', 
                'h4_perf', 'h4_dist', 'h4_kmtid', 'h4_odds', 
                'h5_perf', 'h5_dist', 'h5_kmtid', 'h5_odds']
    cat_features = ['häst','kusk','bana','h1_bana', 'h1_kusk', 'h2_bana', 'h2_kusk','h3_bana', 'h3_kusk','h4_bana', 'h4_kusk','h5_bana', 'h5_kusk']
    sel_features = cat_features+add_features
    
    modellen = 'streck_100' # Stjäl en modell för att genomföra loopen i analys
    analyze(modellen, sel_features, utdelningar, komplett, hist=True)
    
#### podds test ###
selected_features = ['häst', 'kusk', 'bana', 'kön','podds',]
modellen = 'podds_100'                         # 
analyze(modellen, selected_features, utdelningar, komplett)

#### small ###
selected_features = ['häst', 'kusk', 'bana', 'spår', 'dist', 'ålder']
modellen = 'small_100'                         # 0 + 1 + 8: 198
analyze(modellen, selected_features, utdelningar, komplett)

#### pris ####
selected_features = ['häst', 'kusk', 'bana', 'spår', 'dist', 'ålder', 'pris']
modellen = 'pris_100'                          # 1 + 2 + 3: 2965
analyze(modellen, selected_features, utdelningar, komplett)

#### streck+pris ####
selected_features = ['häst', 'kusk', 'bana', 'spår', 'dist', 'ålder', 'streck','pris']
modellen = 'strpr_100'                         # 3 + 15 + 34: 5800
analyze(modellen, selected_features, utdelningar, komplett)

#### all ####
selected_features = ['häst', 'kusk', 'bana', 'spår', 'dist',
                     'ålder', 'streck', 'vodds', 'podds', 'bins', 'pris']
modellen = 'all_100'                            # 6 + 17 + 37: 13807
analyze(modellen, selected_features, utdelningar, komplett)

#### streck ####
selected_features = ['häst', 'kusk', 'bana', 'spår', 'dist', 'ålder', 'streck']
modellen = 'streck_100'                         # 6 + 15 + 37: 13936
analyze(modellen, selected_features, utdelningar, komplett)

#### bins ####
selected_features = ['häst','kusk','bana','bins']
modellen = 'bins_100'
analyze(modellen, selected_features, utdelningar, komplett)


# %%
# model = CatBoostClassifier()
# model.load_model('modeller_bins_100'+'/'+'model_2021-02-13', format='cbm')
# print(model.feature_names_)

    
# %%
