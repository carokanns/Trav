# class_model
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from catboost import Pool
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np

# %%
MODELPATH = 'C:\\Users/peter/Google Drive/Colab Notebooks/Småprojekt/'


# def bins(df):
#     bins = pd.DataFrame({
#         'fr': [1.0, 2.1,  2.9,   3.6,  4.1,  4.8,
#                5.6, 6.3,  7.1,   7.9,  8.7,  9.7,
#                10.7, 11.9, 13.2, 14.8, 16.5, 18.1,
#                20.2, 22.5, 25.2, 28.0, 31.5, 35.3,
#                39.6, 44.7, 50.0, 56.7, 64.3, 75.4],

#         'to': [2.1, 2.9,   3.6,  4.1,  4.8, 5.6,
#                6.3, 7.1,   7.9,  8.7,  9.7, 10.7,
#                11.9, 13.2, 14.8, 16.5, 18.1, 20.2,
#                22.5, 25.2, 28.0, 31.5, 35.3, 39.6,
#                44.7, 50.0, 56.7, 64.3, 75.4, 2000],

#         'bin': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
#                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#     })

#     val = bins.loc[:, 'fr':'to'].apply(tuple, 1).tolist()
#     indx = pd.IntervalIndex.from_tuples(val, closed='right')
#     df['bins'] = bins.loc[indx.get_indexer(df['vodds']), 'bin'].values

#     # Alternativ lösning
#     # res=np.dot((df['vodds'].values[:,None] >= bins['fr'].values) &
#     #           (df['vodds'].values[:,None] < bins['to'].values),
#     #           bins['bin']
#     # )
#     # df['bins'] = res

#     return df


def resultat(pool, model_name, modelpath):
    import pickle

    if modelpath:
        # load the model from disk
        loaded_model = pickle.load(open(modelpath+model_name, 'rb'))

    result = loaded_model.predict(pool)

    return result, loaded_model


def resultat_proba(data, model_name, modelpath):
    import pickle

    if modelpath:
        # load the model from disk
        loaded_model = pickle.load(open(modelpath+model_name, 'rb'))

    result = loaded_model.predict_proba(data)

    return result[:, 1], loaded_model


class Model:
    def __init__(self, name, odds_fr, odds_to, thresh, proba_lim=None, häst_suf='h', med_odds=False, bins=True):
        self.name = name
        self.med_odds = med_odds    # tränat med vodds
        self.odds_fr = odds_fr
        self.odds_to = odds_to
        self.thresh = thresh

        self.proba_lim = proba_lim  # -1 betyder använd inte predict_proba
        if proba_lim == None:
            self.proba_lim = 0.0

        self.bins = bins
        self.häst_suf = häst_suf
        self.trained_model = None
        print(name, 'odds_fr', odds_fr, 'odds_to', odds_to, 'thresh', thresh, 'proba_lim', proba_lim,
              'häst_suf', häst_suf, 'med_odds', med_odds, 'bins', bins)

    def result(self, data, nm):
        dd = data.drop(['datum', 'avd'], axis=1)

        # if self.bins == False:
        #   dd.drop('bins', inplace=True)

        if not self.med_odds:
            dd = dd.drop('vodds', axis=1)

        if self.name.find('CatB') == 0:
            cat_features = ['start', 'spår', 'h1_spår',
                            'h2_spår', 'h3_spår', 'h4_spår', 'h5_spår', 'häst']
            # print(dd[cat_features].info())
            if self.häst_suf == 'u':
                cat_features = ['start', 'spår', 'h1_spår',
                                'h2_spår', 'h3_spår', 'h4_spår', 'h5_spår']
                dd = dd.drop('häst', axis=1)

            dd = Pool(
                cat_features=cat_features,
                data=dd
            )
        else:
            dd = dd.drop('häst', axis=1)

        if self.proba_lim == None or self.proba_lim >= 0:
            #print('proba', self.proba_lim)
            if self.trained_model != None:
                res = self.trained_model.predict_proba(dd)
                res = res[:, 1]
            else:
                res, self.trained_model = resultat_proba(
                    dd, self.name, MODELPATH)  # proba_lim value
        else:
            #print('predict', self.proba_lim)
            if self.trained_model != None:
                res = self.trained_model.predict(dd)
            else:
                res, self.trained_model = resultat(dd, self.name, MODELPATH)

        dd = pd.DataFrame()
        dd[nm+'_res'] = res
        dd[nm+'_vodds'] = (data.vodds >= self.odds_fr)

        dd[nm+'_vodds'] = dd[nm+'_vodds'] * (data.vodds <= self.odds_to)
        if self.proba_lim == None or self.proba_lim >= 0:
            dd[nm+'_förv'] = res * data.vodds
            dd[nm] = dd[nm+'_förv'] >= self.thresh
            dd[nm] = dd[nm] * (dd[nm+'_res'] >= self.proba_lim)
        else:
            dd[nm] = dd[nm+'_res'] >= self.thresh

        dd[nm] = dd[nm] * dd[nm+'_vodds']

        return dd

    def set_trained_model(self, model):
        self.trained_model = model

# %%
