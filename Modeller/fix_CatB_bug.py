# class_model
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import fixa_features as ff
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython import get_ipython
from catboost import Pool

# %%

# %%
MODELPATH = 'C:\\Users/peter/Google Drive/Colab Notebooks/Småprojekt/'


def bins(df):
    bins = pd.DataFrame({'fr': [1.0, 2.2, 3.0, 3.8, 4.4, 5.2,
                                6.0, 6.9, 7.7, 8.7, 9.7, 10.8,
                                12.1, 13.6, 15.4, 17.3, 19.1, 21.6,
                                24.4, 27.3, 30.9, 34.8, 39.3, 44.5,
                                50.1, 56.7, 64.5, 75.3, 89.5, 111.7,
                                553.5],
                         'to': [2.2, 3.0, 3.8, 4.4, 5.2,
                                6.0, 6.9, 7.7, 8.7, 9.7, 10.8,
                                12.1, 13.6, 15.4, 17.3, 19.1, 21.6,
                                24.4, 27.3, 30.9, 34.8, 39.3, 44.5,
                                50.1, 56.7, 64.5, 75.3, 89.5, 111.7,
                                553.5, 2000],
                         'bin': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                 30]
                         })

    val = bins.loc[:, 'fr':'to'].apply(tuple, 1).tolist()
    indx = pd.IntervalIndex.from_tuples(val, closed='right')
    dff = df.copy()
    dff['bins'] = bins.loc[indx.get_indexer(dff['vodds']), 'bin'].values

    return dff


def fitCat(THE_ODDS, HÄST):
    params = {'med': {'fit': {'colsample_bylevel': 0.9127, 'depth': 1, 'l2_leaf_reg': 19.31,  # target': -0.2490: 54-46; vinst=747
                              'leaf_estimation_iterations': 5, 'learning_rate': 0.1872,
                              'min_data_in_leaf': 103, 'n_estimators': 1656, 'subsample': 0.9915},
                      'lim': {'odds_fr': 2.7, 'odds_to': 48.61, 'thresh': 1.42}, },  # target': 41.5

              'utan': {'fit': {'colsample_bylevel': 0.7259, 'depth': 1.2, 'l2_leaf_reg': 2.04,  # target' -0.2551: 79-21; vinst=4088
                               'leaf_estimation_iterations': 1.94, 'learning_rate': 0.0375, 'n_estimators': 1154,
                               'min_data_in_leaf': 145.7, 'subsample': 0.3961},
                       'lim': {'odds_fr': 1.14, 'odds_to': 52.5, 'thresh': 1.79}, },  # target: 78.5
              # utan odds, utan häst med samma params: 76-24 snitt 158.99 spelade av 5099 vinst% 26.0 tot_vinst 4115.0
              }

    #HÄST = True
    #THE_ODDS = 'med'
    print('THE_ODDS =', THE_ODDS, 'HÄST =', HÄST)

    verbose = False
    file_name = F'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\travlopp\\komplett.csv'
    df = pd.read_csv(file_name)

    df = ff.fix_features(df)
    df.reset_index(inplace=True, drop=True)

    if HÄST == False:
        df.drop('häst', axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['datum', 'plac'], axis=1), df.plac == 1,
                                                        test_size=0.2, random_state=202006)

    # vodds_train = X_train.vodds
    # vodds_test = X_test.vodds

    ###### VODDS #######
    if THE_ODDS == 'utan':
        X_train.drop('vodds', axis=1, inplace=True)
        X_test.drop('vodds', axis=1, inplace=True)

    cat_features = ['start']
    if HÄST:
        cat_features = ['start', 'häst']

    y_train = y_train*1
    y_test = y_test*1
    train_pool = Pool(
        cat_features=cat_features,
        data=X_train,
        label=y_train,
    )
    test_pool = Pool(
        cat_features=cat_features,
        data=X_test,
        label=y_test
    )

    if verbose:
        print('start fit')

    from catboost import CatBoostClassifier
    params[THE_ODDS]['fit']['depth'] = int(
        round(params[THE_ODDS]['fit']['depth']))
    params[THE_ODDS]['fit']['n_estimators'] = int(
        round(params[THE_ODDS]['fit']['n_estimators']))
    params[THE_ODDS]['fit']['leaf_estimation_iterations'] = int(
        round(params[THE_ODDS]['fit']['leaf_estimation_iterations']))
    params[THE_ODDS]['fit']['min_data_in_leaf'] = int(
        round(params[THE_ODDS]['fit']['min_data_in_leaf']))
    cbc = CatBoostClassifier(
        **params[THE_ODDS]['fit'],
        scale_pos_weight=1,
        eval_metric='Logloss',
        early_stopping_rounds=100,
    )

    #print(X_train, y_train)
    #print(X_test, y_test)
    cbc.fit(train_pool,
            eval_set=test_pool,
            use_best_model=True,
            verbose=False)

    print('model is fitted: {}'.format(cbc.is_fitted()))
    print('best iteration', cbc.best_iteration_)
    print('model params:\n{}'.format(cbc.get_params()))

    return cbc
