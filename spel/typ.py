import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool


def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    df.drop(['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
            'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'], axis=1, inplace=True)
    if remove_mer:
        df.drop(remove_mer, axis=1, inplace=True)

    return df

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

    def learn(self, X_, y=None, depth=None, learning_rate=None, l2_leaf_reg=None, iterations=500, save=True, verbose=False):
        # X_ måste ha datum och avd

        cbc = CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
            loss_function='Logloss', eval_metric='AUC', verbose=verbose)

        X = self.prepare_for_model(X_)
        if not self.streck:
            X.drop('streck', axis=1, inplace=True)

        X, cat_features = prepare_for_catboost(X)

        X = remove_features(X, remove_mer=['datum', 'avd'])
        
        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'

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

        assert len(X.columns) == len(
            model.feature_names_), f'len(X.columns)  != len(model.feature_names_) in predict {self.name}'
        assert set(X.columns) == set(
            model.feature_names_), 'features in model and in X not equal'
        
        X = X[model.feature_names_]
        print('predict '+self.name)
        # print(model.get_feature_importance(prettified=True)[:3])

        return model.predict_proba(X)[:, 1]
