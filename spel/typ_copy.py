import pandas as pd
import numpy as np
import pickle
import json
from catboost import CatBoostClassifier, Pool

def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    if 'vodds' in df.columns:
        df.drop(['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
                'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'], axis=1, inplace=True)
    if remove_mer and 'avd' in df.columns:
        df.drop(remove_mer, axis=1, inplace=True)

    return df

# remove NaN for cat_features in X and return (X, cat_features)
# ta bort alla features som inte används innan call
def prepare_for_catboost(X_, features=[],remove=True, verbose=False):
    X = X_.copy()
    
    if remove:
        Xtemp = remove_features(X, remove_mer=['avd', 'datum'])
    else:
        Xtemp = X.copy()    

    if len(features) > 0:
        Xtemp = Xtemp[features]
        
    # get numerical features and cat_features
    num_features = list(Xtemp.select_dtypes(include=[np.number]).columns)
    cat_features = list(Xtemp.select_dtypes(include=['object']).columns)

    # check cat_features isna
    if verbose:
        print('Number of NaN in cat before:', X[cat_features].isna().sum()[
          X[cat_features].isna().sum() > 0].sort_values(ascending=False).sum())

    # impute 'missing' for all NaN in cat_features
    X[cat_features] = X[cat_features].fillna('missing')
    if verbose:
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
def lägg_in_diff_motståndare(X_, ant_motståndare):
    X = X_.copy()

    # set X['motståndare1'] to largest streck in every avd
    grouped = X.groupby(['datum', 'avd'])['streck']
    X['diff1'] = grouped.transform(max) - X.streck

    for i in range(2, ant_motståndare+1):
        # set X['motståndare'+str(i)] to ith largest streck in every avd
        X['diff' +
            str(i)] = grouped.transform(lambda x: x.nlargest(i).min()) - X.streck

    return X

class Typ():
    #                  name,   #häst      #motst,  motst_diff, streck, test,        pref
    def __init__(self, name, ant_hästar, motst_ant, motst_diff, streck, test=False, pref=''):
        assert (motst_diff == False and motst_ant == 0) or (motst_ant > 0)
        # assert (ant_favoriter == 0 and only_clear == False) or (ant_favoriter > 0)
        # assert kelly == False or kelly == None, 'kelly skall inte användas mer'
        
        self.name = name                # string - används för filnamn mm

        # extra features att inkludera 
        self.ant_hästar = ant_hästar    # bool - skapa kol med antal hästar per avdelning
        self.motst_ant = motst_ant      # int  - inkludera n features med bästa motståndare (streck)
        self.motst_diff = motst_diff    # bool - ovanstående med diff (streck) istf fasta värden
        self.streck = streck            # bool - inkludera streck som feature
        print('streck:', self.streck,'i init för', self.name)
    
        # urval av rader
        # self.proba = proba              # bool - för prioritering vid urval av rader
        # self.kelly = kelly              # bool - för prioritering vid urval av rader
        # self.ant_favoriter = ant_favoriter # int  - för hur många favoriter (avd där endast en häst spelas) som ska användas
        # self.only_clear = only_clear       # bool - för att bara avvända klara favoriter
        
        self.pref = pref                # string - prefix för map/filnamn
        
        ##### test #####
        if test:
            self.rel_kr = True              # bool - skapa kol med relativt kr gentemot motståndarna
            self.rel_rank = True            # bool - skapa kol med relativ rank gentemot motståndarna
            self.streck_avst = True         # bool - skapa kol med streck avstånd gentemot motståndarna
            self.hx_samma_bana = True       # bool - skapa kol med hx.bana som bana
            self.hx_sammam_kusk = True      # bool - skapa kol med hx.kusk som kusk
            print('Gör denna till produktion')
        else:
            self.rel_kr = False
            self.rel_rank = False
            self.streck_avst = False
            self.hx_samma_bana = False
            self.hx_sammam_kusk = False    

    def get_name(self):
        return self.name
    
    def load_model(self):
        with open(self.pref+'modeller/'+self.name+'.model', 'rb') as f:
            print('Loading model:', self.name)
            model = pickle.load(f)
        return model

    def save_model(self, model):
        print(f'Sparar {self.name+".model"}')
        with open(self.pref+'modeller/'+self.name+'.model', 'wb') as f:
            pickle.dump(model, f)

    def prepare_for_model(self, X_,verbose=False):
        # X_ måste ha datum och avd
        X = X_.copy()
        if verbose:
            print(self.name, end=', ')
        if self.ant_hästar:
            if verbose:
                print('Lägg in ant_hästar', end=', ')
            X = lägg_in_antal_hästar(X)
        if self.motst_diff:
            if verbose:
                print('Lägg in diff motståndare', end=', ')
            X = lägg_in_diff_motståndare(X, self.motst_ant)
        elif self.motst_ant > 0:
            if verbose:
                print('Lägg in motståndare', end=', ')
            X = lägg_in_motståndare(X, self.motst_ant)
        # Behåll streck ända tills learn och predict (används för prioritera rader)
        
        #### test ###
        # vi har några nya tillagda i __init__ ovan.
        # Dessa kolumner läggs till av travdata.py och skall selekteras i prepare_for_model
        if not self.rel_kr:
            X = X.drop(['rel_kr'], axis=1)
        if not self.rel_rank:
            X = X.drop(['rel_rank'], axis=1)
        if not self.streck_avst:
            X = X.drop(['streck_avst'], axis=1) 
        if not self.hx_samma_bana:
            X = X.drop(['h1_samma_bana','h2_samma_bana','h3_samma_bana'], axis=1)   # hx_samma_bana 
        if not self.hx_sammam_kusk:
            X = X.drop(['h1_samma_kusk','h2_samma_kusk','h3_samma_kusk'], axis=1)   # hx_sammam_kusk
        
        if verbose:
            print('Test')
        return X
    
    def learn(self, X_, y=None, X_test_=None, y_test=None, params=None, iterations=1000, save=True, verbose=False):
        # X_ måste ha datum och avd
        assert X_ is not None, 'X är None'
        print('Learning', self.name)
        X = X_.copy()
        X_test=None
        if X_test_ is not None:
            X_test = X_test_.copy()
            
        assert 'streck' in list(X_.columns), 'streck saknas i learn X'
        if params==None:
            #read params from file
            with open(self.pref+'optimera/params_'+self.name+'.json', 'rb') as f:
                params = json.load(f)
                # print(params)
                params = params['params']

        iterations = params['iterations'] if 'iterations' in params else iterations
        params.pop('iterations')  # remove iterations from params
        cbc = CatBoostClassifier(**params,
            iterations=iterations,
            loss_function='Logloss', eval_metric='AUC', verbose=verbose)
        
        X = self.prepare_for_model(X_)
        if not self.streck:
            X.drop('streck', axis=1, inplace=True)

        X, cat_features = prepare_for_catboost(X,verbose=verbose)
        X = remove_features(X, remove_mer=['datum', 'avd'])
        
        assert X[cat_features].isnull().sum().sum() == 0, 'there are NaN values in cat_features'

        if X_test is None or y_test is None:
            cbc.fit(X, y, cat_features, use_best_model=False)
        else:    
            X_test = self.prepare_for_model(X_test,verbose=verbose)
            if not self.streck:
                X_test.drop('streck', axis=1, inplace=True)

            X_test, _ = prepare_for_catboost(X_test,verbose=verbose)
            X_test = remove_features(X_test, remove_mer=['datum', 'avd'])
            assert X.columns.tolist() == X_test.columns.tolist(), 'X and X_test have different columns'
            eval_pool = Pool(X_test, y_test, cat_features=cat_features)
            cbc.fit(X,y,cat_features=cat_features,eval_set=eval_pool, use_best_model=True,early_stopping_rounds=50)
        
            
        if verbose:
            print('best score', cbc.best_score_)
        if save:
            self.save_model(cbc)
        return cbc


    def predict(self, X_,verbose=False,model=None):
        # X_ måste ha datum och avd
        assert 'streck' in list(X_.columns), f'streck saknas i predict X ({self.name})'
        
        X = self.prepare_for_model(X_)
        assert 'streck' in list(X.columns), f'streck saknas efter prepare_for_model i predict X ({self.name})'
        
        if model==None:
            model = self.load_model()
    
        if not self.streck:
            print('drop streck')
            X.drop('streck', axis=1, inplace=True)

        # print('Före prepare for catb - model', self.name,  'columns=:')
        # print('X.columns in predict',list(X.columns))
        # print('features in predict ', list(model.feature_names_))
        
        X, cat_features = prepare_for_catboost(X, model.feature_names_)
        # print('Efter prepare for catb - model', self.name)
        # all features in model
        X = remove_features(X, remove_mer=['datum', 'avd'])
        
        the_diff= list(set(model.feature_names_) - set(X.columns.tolist())) + list(set(X.columns.tolist())- set(model.feature_names_) )  # the difference between them
        assert len(X.columns) == len(model.feature_names_), f'{len(X.columns)}  != {len(model.feature_names_)} {the_diff} in predict {self.name}'
        assert set(X.columns) == set(model.feature_names_), f'features in model and in X not equal {the_diff} in predict {self.name}'
        
        X = X[model.feature_names_]
        if verbose:
            print('predict '+self.name)
        # print(model.get_feature_importance(prettified=True)[:3])

        return model.predict_proba(X)[:, 1]
    
    # method that retruns all the self variables
    def get_params(self):
        return {k:v for k,v in self.__dict__.items() if k[0]!='_'}  