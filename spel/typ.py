import pandas as pd
import numpy as np
import pickle
import json
from catboost import CatBoostClassifier, Pool
import xgboost as xgb


def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    print(type(df))
    if 'vodds' in df.columns:
        df.drop([ 'vodds', 'podds', 'bins', 'h1_dat',
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

# behålla alla NaN
# ta bort alla features som inte används innan call
def prepare_for_xgboost(X_, ohe=True, pred=False, features=[],remove=True, verbose=False, pref=''):
    X = X_.copy()
    
    if remove:
        Xtemp = remove_features(X, remove_mer=['avd', 'datum'])
    else:
        Xtemp = X.copy()    

    if len(features) > 0:
        Xtemp = Xtemp[features]
        
    # get numerical features and cat_features
    cat_features = list(Xtemp.select_dtypes(include=['object']).columns)
    num_features = list(set(Xtemp.columns) - set(cat_features))
    
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
    ITERATIONS = 1000
    EARLY_STOPPING_ROUNDS = 50
    #                  name,   #häst      #motst,  motst_diff, streck, pref
    def __init__(self, name, ant_hästar, motst_ant, motst_diff, streck, pref=''):
        assert (motst_diff == False and motst_ant == 0) or (motst_ant > 0)
        
        self.name = name                # string - används för filnamn mm

        
        # Dessa features läggs till av travdata.py men kan ev selekteras bort i prepare_for_model
        
        self.ant_hästar = ant_hästar    # bool - skapa kol med antal hästar per avdelning
        self.motst_ant = motst_ant      # int  - inkludera n features med bästa motståndare (streck)
        self.motst_diff = motst_diff    # bool - ovanstående med diff (streck) istf fasta värden
        self.streck = streck            # bool - inkludera streck som feature
        print('streck:', self.streck,'i init för', self.name)
        self.rel_kr = True              # bool - skapa kol med relativt kr gentemot motståndarna
        self.rel_rank = True            # bool - skapa kol med relativ rank gentemot motståndarna
        self.streck_avst = True         # bool - skapa kol med streck avstånd gentemot motståndarna
        self.hx_samma_bana = True       # bool - skapa kol med hx.bana som bana
        self.hx_sammam_kusk = True      # bool - skapa kol med hx.kusk som kusk
            
        self.pref = pref                # string - prefix för map/filnamn
        
        
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
        columns_to_drop = []
        if not self.rel_kr:
            columns_to_drop.append("rel_kr")
        if not self.rel_rank:
            columns_to_drop.append("rel_rank")
        if not self.streck_avst:
            columns_to_drop.append("streck_avst")
        if not self.hx_samma_bana:
            columns_to_drop += ["h1_samma_bana", "h2_samma_bana", "h3_samma_bana"]
        if not self.hx_sammam_kusk:
            columns_to_drop += ["h1_samma_kusk", "h2_samma_kusk", "h3_samma_kusk"]
        X = X.drop(columns_to_drop, axis=1)

        return X
    
    # Ny Learn metod som tar hänsyn till att vi kan ha CayBoost eller XGBoost
    def learn(self, X_, y=None, X_test_=None, y_test=None, params=None, iterations=ITERATIONS, save=True, verbose=False):
        assert X_ is not None, 'X är None'
        if self.name.startswith('cat'):
            model_name = 'catboost'
        elif self.name.startswith('xgb'):
            model_name = 'xgboost'
        else:
            raise Exception('unknown model type')
        
        print('Learning', self.name, 'with', model_name)
        X = X_.copy()
        X_test = None
        if X_test_ is not None:
            X_test = X_test_.copy()

        assert 'streck' in list(X_.columns), 'streck saknas i learn X'
        if params is None:
            # Läs in parametrar från fil
            with open(self.pref+'optimera/params_'+self.name+'.json', 'rb') as f:
                params = json.load(f)
                params = params['params']

        iterations = params['iterations'] if 'iterations' in params else iterations
        params.pop('iterations')  # Ta bort iterations från params

        X = self.prepare_for_model(X_)
        if not self.streck:
            X.drop('streck', axis=1, inplace=True)

        if model_name == 'catboost':
            X, cat_features = prepare_for_catboost(X, verbose=verbose)
            assert X[cat_features].isnull().sum().sum(
            ) == 0, 'there are NaN values in cat_features'
            model = CatBoostClassifier(**params,
                                    iterations=iterations,
                                    loss_function='Logloss', eval_metric='AUC', verbose=verbose)
        elif model_name == 'xgboost':
            X, cat_features = prepare_for_xgboost(X, ohe=True, verbose=verbose, pref=self.pref)
            model = xgb.XGBClassifier(**params,
                                    iterations=iterations,
                                    early_stopping_rounds=Typ.EARLY_STOPPING_ROUNDS,
                                    loss_function='Logloss', eval_metric='AUC', verbose=verbose)
        else:
            raise Exception('unknown model type')    

        X = remove_features(X, remove_mer=['datum', 'avd', 'startnr'])

        if X_test is None or y_test is None:
            model.fit(X, y)
        else:
            X_test = self.prepare_for_model(X_test, verbose=verbose)
            if not self.streck:
                X_test.drop('streck', axis=1, inplace=True)

            if model_name == 'catboost':
                X_test, _ = prepare_for_catboost(X_test, verbose=verbose)
            elif model_name == 'xgboost':
                X_test, cat_features = prepare_for_xgboost(X_test, ohe=True, verbose=verbose, pref=self.pref)
            else:
                raise Exception('unknown model type')    

            X_test = remove_features(X_test, remove_mer=['datum', 'avd', 'startnr'])

            # Get the list of column names that are present in X.columns but not in X_test.columns
            the_diff1 = list(set(X.columns.tolist()).difference(set(X_test.columns.tolist())))
            assert len(the_diff1) == 0, f'1. features in X and X_test not equal: {the_diff1} in X.columns'
            # Get the list of column names that are present in X_test.columns but not in X.columns
            the_diff2 = list(set(X_test.columns.tolist()).difference(set(X.columns.tolist())))
            assert len(the_diff2) == 0, f'2. features in X and X_test not equal: {the_diff2} in X_test.columns'
            
            if model_name == 'catboost':
                eval_pool = Pool(X_test, y_test, cat_features=cat_features)
                model.fit(X, y, cat_features=cat_features, eval_set=eval_pool,
                        use_best_model=True, early_stopping_rounds=Typ.EARLY_STOPPING_ROUNDS)
            elif model_name == 'xgboost':
                assert X.columns.tolist() == X_test.columns.tolist(), 'X and X_test have different columns'
                # Create DMatrices for the training and testing data
                train_pool = xgb.DMatrix(X, y, enable_categorical=True)
                eval_pool = xgb.DMatrix(X_test, y_test, enable_categorical=True)

                # Fit the model on the training data and evaluate on the testing data
                model.fit(train_pool, eval_set=eval_pool, use_best_model=True)


        if verbose:
            print('best score', model.best_score_)
            
        if save:
            self.save_model(model)
            
        return model

                
    def predict(self, X_,verbose=False,model=None):
        # X_ måste ha datum och avd
        assert 'streck' in list(X_.columns), f'streck saknas i predict X ({self.name})'
        if self.name.startswith('cat'):
            model_name = 'catboost'
        elif self.name.startswith('xgb'):
            model_name = 'xgboost'
        else:
            raise Exception('unknown model type')

        print('Learning', self.name, 'with', model_name)
        X = X_.copy()
        
        X = self.prepare_for_model(X_)
        
        assert 'streck' in list(X.columns), f'streck saknas efter prepare_for_model i predict X ({self.name})'
        
        if model==None:
            model = self.load_model()
    
        if not self.streck:
            print('drop streck')
            X.drop('streck', axis=1, inplace=True)
        
        if model_name == 'catboost':
            X, _ = prepare_for_catboost(X, verbose=verbose)
        elif model_name == 'xgboost':
            X,_ = prepare_for_xgboost(X,ohe=True, pred=True, pref=self.pref)
        else:
            raise Exception('unknown model type')    
    
        X = remove_features(X, remove_mer=['datum', 'avd', 'startnr'])
        # Get the list of column names that are present in model.feature_names_ but not in X.columns
        the_diff1 = list(set(model.feature_names_).difference(set(X.columns.tolist())))
        # Get the list of column names that are present in X.columns but not in the model.feature_names_
        the_diff2 = list(set(X.columns.tolist()).difference(set(model.feature_names_)))
        assert len(the_diff1) == 0, f'1. features in model and in X not equal {the_diff1} in model.feature_names_ when predict {self.name}'
        assert len(the_diff2) == 0, f'2. features in model and in X not equal {the_diff2} in X.columns when predict {self.name}'
        
        X = X[model.feature_names_]
        if verbose:
            print('predict '+self.name)
         # Create DMatrices for the training and testing data
     
        # fit_pool = xgb.DMatrix(X, enable_categorical=False)
        return model.predict_proba(X)[:, 1]
    
    # method that retruns all the self variables
    def get_params(self):
        return {k:v for k,v in self.__dict__.items() if k[0]!='_'}  