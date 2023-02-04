import logging 
import pandas as pd
import numpy as np
import pickle
import json
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import skapa_modeller as mod

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

def remove_features(df_, remove_mer=[]):
    df = df_.copy()
    print(type(df))
    if 'vodds' in df.columns:
        df.drop(['vodds', 'podds', 'bins', 'h1_dat',
                'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'], axis=1, inplace=True)
    if remove_mer and 'avd' in df.columns:
        df.drop(remove_mer, axis=1, inplace=True)

    return df

# remove NaN for cat_features in X and return (X, cat_features)
# ta bort alla features som inte används innan call


def prepare_for_catboost(X_, verbose=False):
    X = X_.copy()

    return X


def _catboost_encode(X_, y, columns):   # Används av XGBoost
    """ catboost encode måste ha y i work_df """
    from category_encoders import CatBoostEncoder
    X = X_.copy()
    # kopiera häst och kusk till nya kolumner för att spara orginalvärden
    X['häst_namn'] = X['häst'].copy()
    X['kusk_namn'] = X['kusk'].copy()
    encoder = CatBoostEncoder(cols=columns).fit(X[columns], y)

    X[columns] = encoder.transform(X[columns])

    print('CatBoost encoding done')

    return X, encoder


def prepare_for_xgboost(X_, y=None, cat_features=None, encoder=None, pred=False, verbose=False, pref=''):
    if encoder is not None:
        assert y is None and cat_features is None, 'encoder is not None, then y and cat_features must be None'
    else:
        assert y is not None and cat_features is not None, 'encoder is None, then y and cat_features must be set'

    X = X_.copy()
    bool_features = list(X.select_dtypes(include=['bool']).columns)
    X[bool_features] = X[bool_features].astype('int')

    if encoder is not None:
        # print('THE ENCODER',encoder)
        ENC = encoder
        cat_features_ = encoder.get_feature_names()
        X['häst_namn'] = X['häst'].copy()
        X['kusk_namn'] = X['kusk'].copy()
        X[cat_features_] = encoder.transform(X[cat_features_])
    else:
        X, ENC = _catboost_encode(X, y, cat_features)

        # save encoder
        with open(pref + 'xgb_encoder.pkl', 'wb') as f:
            pickle.dump(ENC, f)
            
    assert 'streck' in X.columns, 'streck not in X.columns'
    return X, ENC


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
        # int  - inkludera n features med bästa motståndare (streck)
        self.motst_ant = motst_ant
        # bool - ovanstående med diff (streck) istf fasta värden
        self.motst_diff = motst_diff
        self.streck = streck            # bool - inkludera streck som feature
        print('streck:', self.streck, 'i init för', self.name)
        self.rel_kr = True              # bool - skapa kol med relativt kr gentemot motståndarna
        # bool - skapa kol med relativ rank gentemot motståndarna
        self.rel_rank = True
        # bool - skapa kol med streck avstånd gentemot motståndarna
        self.streck_avst = True
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

    def prepare_for_model(self, X_, verbose=False):
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
            columns_to_drop += ["h1_samma_bana",
                                "h2_samma_bana", "h3_samma_bana"]
        if not self.hx_sammam_kusk:
            columns_to_drop += ["h1_samma_kusk",
                                "h2_samma_kusk", "h3_samma_kusk"]
        X = X.drop(columns_to_drop, axis=1)

        return X

    # Ny Learn metod som tar hänsyn till att vi kan ha CatBoost eller XGBoost
    def learn(self, X_, y=None, X_test_=None, y_test=None, params=None, use_L2_features_=None, iterations=ITERATIONS, save=True, verbose=False):
        logging.info(f'Typ: startar learn() för {self.name}')
        assert X_ is not None, 'X skall inte vara None'
        assert 'streck' in list(X_.columns), 'streck saknas i learn X'
        use_L2_features = None
        if use_L2_features_:
            assert 'streck' in use_L2_features_, 'streck saknas i use_L2_features_'
            use_L2_features = use_L2_features_.copy()
        assert verbose == False, 'verbose=True är inte implementerat i learn'
        assert 'datum' in X_.columns, 'datum saknas i learn X_ i början av learn'

        if self.name.startswith('cat'):
            model_type = 'catboost'
        elif self.name.startswith('xgb'):
            model_type = 'xgboost'
        else:
            raise Exception(f'unknown model {self.name}')

        print('Learning', self.name, 'with', model_type)
        
        X = X_.copy()
        X_test = None
        if X_test_ is not None:
            X_test = X_test_.copy()

        use_features, cat_features, num_features = mod.read_in_features()
        # läs in NUM_FEATURES.txt till num_features
        with open(self.pref+'NUM_FEATURES.txt', 'r', encoding='utf-8') as f:
            num_features = f.read().split()

        # läs in CAT_FEATURES.txt till cat_features
        with open(self.pref+'CAT_FEATURES.txt', 'r', encoding='utf-8') as f:
            cat_features = f.read().split()

        if use_L2_features is None:
            use_features = cat_features + num_features
            assert 'streck' in use_features, f'streck saknas i Learn: use_features ({self.name})'
        else:
            assert 'streck' in use_L2_features, f'streck saknas i Learn: use_L2_features ({self.name})'
            use_features = use_L2_features

        ENC = None

        if params is None:
            # Läs in parametrar från fil
            with open(self.pref+'optimera/params_'+self.name+'.json', 'rb') as f:
                params = json.load(f)
                params = params['params']

        # print('params:', params)
        # print('iterations:', iterations)
        if 'iterations' in params:
            iterations = params['iterations'] 
            params.pop('iterations')

        assert 'datum' in X.columns, 'datum saknas i learn X before prepare_for_model'
        X = self.prepare_for_model(X)

        if not self.streck:
            # X.drop('streck', axis=1, inplace=True)
            use_features.remove('streck')

        assert X[cat_features].isnull().sum().sum() == 0,'there are NaN values in cat_features'

        if model_type == 'catboost':
            X = prepare_for_catboost(X, verbose=verbose)
            model = CatBoostClassifier(**params,
                                       iterations=iterations,
                                       loss_function='Logloss', eval_metric='AUC', verbose=verbose)
        elif model_type == 'xgboost':
            X, ENC = prepare_for_xgboost(X, y, cat_features, encoder=ENC, verbose=verbose, pref=self.pref)

            # Kolla att alla kolumner är numeriska i use_features
            non_numeric_columns = [
                c for c in X[use_features].columns if X[c].dtype.name == 'object']
            assert len(non_numeric_columns) == 0, f'X innehåller non-numeric columns: {non_numeric_columns.columns}'

            model = xgb.XGBClassifier(**params,
                                      #   iterations=iterations,
                                      early_stopping_rounds=Typ.EARLY_STOPPING_ROUNDS if X_test is not None else None,
                                      objective='binary:logistic', eval_metric='auc')
        else:
            raise Exception('unknown model type')

        if self.name[-2:] == 'L1':
            assert len([col for col in X.columns if 'proba' in col]) == 0, f'X innehåller proba-kolumner för L1-modell {self.name}'

        if X_test is None or y_test is None:
            if model_type == 'catboost':
                model.fit(X[use_features], y, cat_features=cat_features)
            elif model_type == 'xgboost':
                model.fit(X[use_features], y, verbose=0)
            else:
                raise Exception('unknown model type')

        else:
            assert 'datum' in X_test.columns, 'datum saknas i learn X_test'
            X_test = self.prepare_for_model(X_test)

            if model_type == 'catboost':
                X_test = prepare_for_catboost(X_test, verbose=verbose)
            elif model_type == 'xgboost':
                X_test, _ = prepare_for_xgboost(X_test, encoder=ENC, verbose=verbose, pref=self.pref)
            else:
                raise Exception('unknown model type')

            if model_type == 'catboost':
                eval_pool = Pool(X_test[use_features],
                                 y_test, cat_features=cat_features)
                model.fit(X[use_features], y, cat_features=cat_features, eval_set=eval_pool,
                          use_best_model=True, early_stopping_rounds=Typ.EARLY_STOPPING_ROUNDS)
            elif model_type == 'xgboost':
                # Kolla att alla kolumner är numeriska i use_features
                non_numeric_columns = [c for c in X_test[use_features].columns if X_test[c].dtype.name == 'object']
                assert len(non_numeric_columns) == 0, f'X innehåller non-numeric columns: {non_numeric_columns.columns}'

                if set(X.columns.tolist()) != set(X_test.columns.tolist()):
                    assert False, f'fit av {model.name}: X and X_test have different columns \nX     : {X.columns} \nX_test: {X_test.columns}'

                # Fit the model on the training data and evaluate on the testing data
                model.fit(X[use_features], y, eval_set=[(X_test[use_features], y_test)], verbose=0)
            else:
                raise Exception('unknown model type')

        if verbose:
            print('best score', model.best_score_)

        if save:
            if self.name[-2:] == 'L2':
                assert len([col for col in use_features if 'proba' in col]) == 4, f' fel antal proba-kolumner för L2-modell {self.name}'
                
            self.save_model(model)
            
            # Save the list of column names to a text file
            with open(self.pref+'modeller/'+self.name+'_columns.txt', "w", encoding="utf-8") as f:
                for col in X[use_features].columns.tolist():
                    f.write(col + '\n')
                

        return model

    def predict(self, X_, use_features_, verbose=False, model=None):
        """ proba_predict med model som finns sparad eller som skickas in
        Args:
            X_ (DataFrame): med "alla" v75-kolumner
            use_features_ (List): De features som ska användas för prediktionen
            verbose (bool, optional): Om extra infp skall skrivas ut. Defaults to False.
            model (tränad model, optional): Om denna finns så skall den användas annars gör vi Load på sparad. Defaults to None.

        Raises:
            Exception: Unknown model type

        Returns:
            Series: Med första kolumnen i alla proba_predict
        """
        logging.info(f'Typ: startar predict med {self.name}')
        X = X_.copy(deep=True)
        if self.name[-2:] == 'L1':
            assert len([col for col in X.columns if 'proba' in col]) == 0, f'X innehåller proba-kolumner för L1-modell {self.name}'
            assert len([col for col in use_features_ if 'proba' in col]) == 0, f'use_features_ innehåller proba-kolumner för L1-modell {self.name}'
        elif self.name[-2:] == 'L2':
            assert len([col for col in use_features_ if 'proba_' in col]) == 4, f'use_features_ har fel antal proba-kolumner till L2-modell {self.name}'
            # obs att det är 4 ifyllda L1-proba-kolumner plus 4 tomma L2-proba-kolumner i X från början, som fylls i efterhand
            assert len([col for col in X.columns if 'proba_' in col]) == 8, f'X fel antal proba-kolumner för L2-modell {self.name} {X.columns}'
            
        # X måste ha datum och avd
        assert 'datum' in X.columns, f'datum saknas i predict: X ({self.name})'
        assert 'avd' in X.columns, f'avd saknas i predict: X ({self.name})'
        assert 'streck' in X.columns, f'streck saknas i predict: X ({self.name})'
        assert 'streck' in use_features_, f'streck saknas i predict: use_features ({self.name})'
        
        if self.name.startswith('cat'):
            model_name = 'catboost'
        elif self.name.startswith('xgb'):
            model_name = 'xgboost'
        else:
            raise Exception('unknown model type')

        print('Predicting', self.name, 'with', model_name)
        
        use_features = use_features_.copy()

        X = self.prepare_for_model(X_)
        assert 'streck' in X.columns, f'streck saknas efter prepare_for_model i predict X ({self.name})'

        if model == None:
            model = self.load_model()

        if not self.streck:
            print('drop streck')
            use_features.remove('streck')

        if model_name == 'catboost':
            logging.info(f'Typ: {self.name} - catboost')
            X = prepare_for_catboost(X, verbose=verbose)
        elif model_name == 'xgboost':
            logging.info(f'Typ: {self.name} - xgboost')
            # xgb_encoder till ENC
            with open(self.pref+'xgb_encoder.pkl', 'rb') as f:
                ENC = pickle.load(f)

            X, _ = prepare_for_xgboost(X, encoder=ENC, pred=True, pref=self.pref)
            missing_items2 = [item for item in use_features if item not in model.get_booster().feature_names]
            assert len(missing_items2) == 0, f"The following items in 'use_features' are not found in modellens features': {missing_items2}"
        else:
            raise Exception('unknown model type')

        missing_items = [item for item in use_features if item not in X.columns.tolist()]
        assert len(missing_items) == 0, f"The following items in 'use_features' are not found in 'X.columns': {missing_items}"
        
        if verbose:
            print('predict '+self.name, 'with streck', self.streck, "found streck in use_features",
                  'streck' in use_features, "\nuse_features", use_features)

        assert len(set(X.columns.tolist())) == len(X.columns.tolist()), f'X.columns has doubles: {X.columns.tolist()}'
        assert len(set(use_features)) == len(use_features), f'use_features has doubles: {use_features}'
        
        logging.info(f'Typ: predict med {self.name} ')
  
        return model.predict_proba(X[use_features])[:, 1]

    # method that retruns all the self variables
    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if k[0] != '_'}
