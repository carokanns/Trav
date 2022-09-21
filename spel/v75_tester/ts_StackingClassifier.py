
#%%
##############################################################
###### En KLASS för Stacking Classifier för TIME SERIES ######
##############################################################

# import typ_copy as tp
# import travdata as td
import time
import concurrent.futures
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

pref = '../'


import pickle

from catboost import CatBoostClassifier, Pool

import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')

#%%

class ts_stacking():
    def __init__(self, estimators, final_estimator=None, passthrough=[], n_splits=5, random_state=2022, pref=''):
        """ - init - spcial för tscv stacking 
            - estimators är en lista med tuples (estimator_name, estimator)
            - final_estimator är den estimator som gör den sluliga prediktionen (metamodell), 
               default är RandomForestClassifier med 100 interna träd och min_samples_split=50
            - passthrough är en lista med kolumner utöver stack-predctions som ska kopieras vidare till stacken
             
            Lämplig workflow:
            1. ts_stacking = ts_stacking(estimators, final_estimator, passthrough)
            2. ts_stacking.fit_all(X, y) (inkl skapa_stack_data och fit_final_estimator)
            3. ts_stacking.predict_all(X)
        """
        self.pref = pref
        self.est_dict = {}
        for name, est in estimators:
            self.est_dict[name] = est
        if final_estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            final_estimator = RandomForestClassifier(n_estimators=100,min_samples_split=50, random_state=random_state)
        self.final_estimator=final_estimator
        self.passthrough = passthrough
        self.n_splits = n_splits
        
    #################### fit estimators #######################################

    def _fit_estimator(self, est_name, X, y):
        """ fit en estimator """
        self.est_dict[est_name].fit(X,y)
        
    def _fit_all_estimators(self, X, y):
        for k, v in self.est_dict.items():
            print(f'fitting {k}')
            self._fit_estimator(k, X, y)
            
   ###################### skapa stack #########################################
    def skapa_stack_data(self, X, y):
        """ skapa stacked input till final estimator """
        self.X_stacked = pd.DataFrame()
        self.y_stacked['y'] = y
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        # loop over all splits
        for train_index, test_index in tscv.split(X):  
            print(train_index[-1])
            if self.passthrough is None:
                X_new = pd.DataFrame()
                y_new = pd.Series()
            else:
                X_new = X.iloc[test_index].copy()
                y_new = y.iloc[test_index]
            
            self._fit_all_estimators(X.iloc[train_index], y.iloc[train_index])
            for k, v in self.est_dict.items():
                print(f'predicting {k}')
                X_new[k] = v.predict_proba(X.iloc[test_index])[:, 1]
                
            # concat to stacked dataframe
            self.X_stacked = pd.concat([self.X_stacked, X_new])
            self.y_stacked = pd.concat([self.y_stacked, y_new])
            
   #################### final estimator #######################################
    def fit_final_estimator(self):
        """ fit final estimator """
        self.final_estimator.fit(self.X_stacked, self.y_stacked)    
   #################### komplett fit ###########################################     
    def fit_all(self, X, y):
        """ fit all """
        self.skapa_stack_data(X, y)
        self.fit_final_estimator()
    #################### komplett predict #######################################     
   
    def predict_all(self, X):
       """ predict all """
       X_new = X.copy()
       for k, v in self.est_dict.items():
           print(f'predicting {k}')
           X_new[k] = v.predict_proba(X)[:, 1]
       return self.final_estimator.predict(X_new)
   
#%%


#%%
if __name__ == '__main__':
    pref = '../'
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.max_rows', 120)

    from catboost import CatBoostClassifier, Pool
    import concurrent.futures
    import time
    import sys
    sys.path.append(
        'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel')

    import typ_copy as tp
    import travdata as td

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf2 = RandomForestClassifier(n_estimators=1000, random_state=42)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_jobs=6)
    
    v75 = td.v75(pref=pref)
    
    df, _ = v75.förbered_data()  # num hanteras av catboost
    categoricals = df.select_dtypes(include=['object']).columns
    df,enc = v75.förbered_data(target_encode_list=categoricals) 
    
    df = v75.test_lägg_till_kolumner()
    ts_stack = ts_stacking([('rf', rf), ('rf2', rf2), ('knn', knn)])
    ts_stack.skapa_stack_data(df.drop('y', axis=1), df['y'])

# %% 
