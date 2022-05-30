import pandas as pd
import numpy as np
import pickle

pref='../'


class v75():
    def __init__(self, filnamn='all_data.csv', pref=''):
        self.pref=pref
        self.filnamn = pref+filnamn
        self.df = self.load_df()
        self.work_df = self.df.copy()
        
    def rensa_saknade_avd(self):
        saknas = ['2015-08-15', '2016-08-13', '2017-08-12']
        self.work_df = self.work_dfdf[~self.work_df.datum.isin(saknas)]
        return self.work_df
    
    def concat(self, ny_df):
        features = list(self.df.columns)
        assert set(features) == set(list(ny_df.columns)), 'Features in ny_df is not the same as in self.df'
        assert features == list(ny_df.columns), 'Features in ny_df and self.df are not equal'
        
        self.df = pd.concat([self.df, ny_df], axis=0)
        self.work_df = self.df.copy()
        return self.df
    
    def load_df(self):
        self.df = pd.read_csv(self.filnamn)
        return self.df
    
    def save_df(self):
        self.df.to_csv(self.filnamn, index=False)
        
    def remove_features(self, remove=['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
                'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'],remove_mer=[]):
        if remove:
            self.work_df.drop(remove, axis=1, inplace=True)
    
        if remove_mer: 
            self.work_df.drop(remove_mer, axis=1, inplace=True)

        return self.work_df

    