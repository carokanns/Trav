"""_summary_
    En class för att hantera all_data.csv
    Skall standardisera förberedelser inför ML-körningar 
"""
import pandas as pd
import numpy as np

class v75():
    def __init__(self, filnamn='all_data.csv', pref=''):
        self.pref=pref
        self.filnamn = pref+filnamn
        print(self.filnamn)
        self.df = self.load_df()   # uppdateras enbart av ny data
        self.work_df = self.df.copy()   # arbetskopia att köra all ML mot

        
    #################### Ta bort oanvändbara omgångar #######################################
    def _rensa_saknade_avd(self):
        """ Dessa omgångar saknar vissa avdelningar och kan inte användas """
        
        saknas = ['2015-08-15', '2016-08-13', '2017-08-12']
        self.work_df = self.work_df[~self.work_df.datum.isin(saknas)]
    
    #################### Konkatenera in ny data ############################################
    def concat(self, ny_df, save=True):
        """ efter web scraping kan ny data läggas till """
        features = list(self.df.columns)
        assert set(features) == set(list(ny_df.columns)), 'Features in ny_df is not the same as in self.df'
        assert features == list(ny_df.columns), 'Features in ny_df and self.df are not equal'
        
        self.df = pd.concat([self.df, ny_df], axis=0)
        self.df.drop_duplicates(subset=['datum', 'avd', 'häst'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.work_df = self.df.copy()
        
        if save==True:
            self.save_df()
            
        return self.df
    
    #################### Load och save #####################################################
    def load_df(self):
        print('Loading dataframe from the file:', self.filnamn)
        self.df = pd.read_csv(self.filnamn)
        return self.df
    
    def save_df(self):
        self.df.to_csv(self.filnamn, index=False)
        
    #################### Features som inte används ###########################################
    def _remove_features(self, remove=['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
                'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'],remove_mer=[]):
        """ rensa bort features som inte ska användas """
        
        if remove:
            self.work_df.drop(remove, axis=1, inplace=True)
    
        if remove_mer: 
            self.work_df.drop(remove_mer, axis=1, inplace=True)

        return self.work_df

    def förbered_data(self):
        """ En komplett förberedelse innan ML
        Returns:
            self.work_df: Färdig df att användas för ML
        """
        self.work_df = self.df.copy()
        # rensa omgångar som saknar avdelningar
        self._rensa_saknade_avd()
        
        # ta bort nummer från travbana i history (i.e Åby-1 -> Åby, etc)
        self.work_df.loc[:,'h1_bana'] = self.work_df.h1_bana.str.split('-').str[0]
        self.work_df.loc[:,'h2_bana'] = self.work_df.h2_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h3_bana'] = self.work_df.h3_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h4_bana'] = self.work_df.h4_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h5_bana'] = self.work_df.h5_bana.str.split('-').str[0]

        # lower case för häst, bana, kusk and hx_bana
        for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
            self.work_df.loc[:, f] = self.work_df[f].str.lower()

        self._remove_features()
        
        y = (self.work_df.plac==1) * 1
        
        return self.work_df.drop(['plac'], axis=1), y
    
    def get_df(self):
        return self.df
    
    def get_work_df(self):  # returnerar arbetskopia
        return self.work_df
