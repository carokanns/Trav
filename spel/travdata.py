"""_summary_
    En class för att hantera all_data.csv
    Standardiserar förberedelser inför ML-körningar 
"""
import pandas as pd
import numpy as np

class v75():
    def __init__(self, filnamn='all_data.csv', pref=''):
        """ init - två dataset skapas, df (originalet) och work_df (arbetskopian) """
        self.pref=pref
        self.filnamn = pref+filnamn
        print(self.filnamn)
        self.df = self.load_df()        # kan uppdateras enbart med concat av ny data
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
        """ ladda in all_data.csv """
        print('Loading dataframe from the file:', self.filnamn)
        self.df = pd.read_csv(self.filnamn)
        return self.df
    
    def save_df(self):
        """ sparar df (ej working_df) till all_data.csv """
        self.df.to_csv(self.filnamn, index=False)
        
    #################### Handle missing #####################################################    
    def _handle_missing_num(self):
        """ Fyll i saknade numeriska värden med 0 """
        categoricals = self.work_df.select_dtypes(include=['object']).columns
        # check if column y is in the dataframe
        
        assert 'y' in self.work_df.columns, 'y is not in the work_df'
        numericals = self.work_df.drop('y', axis=1).select_dtypes(exclude=['object']).columns
        self.work_df[numericals] = self.work_df[numericals].fillna(0)
        
    def _handle_missing_cat(self):
        """ Fyll i saknade kategoriska värden med 'missing' """
        categoricals = self.work_df.select_dtypes(include=['object']).columns
        self.work_df[categoricals] = self.work_df[categoricals].fillna('missing')
        # return self.work_df    
        
    #################### Handle high cardinality ############################################
    def _handle_high_cardinality(self, column, threshold=0.75, max=10):
        """ Reducera cardinality baserat på frekvens """
        threshold_value=int(threshold*len(self.work_df[column].unique()))
        
        categories_list=[]
        s=0
        #Create a counter dictionary of the form unique_value: frequency
        counts=self.work_df[column].value_counts()
        # print(counts)
        values = list(counts.index)

        #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
        for e,i in enumerate(counts):
            if max and e >= max:
                break
            
            #Add the frequency to the global sum
            s+=i
            #Append the category name to the list
            categories_list.append(values[e])
            #Check if the global sum has reached the threshold value, if so break the loop
            if s>=threshold_value:
                break
        #Append the category Other to the list
        categories_list.append('Other')

        #Replace all instances not in our new categories by Other  
        new_column=self.work_df[column].apply(lambda x: x if x in categories_list else 'Other')
        self.work_df[column]=new_column

        # return self.work_df
        
        
    #################### Features som inte används ##########################################
    def _remove_features(self, remove=['startnr', 'vodds', 'podds', 'bins', 'h1_dat',
                'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'],remove_mer=[]):
        """ rensa bort features som inte ska användas """
        
        if remove:
            self.work_df.drop(remove, axis=1, inplace=True)
    
        if remove_mer: 
            self.work_df.drop(remove_mer, axis=1, inplace=True)

        return self.work_df

    def förbered_data(self, missing_num=True, missing_cat=True, cardinality_list=[]):
        """ En komplett förberedelse innan ML
        Returns:
            self.work_df: Färdig df att användas för ML
        """
        self.work_df = self.df.copy()
        # rensa omgångar som saknar avdelningar
        self._rensa_saknade_avd()
        
        # ta bort nummer från travbana i history (i.e Åby-1 -> Åby, etc)
        self.work_df.loc[:, 'h1_bana'] = self.work_df.h1_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h2_bana'] = self.work_df.h2_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h3_bana'] = self.work_df.h3_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h4_bana'] = self.work_df.h4_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h5_bana'] = self.work_df.h5_bana.str.split('-').str[0]

        # lower case för häst, bana, kusk and hx_bana
        for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
            self.work_df.loc[:, f] = self.work_df[f].str.lower()

        self._remove_features()
        
        self.work_df['y'] = (self.work_df.plac==1) * 1
        self.work_df = self.work_df.drop(['plac'], axis=1)

        if missing_cat:
            self._handle_missing_cat()
            
        if missing_num:
            self._handle_missing_num()    
            
        if cardinality_list:
            for col in cardinality_list:
                assert col in self.work_df.columns, f'{col} is not in work_df'
                self._handle_high_cardinality(col)
            
        return self.work_df
    
    def train_test_split(self, train_size=0.8):
        """ Splits data into train and test set based on train_size """
        datumar = self.work_df.datum.unique()
        train_datum = datumar[:int(train_size*len(datumar))]
        test_datum = datumar[int(train_size*len(datumar)):]
        
        train = self.work_df[self.work_df.datum.isin(train_datum)].copy()
        test = self.work_df[self.work_df.datum.isin(test_datum)].copy()
        # rename plac in test and train to y
        train = train.rename(columns={'plac':'y'})
        test = test.rename(columns={'plac':'y'}, inplace=False)
        return train, test
    
    def get_df(self):
        """ returnerar df (original)"""
        return self.df
    
    def get_work_df(self):  
        """ returnerar arbetskopian """
        return self.work_df
