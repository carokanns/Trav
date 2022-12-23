"""_summary_
    En class för att hantera all_data.csv
    Standardiserar förberedelser inför ML-körningar 
"""
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from IPython.display import display


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
    def concat(self, ny_df, update_work=True, save=True):
        """ efter web scraping kan ny data läggas till """
        features = list(self.df.columns)
        assert set(features) == set(list(ny_df.columns)), 'Features in ny_df is not the same as in self.df'
        assert features == list(ny_df.columns), 'Features in ny_df and self.df are not equal'
        
        self.df = pd.concat([self.df, ny_df], axis=0)
        self.df.drop_duplicates(subset=['datum', 'avd', 'häst'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        if update_work==True:
            self.work_df = self.df.copy()
        
        if save==True:
            self.save_df()
        
    
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
        
        assert 'y' in self.work_df.columns, 'y is missing in the work_df'
        numericals = self.work_df.drop('y', axis=1).select_dtypes(exclude=['object']).columns
        self.work_df[numericals] = self.work_df[numericals].fillna(0)
        
    def _handle_missing_cat(self):
        """ Fyll i saknade kategoriska värden med 'missing' """
        categoricals = self.work_df.select_dtypes(include=['object']).columns
        self.work_df[categoricals] = self.work_df[categoricals].fillna('missing')
    
        
    #################### Handle high cardinality ############################################
    def _handle_high_cardinality(self, column, threshold=0.33, max=10):
        """ Reducera cardinality baserat på frekvens """
    
        threshold_value=int(threshold*len(self.work_df[column]))
        # print('Threshold:value for', column, 'is', threshold_value)
        categories_list=[]
        s=0
        #Create a counter dictionary of the form unique_value: frequency
        counts=self.work_df[column].value_counts()
        # print('5 counts\n', counts[0:5])
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
                # print(f'{column} s = {s} more than threshold_value={threshold_value}. {e} categories')
                break
        #Append the category Other to the list
        categories_list.append('Other')

        #Replace all instances not in our new categories by Other  
        new_column=self.work_df[column].apply(lambda x: x if x in categories_list else 'Other')
        self.work_df[column]=new_column

    
    def _target_encode(self, columns):
        """ target encode måste y i work_df """
        y = self.work_df.pop('y')
        encoder = TargetEncoder(cols=columns, min_samples_leaf=20, smoothing=10).fit(self.work_df, y)
        
        self.work_df= encoder.transform(self.work_df)
        
        self.work_df['y'] = y
        
        print('Target encoding done')
        display(self.work_df.head())
        return encoder  
        
    #################### Features som inte används ##########################################
    def _remove_features(self, remove=[ 'vodds', 'podds', 'bins', 'h1_dat',
                'h2_dat', 'h3_dat', 'h4_dat', 'h5_dat'],remove_mer=[]):
        """ rensa bort features som inte ska användas """
        
        if remove:
            self.work_df.drop(remove, axis=1, inplace=True)
        if remove_mer: 
            self.work_df.drop(remove_mer, axis=1, inplace=True)

        return self.work_df

    def förbered_data(self, extra=False,        # Add extra features
                      missing_num=True,         # Handle missing numerics
                      missing_cat=True,         # Handle missing categoricals
                      cardinality_list=[],      # Handle high cardinality
                      target_encode_list=[],    # Use this list for Target encoding (creating encoder)
                      encoder=None,             # Use this encoder for Target encoding (transform)  
                      remove=True,              # Remove default features
                      remove_mer=[]):           # Remove more features not default
        """ En komplett förberedelse innan ML
        Returns:
            self.work_df: Färdig df att användas för ML
        """
        self.work_df = self.df.copy()
        # rensa omgångar som saknar avdelningar
        self._rensa_saknade_avd()
        
        # set datum to datetime
        self.work_df['datum'] = pd.to_datetime(self.work_df['datum']).dt.date
        # self.work_df['datum'] = self.work_df['datum'].dt.date
        # ta bort suffix-nummer från travbana i history (i.e Åby-1 -> Åby, etc)
        self.work_df.loc[:, 'h1_bana'] = self.work_df.h1_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h2_bana'] = self.work_df.h2_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h3_bana'] = self.work_df.h3_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h4_bana'] = self.work_df.h4_bana.str.split('-').str[0]
        self.work_df.loc[:, 'h5_bana'] = self.work_df.h5_bana.str.split('-').str[0]

        # lower case för häst, bana, kusk and hx_bana
        for f in ['häst', 'bana', 'kusk', 'h1_kusk', 'h2_kusk', 'h3_kusk', 'h4_kusk', 'h5_kusk', 'h1_bana', 'h2_bana', 'h3_bana', 'h4_bana', 'h5_bana']:
            self.work_df.loc[:, f] = self.work_df[f].str.lower()

        if remove:
            self._remove_features(remove_mer=remove_mer)
        
        if 'plac' in self.work_df.columns.to_list():    
            print('plac finns i df')
            self.work_df['y'] = (self.work_df.plac==1) * 1
            self.work_df = self.work_df.drop(['plac'], axis=1)
        else:
            print('No plac in df')
            
        if missing_cat:
            self._handle_missing_cat()
            
        if missing_num:
            self._handle_missing_num()    
            
        if cardinality_list:
            for col in cardinality_list:
                assert col in self.work_df.columns, f'cardinality_list: {col} is not in work_df'
                self._handle_high_cardinality(col)
                
        if extra:
            _ = self.test_lägg_till_kolumner()
            
        if len(target_encode_list)>0 and encoder:
            display("WARNING: Don't give both encoder and target_encode_list - the list ignored" )    
            
        if encoder:
            print('Using existing encoder for encoding')
            # y = self.work_df.pop('y')
            cols = encoder.get_feature_names()
            self.work_df = encoder.transform(self.work_df[cols])          
            # self.work_df['y'] = y      
            
        elif len(target_encode_list) > 0:
            print('Creating new encoder')
            for col in target_encode_list:
                assert col in self.work_df.columns, f'target_encode_list: {col} is not in work_df'
            
            encoder = self._target_encode(target_encode_list)

        return self.work_df, encoder
    
    def train_test_split(self, train_size=0.7):
        """ Splits data into train and test sets (time dependent) based on train_size """
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
    
########################################################################### 
################################ TEST #####################################
    #  a - ✔️ plats(streck)/ant_hästar_i_avd (antal startande hästar i avd)
    #  b - ❌ pris / tot_pris_i_avd - funkar ju inte. Alla har samma pris per avd!
    #  c - ✔️ kr / tot_kr_i_avd - ersätt kr 
    #  d - ✔️ Avstånd till ettan (streck. Kanske proba för meta_model?) 
    #  e - ✔️ hx_bana samma som bana 
    #  f - ✔️ hx_kusk samma som kusk 
    
    def test_lägg_till_kolumner(self):
        """
        Testar nya kolumner b-f ovan
        Körs typiskt efter förbered_data()
        """
        
        ##### kr/total_kr_avd ******
        sum_kr = self.work_df.groupby(['datum', 'avd']).kr.transform(lambda x: x.sum())
        self.work_df['rel_kr'] = self.work_df.kr/sum_kr
        self.work_df.drop(['kr'], axis=1, inplace=True)
        
        ##### avst till ettan (streck) ******
        self.work_df['max_streck'] = self.work_df.groupby(['datum', 'avd']).streck.transform(lambda x: x.max())
        self.work_df['streck_avst'] = self.work_df.max_streck - self.work_df.streck
        self.work_df.drop(['max_streck'], axis=1, inplace=True)
        
        ##### ranking per avd / ant_startande ******
        rank_per_avd = self.work_df.groupby(['datum', 'avd'])['streck'].rank(ascending=False, method='dense')
        count_per_avd = self.work_df.groupby(['datum', 'avd']).streck.transform(lambda x: x.count())
        self.work_df['rel_rank'] = rank_per_avd/count_per_avd
        
        ##### hx samma bana (h1-h3)
        self.work_df['h1_samma_bana'] = self.work_df.bana == self.work_df.h1_bana
        self.work_df['h2_samma_bana'] = self.work_df.bana == self.work_df.h2_bana
        self.work_df['h3_samma_bana'] = self.work_df.bana == self.work_df.h3_bana

        ##### hx samma kusk (h1-h3)
        self.work_df['h1_samma_kusk'] = self.work_df.kusk == self.work_df.h1_kusk
        self.work_df['h2_samma_kusk'] = self.work_df.kusk == self.work_df.h2_kusk
        self.work_df['h3_samma_kusk'] = self.work_df.kusk == self.work_df.h3_kusk

        return self.work_df

