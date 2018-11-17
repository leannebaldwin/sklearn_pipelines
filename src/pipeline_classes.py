import pandas as pd 
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin


class Featurizer(BaseEstimator, TransformerMixin):
    """Clean incoming df to fit into model"""
    
    def __init__(self, cols=None):
        """INPUT: a data_type_dict to determine which columns are 
                  continueous and categorical
                  an optional cols list of columns to select"""
        if cols==None:
            self.cols = ['body_length', 
                            'channels', 
                            'country', 
                            'currency', 
                            'description', 
                            'email_domain', 
                            'event_created', 
                            'event_end',
                            'event_published', 
                            'event_start', 
                            'fb_published', 
                            'has_analytics',
                            'has_header', 
                            'has_logo', 
                            'listed', 
                            'name', 
                            'name_length', 
                            'object_id',
                            'org_desc', 
                            'org_facebook', 
                            'org_name', 
                            'org_twitter', 
                            'payee_name',
                            'payout_type', 
                            'previous_payouts', 
                            'sale_duration', 
                            'show_map',
                            'ticket_types', 
                            'user_age', 
                            'user_created', 
                            'user_type',
                            'venue_address', 
                            'venue_country', 
                            'venue_latitude', 
                            'venue_longitude',
                            'venue_name', 
                            'venue_state']
        else:
            self.cols = cols
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """tranform and clean incoming training or test"""
        df = X.copy()
        df = df.loc[:,self.cols]
        df['event_duration'] = df['event_end']-df['event_start']
        df['has_payee_name'] = df['payee_name'].apply(self.is_empty)
        df['has_header'] = df['has_header'].fillna(0)
        df['has_previous_payouts'] = df['previous_payouts'].apply(self.is_empty)
        df['has_payout_type'] = df['payout_type'].apply(self.is_empty)
        df['has_facebook'] = df['org_facebook'].apply(self.is_not_zero)
        df['has_twitter'] = df['org_twitter'].apply(self.is_not_zero)
        df['country'] = df['country'].apply(self.replace_empty_with_none)
        drop_list = ['description',
                    'event_created',
                    'event_end',
                    'event_published',
                    'event_start',
                    'name',
                    'object_id',
                    'payee_name',
                    'ticket_types',
                    'user_created',
                    'venue_address',
                    'venue_country',
                    'venue_longitude',
                    'venue_latitude',
                    'venue_name',
                    'venue_state',
                    'previous_payouts',
                    'email_domain',
                    'org_name',
                    'org_twitter',
                    'org_facebook',
                    'org_desc']
        return df.drop(drop_list, axis=1)

    @staticmethod  
    def is_not_zero(x):
        if x == 0:
            return 0
        return 1

    @staticmethod
    def is_empty(x):
        if not x:
            return 0
        return 1

    @staticmethod
    def max_cost(row):
        """Find the hightest ticket price from a row in df['ticket_types']
        input: [{'event_id': 527017,
                'cost': 25.0,
                'availability': 1,
                'quantity_total': 800,
                'quantity_sold': 0},
                {'event_id': 527017,
                'cost': 50.0,
                'availability': 1,
                'quantity_total': 100,
                'quantity_sold': 0},
                {'event_id': 527017,
                'cost': 550.0,
                'availability': 1,
                'quantity_total': 20,
                'quantity_sold': 0}]
        output: 550.0 """
        maximum = 0
        for item in row:
            if item['cost'] >= maximum:
                maximum = item['cost']
        return maximum
    
    @staticmethod
    def replace_empty_with_none(x):
        if not x:
            return 'None'
        else: 
            return x


class Imputer(BaseEstimator, TransformerMixin):
    """Impute either mode or mean into cleaned and dummied data"""
    def __init__(self, cols_dict=None):
        if cols_dict==None:
            self.cols_dict = {'body_length':'cont', 
                                'channels':'cat', 
                                'country':'cat', 
                                'currency':'cat', 
                                'fb_published':'cat', 
                                'has_analytics':'cat', 
                                'has_header':'cat', 
                                'has_logo':'cat', 
                                'listed':'cat',
                                'name_length':'cont', 
                                'payout_type':'cat', 
                                'sale_duration':'cont', 
                                'show_map':'cat', 
                                'user_age':'cont',
                                'user_type':'cat', 
                                'event_duration':'cont', 
                                'has_payee_name':'cat', 
                                'has_previous_payouts':'cat',
                                'has_payout_type':'cat', 
                                'has_facebook':'cat', 
                                'has_twitter':'cat'}
        else:
            self.cols_dict = cols_dict

    def fit(self, X, y=None):
        """save the values to impute into each column"""
        df = X
        self.averages = {}
        for col, val in self.cols_dict.items():
            if val=='cat':
                self.averages[col] = 'None'
            if val=='cont':
                self.averages[col] = df.loc[:,col].mean()
        return self

    def transform(self, X):
        """for each column in df, impute the columns mean or mode if nan"""
        df = X.copy()
        for col in df.columns:
            df[col] = df[col].fillna(self.averages[col])
        return df

    
class Dummifier(BaseEstimator, TransformerMixin):
    """Dummify certain columns in a DataFrame"""
    def __init__(self, cols_to_dummy=None):
        if cols_to_dummy==None:
            self.cols_to_dummy = ['channels', 
                                  'country', 
                                  'currency', 
                                  'fb_published', 
                                  'has_analytics', 
                                  'has_header', 
                                  'has_logo', 
                                  'listed',
                                  'payout_type', 
                                  'show_map', 
                                  'user_type', 
                                  'has_payee_name', 
                                  'has_previous_payouts',
                                  'has_payout_type', 
                                  'has_facebook', 
                                  'has_twitter']
        else:
            self.cols_to_dummy = cols_to_dummy 
        self.unique_items = {}

    def fit(self, X, y=None):
        df = X
        for col in self.cols_to_dummy:
            self.unique_items[col] = df[col].unique()
        return self
            
    def transform(self, X):
        df = X.copy()
        dummy_df = pd.DataFrame()
        for col in self.cols_to_dummy:
            columns = self.unique_items[col]
            for item in columns:
                if item==None:
                    continue
                dummy_df[f'{col}_{item}'] = df[col]==item
            dummy_df = dummy_df.iloc[:,:-1]    
        df = df.drop(self.cols_to_dummy, axis=1)
        dummy_df = dummy_df.astype(int)
        df = pd.concat([df, dummy_df], axis=1)
        return df

    
class Standardizer(BaseEstimator, TransformerMixin):
    """Standardize continuous columns"""
    def __init__(self, continuous_cols=None):
        if continuous_cols==None:
            self.continous_cols = ['body_length', 'name_length', 
                                   'sale_duration', 'user_age', 
                                   'event_duration']
        else:
            self.continous_cols = continuous_cols

    def fit(self, X, y=None):
        df = X
        self.means = {}
        self.standard_devs = {}
        for col in self.continous_cols:
            self.means[col] = df[col].mean()
            self.standard_devs[col] = df[col].std()
        return self
    
    def transform(self, X):
        df = X.copy()
        for col in self.continous_cols:
            df[col] = (df[col]-self.means[col])/self.standard_devs[col]
        return df

        
