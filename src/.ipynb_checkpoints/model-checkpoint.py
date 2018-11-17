import pandas as pd
import numpy as np
import json
from src.pipeline_classes import Featurizer, Imputer, Standardizer, Dummifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import pickle

def load(file):
    """Return X and y from training data, no manipuation"""
    data = pd.read_json(file)
    data['fraud'] = data['acct_type'].str.contains('fraud')
    cols = ['body_length', 
            'channels', 
            'country', 
            'currency', 
            'delivery_method',
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
    return data[cols], data['fraud']

def get_training_data():
    """Automatically get data, clean, and featurize data"""
    X, y = load('../data/data.json')
    X_cleaned = Featurizer().transform(X)
    imputer = Imputer()
    imputer.fit(X_cleaned)
    df = imputer.transform(X_cleaned)
    dummier = Dummifier()
    dummier.fit(df)
    df = dummier.transform(df)
    standarizer = Standardizer()
    standarizer.fit(df)
    df = standarizer.transform(df)
    return df, y

def train(df, y): 
    """Input cleaned, featurized data, train/test split, 
    train and return a log-loss"""
    splits = split_once(df, y)
    te_idx = splits[0][0]
    tr_idx = splits[0][1]
    X_train = df.iloc[tr_idx,:]
    X_test = df.iloc[te_idx,:]
    y_train = y.iloc[tr_idx] 
    y_test = y.iloc[te_idx]
    clf = RandomForestClassifier(n_estimators=5000, max_depth=25)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    return log_loss(y_test, preds.T[1]), clf

def fraud_pipeline():
    """instantiate a pipeline object"""
    pipeline = Pipeline([
        ('featurizer', Featurizer()),
        ('imputer', Imputer()),
        ('dummifier', Dummifier()),
        ('standardizer', Standardizer()),
        ('model', RandomForestClassifier(n_estimators=5000, 
                                         max_depth=25))
        ])
    return pipeline

def pickle_pipeline(pipeline, output_name):
    """Save fitted pipeline to pickle file"""
    with open(output_name, 'wb') as f:
        pickle.dump(pipeline, f)

def split_once(df,y):
    """split data using StratifiedShuffleSplit 
    and return split indexes"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    split = sss.split(df, y)
    splits = list(split)
    return splits