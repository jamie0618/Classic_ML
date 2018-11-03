# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def process_na(df, strategy='drop'):
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'most_frequent':
        for col in df.columns:
            most_frequent = df[col].value_counts().idxmax()
            df[col] = df[col].fillna(most_frequent)
    
    return df

def one_hot_encoding(df, discrete_cols):
    
    for col in discrete_cols:
        df_tmp = pd.get_dummies(df[col])
        for sub_col in df_tmp.columns:
            df[col+'_'+sub_col] = df_tmp[sub_col]
        df = df.drop(col, axis=1)
    return df
    
def load_adult_dataset(one_hot=False, random_state=0, train_frac=0.8):
    
    with open('adult/adult.data.txt', 'r') as f:
        data = f.readlines()
        data = [ v.strip('\n').split(', ') for v in data ]
                
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race',
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'label']
    discrete_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                     'relationship', 'race', 'sex', 'native-country']

    df = pd.DataFrame(data, columns=cols)
    df = df.replace('?', np.NaN)
    df = process_na(df, strategy='drop')
    
    if one_hot:
        df = one_hot_encoding(df, discrete_cols)
    
    df_train = df.sample(frac=train_frac, random_state=random_state)
    df_test = df.drop(df_train.index)
    
    return df_train, df_test

def load_iris_dataset(random_state=0, train_frac=0.8):
    
    with open('iris/iris.data.txt', 'r') as f:
        data = f.readlines()
        data = [ v.strip('\n').split(',') for v in data ]  
                
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    df = pd.DataFrame(data, columns=cols)
    df_train = df.sample(frac=train_frac, random_state=random_state)
    df_test = df.drop(df_train.index)
    
    return df_train, df_test

def load_forestfire_dataset(one_hot=False, random_state=0, train_frac=0.8):
    
    df = pd.read_csv('forest_fire/forest_fire.csv')
    
    if one_hot:
        df = one_hot_encoding(df, ['month', 'day'])
        
    df_train = df.sample(frac=train_frac, random_state=random_state)
    df_test = df.drop(df_train.index)
    
    return df_train, df_test
    
if __name__ == '__main__':
    
    df_train_adult, df_test_adult = load_adult_dataset(one_hot=False)    
    df_train_iris, df_test_iris = load_iris_dataset()
    df_train_forestfire, df_test_forestfire = load_forestfire_dataset(one_hot=False)