# -*- coding: utf-8 -*-

import sys
import os
lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

import numpy as np
import pandas as pd

from dataset import load_data

def check_numeric(col):
    ### 檢查某 feature 中是屬於 continuous 或 categorical
    if col.dtype == float:
        return True
    else:
        return False

def entropy(col):
    ### sum of -p*log2(p)
    ent = 0
    unique = np.unique(col)
    for value in unique:
        prob = len(np.where(col==value)[0]) / len(col)    
        ent += -1 * prob * np.log2(prob)
    return ent
    
def calculate_gain_ratio(df, col_name, label_col_name='label'):

    ent = entropy(df['label'])
    if check_numeric(df[col_name]):  ### continuous feature
        unique = np.unique(df[col_name])
        decision_stumps = (unique[:-1] + unique[1:])/2  ### 根據可能數值兩兩的中位數當作切分點
        max_gain_ratio = 0
        for d in decision_stumps:
            gain = ent
            IV = 0
            df_left = df.loc[df[col_name] < d]
            df_right = df.loc[df[col_name] > d]
            gain -= (len(df_left)/len(df)) * entropy(df_left[label_col_name])
            gain -= (len(df_right)/len(df)) * entropy(df_right[label_col_name])
            IV -= (len(df_left)/len(df)) * np.log2((len(df_left)/len(df)))
            IV -= (len(df_right)/len(df)) * np.log2((len(df_right)/len(df)))
            if gain/IV > max_gain_ratio:
                max_gain_ratio = gain/IV
                decision_stump = d    
        return max_gain_ratio, decision_stump, True
        
    else:   ### categorical feature
        gain = ent
        IV = 0
        for value, df_group in df[col_name].groupby(col_name):
            sub_ent = entropy(df_group[label_col_name])
            gain -= (len(df_group)/len(df)) * sub_ent
            IV -= (len(df_group)/len(df)) * np.log2((len(df_group)/len(df)))
        return gain/IV, 0, False
    
class Node:
    def __init__(self, feature=None, decision_stump=None, label=None):
        self.children = []
        self.feature = feature
        self.decision_stump = decision_stump
        self.label = label

    def add_child(self, data):
        self.children.append(data)

def generate_decision_tree(df, label_col_name='label'):
    
    ### 此節點全部 class 都相同
    if len(np.unique(df[label_col_name])) == 1:
        return Node(label=df[label_col_name].values[0])
        
    ### 此節點全部 feature 都相同，選最多的 class 當最後 label
    ### 此節點只剩下一個 feature，選最多的 class 當最後 label
    df_check = df.drop([label_col_name], axis=1)
    if np.all(df_check.eq(df_check.iloc[0]).all(axis=1)) or len(df.columns) == 2:       
        label = df[label_col_name].value_counts().index[0]
        return Node(label=label)

    max_gain_ratio = 0
    for col in df.columns:
        if col == label_col_name: # 不需要計算 label 這欄
            continue
        gain_ratio, decision_stump, numeric = calculate_gain_ratio(df, col)
        if gain_ratio > max_gain_ratio:
            max_col = col # feature 名稱
            max_gain_ratio = gain_ratio 
            max_decision_stump = decision_stump # 如果是 continuous feature，根據多少來切
            max_numeric = numeric # True 表示是 continuous feature     
            
    if max_numeric:
        node = Node(feature=max_col, decision_stump=max_decision_stump)
        ### 記得要切完之後要把該項 feature 刪去
        df_left = df.loc[df[max_col] < max_decision_stump].drop(max_col, axis=1)
        df_right = df.loc[df[max_col] > max_decision_stump].drop(max_col, axis=1)
        node.add_child(generate_decision_tree(df_left))
        node.add_child(generate_decision_tree(df_right))
    else:
        node = Node(feature=max_col)
        for value, df_group in df.groupby(max_col):
            node.add_child(generate_decision_tree(df_group)).drop(max_col, axis=1)
 
    return node

def predict(decision_tree, df):
    
    return
        
if __name__ == '__main__':
    
    data_path = os.path.join(lib_path, 'dataset', 'iris', 'iris.data.txt')
    df_train_iris, df_test_iris = load_data.load_iris_dataset(data_path)
    
    decision_tree = generate_decision_tree(df_train_iris)
    
