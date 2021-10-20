# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:24:41 2021

@author: jiguifang
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences

from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from model.dssm import DSSM

'''
序列填充，0
'''

# 获取训练集和测试集
def data_process(data_path, samp_rows=10000):
    data = pd.read_csv(data_path, nrows=samp_rows)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    train = data.iloc[:int(len(data)*0.8)].copy()
    test = data.iloc[int(len(data)*0.8):].copy()
    return train, test, data


def get_user_feature(data):
    data_group = data[data['rating'] == 1]
    data_group = data_group[['user_id', 'movie_id']].groupby('user_id').agg(list).reset_index()
    # 分组后，多行合并到一个列表中
    data_group['user_hist'] = data_group['movie_id'].apply(lambda x: '|'.join([str(i) for i in x]))
    # print(data_group.columns, data_group.head(10))
    # 合并用户看过的电影
    data = pd.merge(data_group.drop('movie_id', axis=1), data, on='user_id')
    # data_group.drop('movie_id', axis=1) 删除movie_id列
    
    
    data_group = data[['user_id', 'rating']].groupby('user_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
    # 计算用户看过电影的平均分
    data = pd.merge(data_group, data, on='user_id')
    return data


def get_item_feature(data):
    data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='movie_id')
    return data


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",\
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))
    
    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length) # 最长的特征
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    print("user_hist_list: \n")

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')
    return test_hist


if __name__ == '__main__':
    # %%
    data_path = './data/movielens.txt'
    train, test, data = data_process(data_path, samp_rows=10000)
    train = get_user_feature(train)
    train = get_item_feature(train)
    print(train.columns)
    '''
    ['movie_id', 'item_mean_rating', 'user_id', 'user_mean_rating',
       'user_hist', 'rating', 'timestamp', 'gender', 'age', 'occupation',
       'zipcode', 'title', 'genres']
    '''
    
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
    dense_features = ['user_mean_rating', 'item_mean_rating']
    target = ['rating']
    
    user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
    item_sparse_features, item_dense_features = ['movie_id', ], ['item_mean_rating']
    
    
    # 1.Label Encoding for sparse features,and process sequence features
    '''
    le.fit([1, 2, 2, 6])
    le.classes_   # fit之后类别数组
    le.fit(["paris", "paris", "tokyo", "amsterdam"]) 
    le.transform(["tokyo", "tokyo", "paris"])    编码后：array([2, 2, 1]，dtype=int64)
    '''
    for feat in sparse_features:
        lbe = LabelEncoder()   # 类似与字典的形式编码
        lbe.fit(data[feat])  # 统计有多少个类别  
        train[feat] = lbe.transform(train[feat])   # 类别编码
        test[feat] = lbe.transform(test[feat])
    
    '''
    数据归一化
    （x - x.min) / (x.max - x.min)
    稠密数据归一化处理
    '''
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    train[dense_features] = mms.transform(train[dense_features])
    
    # 2.preprocess the sequence feature
    genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')
    
    user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                            for i, feat in enumerate(user_sparse_features)] + \
                        [DenseFeat(feat, 1, ) for feat in user_dense_features]
                        
    item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                            for i, feat in enumerate(item_sparse_features)] + \
                        [DenseFeat(feat, 1, ) for feat in item_dense_features]
                        
    item_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=4),
                                                    maxlen=genres_maxlen, combiner='mean', length_name=None)]

    user_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=3470, embedding_dim=4),
                                                    maxlen=user_maxlen, combiner='mean', length_name=None)]
    
    # 3.generate input data for model
    user_feature_columns += user_varlen_feature_columns
    item_feature_columns += item_varlen_feature_columns
    
    # add user history as user_varlen_feature_columns
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    
    '''train_genres_list genres的向量表示 [num, mxlen] mxlen 表示最长包含的属性'''
    train_model_input["genres"] = train_genres_list          
    train_model_input["user_hist"] = train_user_hist
    # print(type(train_model_input), train_model_input.keys()) 
    '''
    train_model_input 字典 
    keys
    ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'user_mean_rating', 
     'item_mean_rating', 'genres', 'user_hist']
    
    '''
    
    
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    
    print(item_dense_features)
    model = DSSM(user_feature_columns, item_feature_columns, task='binary', device=device)
    
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'])
    
    # 训练 继承base_tower
    model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
                                                     