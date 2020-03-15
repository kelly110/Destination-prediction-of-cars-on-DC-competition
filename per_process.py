# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:13:50 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from math import radians, atan, tan, sin, acos, cos
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

train = pd.read_csv('train_new.csv', low_memory=False)
test = pd.read_csv('test_new.csv', low_memory=False)
# ################################# 准备坐标点数据################################

trL = train.shape[0] * 2
X = np.concatenate([train[['start_lat', 'start_lon']].values,
                    train[['end_lat', 'end_lon']].values,
                    test[['start_lat', 'start_lon']].values])
#plt.figure(1)
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()
# #############################################################################
# 对经纬度坐标点进行密度聚类 
db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(X)
labels = db.labels_
# 打印聚类数
n_clusters_ = len(set(labels))
print('Estimated number of clusters: %d' % n_clusters_)
'''
#plt.figure(2)
#for i in range(n_clusters_):
##        print('簇 ', i, '的所有样本:')
#    one_cluster = X[labels == i]
##        print(one_cluster)
#    plt.plot(one_cluster[:,0],one_cluster[:,1],'o')
#    
#plt.show()
'''
# 训练集聚类label
info = pd.DataFrame(X[:trL,:], columns=['lat', 'lon'])
info['block_id'] = labels[:trL]
clear_info = info.loc[info.block_id != -1, :]
print('The number of miss start block in train data', (info.block_id.iloc[:trL//2] == -1).sum())
print('The number of miss end block in train data', (info.block_id.iloc[trL//2:] == -1).sum())
# 测试集聚类label
test_info = pd.DataFrame(X[trL:,:], columns=['lat', 'lon'])
test_info['block_id'] = labels[trL:]
print('The number of miss start block in test data', (test_info.block_id == -1).sum())
# 将聚类label拼接到训练集和测试集上
train['start_block'] = info.block_id.iloc[:trL//2].values
train['end_block'] = info.block_id.iloc[trL//2:].values
test['start_block'] = test_info.block_id.values
good_train_idx = (train.start_block != -1) & (train.end_block != -1)
print('The number of good training data', good_train_idx.sum())
good_train = train.loc[good_train_idx, :]
print('saving new train & test data')
# 为训练集和测试集生成is_holiday 和 hour字段
def time(t):#时间分段   
    if t<7:
        tb=0
    elif t<=9:
        tb=1
    elif t<=11:
        tb=2
    elif t<=13:
        tb=3
    elif t<15:
        tb=4
    elif t<17:
        tb=5
    elif t<=19:
        tb=6
    elif t<22:
        tb=7
    else:
        tb=8
    return tb    
def transformer(df):
    special_holiday = ['2018-01-01'] + ['2018-02-%d' % d for d in range(15, 22)] + \
                      ['2018-04-%2d' % d for d in range(5, 8)] + \
                      ['2018-04-%d' % d for d in range(29, 31)] + ['2018-05-01'] +\
                      ['2018-06-%d' % d for d in range(16, 19)] + \
                      ['2018-09-%d' % d for d in range(22, 25)] + \
                      ['2018-10-%2d' % d for d in range(1, 8)]
    special_workday = ['2018-02-%d' % d for d in [11, 24]] + \
                      ['2018-04-08'] + ['2018-04-28'] + \
                      ['2018-09-%d' % d for d in range(29, 31)]
    for t_col in ['start_time']:
        tmp = df[t_col].map(pd.Timestamp)
#        df_hour=df[]        
        
        df['hour'] = tmp.map(lambda t: time(t.hour))
#        df['half'] = tmp.map(lambda t: t.minute // 30)
        df['day'] = tmp.map(lambda t: t.dayofweek)
        tmp_date = df[t_col].map(lambda s: s.split(' ')[0])
        not_spworkday_idx = ~tmp_date.isin(special_workday)
        spholiday_idx = tmp_date.isin(special_holiday)
        weekend_idx = (df['day'] >= 5)
        df['is_holiday'] = ((weekend_idx & not_spworkday_idx) | spholiday_idx).astype(int)


transformer(train)
transformer(test)
train.to_csv('F:/AI/data/good_train.csv', index=None)
test.to_csv('F:/AI/data/good_test.csv', index=None)
#train = pd.read_csv('good_train.csv', low_memory=False)
#test = pd.read_csv('good_test.csv', low_memory=False)