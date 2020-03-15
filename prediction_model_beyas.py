# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:12:06 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from math import radians, atan, tan, sin, acos, cos
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('good_train.csv', low_memory=False)
test = pd.read_csv('good_test.csv', low_memory=False)

# 根据训练集 计算朴素贝叶斯算法需要使用的 条件概率
Probability = {}
## P(start_block|end_block)
name = 'start_block'
pname = 'P(start_block|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts()) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
## P(out_id|end_block)
name = 'out_id'
pname = 'P(out_id|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts()) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P(is_holiday|end_block)
name = 'is_holiday'
pname = 'P(is_holiday|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts() + 3.) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P((is_holiday, hour)|end_block)
pname = 'P((is_holiday, hour)|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (5 + 1.0 * g.groupby(['is_holiday', 'hour']).size()) / (len(g))
tmp = train.groupby('end_block').apply(tmp_func).reset_index().rename(columns={0: pname})
print(tmp.head())
Probability[pname] = tmp
## P(day|end_block)
name = 'day'
pname = 'P(day|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P(hour|end_block)
name = 'hour'
pname = 'P(hour|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
# 根据训练集 计算先验概率
pname = 'P(end_block)'
print('calculating %s' % pname)
tmp = train.end_block.value_counts().reset_index()
tmp.columns = ['end_block', pname]
Probability[pname] = tmp
## 计算后验概率 
## P(end_block|(start_block, out_id, is_holiday, hour)) = P(end_block) *
##                         P(start_block|end_block) * P(out_id|end_block) * P((is_holiday, hour)|end_block)
is_local = False  # 是否线下验证
if is_local:
    predict_info = train.copy()
    predict_info = predict_info.rename(columns={'end_block': 'true_end_block', 'end_lat': 'true_end_lat', 'end_lon': 'true_end_lon'})
else:
    predict_info = test.copy()
##
predict_info = predict_info.merge(Probability['P(out_id|end_block)'], on='out_id', how='left')
print(predict_info['P(out_id|end_block)'].isnull().sum())
predict_info['P(out_id|end_block)'] = predict_info['P(out_id|end_block)'].fillna(1e-5)
##
predict_info = predict_info.merge(Probability['P(is_holiday|end_block)'], on=['is_holiday', 'end_block'], how='left')
print(predict_info['P(is_holiday|end_block)'].isnull().sum())
predict_info['P(is_holiday|end_block)'] = predict_info['P(is_holiday|end_block)'].fillna(1e-4)
##
predict_info = predict_info.merge(Probability['P(day|end_block)'], on=['day', 'end_block'], how='left')
print(predict_info['P(day|end_block)'].min(), predict_info['P(day|end_block)'].isnull().sum())
predict_info['P(day|end_block)'] = predict_info['P(day|end_block)'].fillna(1e-4)
##
predict_info = predict_info.merge(Probability['P((is_holiday, hour)|end_block)'], on=['is_holiday', 'hour', 'end_block'], how='left')
print(predict_info['P((is_holiday, hour)|end_block)'].isnull().sum())
predict_info['P((is_holiday, hour)|end_block)'] = predict_info['P((is_holiday, hour)|end_block)'].fillna(1e-4)
##
predict_info = predict_info.merge(Probability['P(start_block|end_block)'], on=['start_block', 'end_block'], how='left')
print(predict_info['P(start_block|end_block)'].isnull().sum())
predict_info['P(start_block|end_block)'] = predict_info['P(start_block|end_block)'].fillna(1e-5)
##
predict_info = predict_info.merge(Probability['P(end_block)'], on='end_block', how='left')
print(predict_info['P(end_block)'].isnull().sum())
predict_info['P(end_block)'] = predict_info['P(end_block)'].fillna(1e-1)
predict_info['P(end_block|(start_block, out_id, is_holiday, hour))'] = predict_info['P((is_holiday, hour)|end_block)'] * \
                                                    predict_info['P(out_id|end_block)'] * \
                                                    predict_info['P(start_block|end_block)'] * \
                                                    predict_info['P(end_block)']
which_probability = 'P(end_block|(start_block, out_id, is_holiday, hour))'
# 生成每个聚类label的经纬度
block_lat_lon = train.groupby('end_block')[['end_lat', 'end_lon']].mean().reset_index()
#block_lat_lon = train.groupby('end_block')[['end_lat', 'end_lon']].agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()
predict_info = predict_info.merge(block_lat_lon, on='end_block', how='left')
print(predict_info[['start_lat', 'start_lon', 'end_lat', 'end_lon']].describe())
predict_result = predict_info.groupby('r_key').apply(lambda g: g.loc[g[which_probability].idxmax(), :]).reset_index(drop=True)
if not is_local:
    output_result = test[['r_key', 'start_lat', 'start_lon']].merge(predict_result[['r_key', 'end_lat', 'end_lon']], on='r_key', how='left')
    print(output_result.end_lat.isnull().sum())
    # 冷启动暂时用其实经纬度作为预测结果 
    nan_idx = output_result.end_lat.isnull()
    output_result.loc[nan_idx, 'end_lat'] = output_result['start_lat'][nan_idx]
    output_result.loc[nan_idx, 'end_lon'] = output_result['start_lon'][nan_idx]
    #output_result[['start_lat', 'end_lat', 'end_lon']].describe()
    print(output_result.head())
    print(output_result.info())
    output_result[['r_key', 'end_lat', 'end_lon']].to_csv('bayes4.csv', index=None)
