#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
# In[3]:

#to split the data for Ml models

features = pd.read_csv("geometric_features.csv")
# features = df.sample(frac=0.50)

X = features.drop(['class'], axis=1)
y = features[['class']].values.flatten()
# y = features[['class']].values

# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# print(y_train)
# print(type(y_train))

y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
# print(y_train)
y_train.columns =['true_labels']
y_test.columns =['true_labels']
# print(y_train)
# print('X_train',X_train.shape)
# print('X_test',X_test.shape)
# print('y_train',y_train.shape)
# print('y_test',y_test.shape)

# print('X_train',type(X_train))
# print('X_test',type(X_test))
# print('y_train',type(y_train))
# print('y_test',type(y_test))


X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
y_test.to_csv('algo_results.csv')

'''

#to split the data for Dl models

pointnet=pd.read_csv('data/pointnet.csv')
count=0
for index, row in pointnet.iterrows():
#     if count==10:
#         break
#     count+=1
    temp=row['pointnet.csv'].split('.')[0].split('_')
    row['pointnet.csv']=temp[-2]+'_'+temp[-1]
#     print(row['pointnet.csv'])
# pointnet['pointnet.csv']=pointnet['pointnet.csv'].split('d')[-1]
# pointnet
# print(pointnet)
voxnet=pd.read_csv('data/voxnet.csv')
count=0
for index, row in voxnet.iterrows():
#     if count==10:
#         break
#     count+=1
    temp=row['202020000_1.npy'].split('.')
#     print(temp)
    row['202020000_1.npy']=temp[0]
# print('voxnet')
# print(voxnet)
common=voxnet.merge(pointnet, left_on='202020000_1.npy', right_on='pointnet.csv')
# print('common')
common['pointnet.csv'].to_csv('data/dlcommon.csv')
