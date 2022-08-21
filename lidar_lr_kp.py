#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[13]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from time import perf_counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# # **IMPORT DATA, TEST/TRAIN SPLIT, SCALE**

# In[2]:



# features = pd.read_csv('../data/paris_lille/geometric_features.csv')

# features = pd.read_csv('geometric_features.csv')


# X = features.drop(['class'], axis=1)
# y = features[['class']].values.flatten()

# X_train, X_test, y_train, y_test = train_test_split(X, y)


## kp reading testing and training data seperately 
X_train=pd.read_csv('../X_train.csv')
X_test=pd.read_csv('../X_test.csv')
y_train=pd.read_csv('../y_train.csv')
y_test=pd.read_csv('../y_test.csv')
y_train=y_train['true_labels'].to_numpy()
y_test=y_test['true_labels'].to_numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # **BASE MODEL**

# In[3]:


logistic_model = LogisticRegression(multi_class='multinomial', solver='newton-cg',C=10.0)
start_time=perf_counter()
logistic_model.fit(X_train, y_train)
end_time=perf_counter()
print('total training time: ',end_time-start_time)

y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# # kp save to csv file
# results=pd.read_csv('algo_results.csv')
# algo_name='lr'
# results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
# # print('results from 1st column',results[2:])
# print(results)
# print('results',results.shape)
# y_pred=pd.DataFrame(y_pred)
# y_pred.columns=[algo_name]
# try:
#     results[algo_name]=y_pred[algo_name]
# except:
#     results=pd.concat([results, y_pred[algo_name]],axis=1)
# print('y_pred',y_pred.shape)
# print('results',results.shape)
# results.to_csv('algo_results.csv')


'''
# # **HYPERPARAMETER TUNING**

# In[7]:


model = LogisticRegression()

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

params_grid = {
    'solver' : solvers,
    'penalty' : penalty,
    'C' : c_values
}


# In[8]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
grid_search = GridSearchCV(estimator=model, param_grid=params_grid, n_jobs=-1, cv=cv, scoring='f1_micro')
search_results = grid_search.fit(X_train, y_train)


# In[9]:


search_results.best_estimator_


# In[14]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))
'''



# ## Karnik ## Creating CSV of F1-scores
# cls_report=classification_report(y_test, y_pred, digits=3,output_dict=True)
# import csv
# from os.path import exists
# import pandas as pd
# file_exists = exists('test.csv')
# if(file_exists==False):
#     f = open('test.csv' ,'w')
#     # create the csv writer
#     writer = csv.writer(f)
    
#     main_row=logistic_model.classes_.tolist()
#     main_row.insert( 0, 'Name')
# #     print(main_row)
#     # write a row to the csv file
#     writer.writerow(main_row)

#     # close the file
#     f.close()
 
# for i, value in enumerate(logistic_model.classes_):
#     try:
#         f1_score=cls_report[value]['f1-score']
#         class_name=value
#         df = pd.read_csv("test.csv")
#         df.loc[0,'Name'] = 'Logistic_regression'
#         df.loc[0,class_name] = f1_score
#         df.to_csv("test.csv", index=False)
#     except:
#         continue

# ########




# # **CONFUSION MATRIX**

# In[15]



# cm = confusion_matrix(y_test, y_pred, labels=logistic_model.classes_)
# cmd = ConfusionMatrixDisplay(cm, display_labels=logistic_model.classes_)
# fig = plt.figure(figsize=(20,20))
# ax = fig.add_subplot(111)
# cmd.plot(ax=ax, xticks_rotation='vertical');

# plt.savefig('confusionmatrix_lr.png', dpi=600)

