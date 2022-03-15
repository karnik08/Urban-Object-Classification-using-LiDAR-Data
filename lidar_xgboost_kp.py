#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[8]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")


# # **IMPORT DATA, TEST/TRAIN SPLIT, SCALE**

# In[3]:


#features = pd.read_csv("../data/paris_lille/geometric_features.csv")
# features = pd.read_csv("geometric_features.csv")

# X = features.drop(['class'], axis=1)
# y = features[['class']].values.flatten()

# print(len(X))
# X=X[:300]
# y=y[:300]


# X_train, X_test, y_train, y_test = train_test_split(X, y)

## kp reading testing and training data seperately 
X_train=pd.read_csv('X_train.csv')
X_test=pd.read_csv('X_test.csv')
y_train=pd.read_csv('y_train.csv')
y_test=pd.read_csv('y_test.csv')
y_train=y_train['true_labels'].to_numpy()
y_test=y_test['true_labels'].to_numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# # **BASELINE XGBOOST**

# In[17]:


model = XGBClassifier(tree_method='gpu_hist',gpu_id=0)
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))

# kp save to csv file
results=pd.read_csv('algo_results.csv')
algo_name='xgboost'
results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
# print('results from 1st column',results[2:])
print(results)
print('results',results.shape)
y_pred=pd.DataFrame(y_pred)
y_pred.columns=[algo_name]
try:
    results[algo_name]=y_pred[algo_name]
except:
    results=pd.concat([results, y_pred[algo_name]],axis=1)
print('y_pred',y_pred.shape)
print('results',results.shape)
results.to_csv('algo_results.csv')



# # **CONFUSION MATRIX**

# In[13]:


cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cmd = ConfusionMatrixDisplay(cm, display_labels=model.classes_)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
cmd.plot(ax=ax, xticks_rotation='vertical');

plt.savefig('confusionmatrix_xgboost.png', dpi=600)


# # **FEATURE IMPORTANCE**

# In[14]:


importance = model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[16]:

    
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
    
#     main_row=model.classes_.tolist()
#     main_row.insert( 0, 'Name')
# #     print(main_row)
#     # write a row to the csv file
#     writer.writerow(main_row)

#     # close the file
#     f.close()
 
# for i, value in enumerate(model.classes_):
#     try:
#         f1_score=cls_report[value]['f1-score']
#         class_name=value
#         df = pd.read_csv("test.csv")
#         df.loc[1,'Name'] = 'XGBoost'
#         df.loc[1,class_name] = f1_score
#         df.to_csv("test.csv", index=False)
#     except:
#         continue

# ########


plt.figure(figsize=(25,15))
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.xticks(range(len(model.feature_importances_)), X.columns)
plt.savefig('featureimportance_xgboost.png', dpi=600)
plt.show()

