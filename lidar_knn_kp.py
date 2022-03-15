#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[2]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import randint

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# # **IMPORT DATA, TEST/TRAIN SPLIT, SCALE**

# In[3]:


# features = pd.read_csv("../data/paris_lille/geometric_features.csv")
# features = pd.read_csv("geometric_features.csv")

# X = features.drop(['class'], axis=1)
# y = features[['class']].values.flatten()

# X_train, X_test, y_train, y_test = train_test_split(X, y)


# print('X_test keys',X_test.keys())

# y_train = y_train.values.flatten()
# y_train=y_train.iloc[:,1].to_numpy()
# y_test=y_test.iloc[:,1].to_numpy()
X_train=pd.read_csv('X_train.csv')
X_test=pd.read_csv('X_test.csv')
y_train=pd.read_csv('y_train.csv')
y_test=pd.read_csv('y_test.csv')
y_train=y_train['true_labels'].to_numpy()
y_test=y_test['true_labels'].to_numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(y_train)
# print(y_train.iloc[:,1])
# print(type(y_train))

# print('training data',type(X_train))
# print('shape',X_train.shape)
# print(y_train)



# # **BASELINE KNN**

# In[4]:


knn_model = KNeighborsClassifier(algorithm='kd_tree',metric='euclidean',n_neighbors=9,weights='distance')
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)


# print('X_test keys',X_test.keys())
print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# kp save to csv file
results=pd.read_csv('algo_results.csv')
algo_name='knn'
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


'''
## Karnik ## Creating CSV of F1-scores
cls_report=classification_report(y_test, y_pred, digits=3,output_dict=True)
import csv
from os.path import exists
import pandas as pd
file_exists = exists('test.csv')
if(file_exists==False):
    f = open('test.csv' ,'w')
    # create the csv writer
    writer = csv.writer(f)
    
    main_row=knn_model.classes_.tolist()
    main_row.insert( 0, 'Name')
#     print(main_row)
    # write a row to the csv file
    writer.writerow(main_row)

    # close the file
    f.close()
 
for i, value in enumerate(knn_model.classes_):
    try:
        f1_score=cls_report[value]['f1-score']
        class_name=value
        df = pd.read_csv("test.csv")
        df.loc[4,'Name'] = 'K-Nearest Neighbors'
        df.loc[4,class_name] = f1_score
        df.to_csv("test.csv", index=False)
    except:
        continue

########

'''

'''
# # **HYPERPARAMETER TUNING PART I**

# In[5]:


model = KNeighborsClassifier()

n_neighbors = randint(5,100)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
algorithm = ['ball_tree', 'kd_tree']

params_grid = {'n_neighbors': n_neighbors,
               'weights': weights,
               'metric' : metric,
               'algorithm': algorithm}


# In[6]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
random_search = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=cv, param_distributions=params_grid, scoring='f1_micro')
search_results = random_search.fit(X_train, y_train)


# In[7]:


search_results.best_estimator_


# In[8]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))




# # **HYPERPARAMETER TUNING PART II**

# In[21]:


model = KNeighborsClassifier()

n_neighbors = range(0,100)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
algorithm = ['ball_tree', 'kd_tree']

params_grid = {'n_neighbors': n_neighbors,
               'weights': weights,
               'metric' : metric,
               'algorithm': algorithm}


# In[22]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
grid_search = GridSearchCV(estimator=model, param_grid=params_grid, n_jobs=-1, cv=cv, scoring='f1_micro')
search_results = grid_search.fit(X_train, y_train)


# In[23]:


search_results.best_estimator_


# In[24]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# # **CONFUSION MATRIX**

# In[28]:


cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
cmd = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
cmd.plot(ax=ax, xticks_rotation='vertical');

plt.savefig('confusionmatrix_knn.png', dpi=600)


# # **FEATURE IMPORTANCE**

# In[29]:


results = permutation_importance(best_model, X_test, y_test, scoring='f1_micro')
importance = results.importances_mean
for i,v in enumerate(importance):
    print('Feature: %0d, Score %.5f' % (i,v))


# In[ ]:


plt.figure(figsize=(25,15))
plt.bar(range(len(results.importances_mean)), results.importances_mean)
plt.xticks(range(len(results.importances_mean)), X.columns)
plt.savefig('featureimportance_knn.png', dpi=600)
plt.show()
'''
