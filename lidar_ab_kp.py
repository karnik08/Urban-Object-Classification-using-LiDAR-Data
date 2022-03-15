#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[3]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import randint, uniform

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# # **IMPORT DATA, TEST/TRAIN SPLIT, SCALE**

# In[4]:


# features = pd.read_csv("geometric_features.csv")

# X = features.drop(['class'], axis=1)
# y = features[['class']].values.flatten()

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


# # **BASELINE ADABOOST**

# In[3]:


# ab_model = AdaBoostClassifier()
# ab_model.fit(X_train, y_train)
# y_pred = ab_model.predict(X_test)

# print(classification_report(y_test, y_pred, digits=3))
# print(f1_score(y_test, y_pred, average='micro'))


# In[4]:


# svc_model = SVC(probability=True)
# ab_model = AdaBoostClassifier(base_estimator=svc_model,learning_rate=6.37, n_estimators=367)
base = DecisionTreeClassifier(max_depth=9)
ab_model = AdaBoostClassifier(base_estimator=base,learning_rate=2.75, n_estimators=490)
ab_model.fit(X_train, y_train)
y_pred = ab_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# kp save to csv file
results=pd.read_csv('algo_results.csv')
algo_name='ab'
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
    
#     main_row=ab_model.classes_.tolist()
#     main_row.insert( 0, 'Name')
# #     print(main_row)
#     # write a row to the csv file
#     writer.writerow(main_row)

#     # close the file
#     f.close()
 
# for i, value in enumerate(ab_model.classes_):
#     try:
#         f1_score=cls_report[value]['f1-score']
#         class_name=value
#         df = pd.read_csv("test.csv")
#         df.loc[5,'Name'] = 'AdaBoost'
#         df.loc[5,class_name] = f1_score
#         df.to_csv("test.csv", index=False)
#     except:
#         f1_score='not tested'
#         class_name=value
#         df = pd.read_csv("test.csv")
#         df.loc[5,'Name'] = 'AdaBoost'
#         df.loc[5,class_name] = f1_score
#         df.to_csv("test.csv", index=False)
#         continue

# ########
'''

# # **HYPERPARAMETER TUNING PART I**

# In[36]:


model = AdaBoostClassifier()

n_estimators = randint(5, 500)
learning_rate = uniform(.0001, 10)

params_grid = {'n_estimators': n_estimators,
               'learning_rate' : learning_rate}


# In[37]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
random_search = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=cv, param_distributions=params_grid, scoring='f1_micro')
search_results = random_search.fit(X_train, y_train)


# In[38]:


search_results.best_estimator_


# In[39]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# # **HYPERPARAMETER TUNING PART II**

# In[25]:


def makeModels():
    models = dict()
    
    for i in range(1, 20):
        base = DecisionTreeClassifier(max_depth=i)
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
    
    return models


def evaluateModels(model, X, y):
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    return scores
    
models = makeModels()
results, names = list(), list()

for name, model in models.items():
    scores = evaluateModels(model, X_train, y_train)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


# In[26]:


base = DecisionTreeClassifier(max_depth=9)
model = AdaBoostClassifier(base_estimator=base)

n_estimators = randint(5, 500)
learning_rate = uniform(.0001, 10)

params_grid = {'n_estimators': n_estimators,
               'learning_rate' : learning_rate}


# In[27]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
random_search = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=cv, param_distributions=params_grid, scoring='f1_micro')
search_results = random_search.fit(X_train, y_train)


# In[28]:


search_results.best_estimator_


# In[29]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# # **CONFUSION MATRIX**

# In[30]:


cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
cmd = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
cmd.plot(ax=ax, xticks_rotation='vertical');

plt.savefig('confusionmatrix_adaboost.png', dpi=600)


# # **FEATURE IMPORTANCE**

# In[40]:


importance = best_model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[41]:


plt.figure(figsize=(25,15))
plt.bar( range(len(best_model.feature_importances_)), best_model.feature_importances_)
plt.xticks(range(len(best_model.feature_importances_)), X.columns)
plt.savefig('featureimportance_adaboost.png', dpi=600)
plt.show()

'''