#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[2]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import randint, uniform

import matplotlib.pyplot as plt
from time import perf_counter
import warnings
warnings.filterwarnings("ignore")


# # **IMPORT DATA, TEST/TRAIN SPLIT, SCALE**

# In[3]:


# features = pd.read_csv("geometric_features.csv")

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



# # **BASELINE GRADIENT BOOSTING**

# In[4]:


gb_model = GradientBoostingClassifier(learning_rate=15, n_estimators=18, max_depth=8)
start_time=perf_counter()
gb_model.fit(X_train, y_train)
end_time=perf_counter()
print('total_training_time: ',end_time-start_time)
y_pred = gb_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))

# # **FEATURE IMPORTANCE**

# In[33]:


importance = gb_model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# kp save to csv file
# results=pd.read_csv('algo_results.csv')
# algo_name='gb'
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
    
#     main_row=gb_model.classes_.tolist()
#     main_row.insert( 0, 'Name')
# #     print(main_row)
#     # write a row to the csv file
#     writer.writerow(main_row)

#     # close the file
#     f.close()
 
# for i, value in enumerate(gb_model.classes_):
#     try:
#         f1_score=cls_report[value]['f1-score']
#         class_name=value
#         df = pd.read_csv("test.csv")
#         df.loc[6,'Name'] = 'Gradient_Boosting'
#         df.loc[6,class_name] = f1_score
#         df.to_csv("test.csv", index=False)
#     except:
#         continue

# ########

'''
# In[16]:


gb_model.get_params()


# # **HYPERPARAMETER TUNING PART I**

# In[33]:


model = GradientBoostingClassifier()

params_grid = {
    "n_estimators" : randint(5,500),
    "max_depth" : randint(1,10),
    "learning_rate": uniform(0.01, 100)
}


# In[34]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
random_search = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=cv, param_distributions=params_grid, scoring='f1_micro')
search_results = random_search.fit(X_train, y_train)


# In[35]:


search_results.best_estimator_


# In[36]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# # **HYPERPARAMETER TUNING PART II**

# In[27]:


model = GradientBoostingClassifier()

n_estimators = range(0,100,10)
max_depth = range(1,5)
learning_rate = np.arange(.1,2.5,.1) 

params_grid = {
    "n_estimators" : n_estimators, 
    "max_depth" : max_depth,
    "learning_rate": learning_rate
}


# In[28]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
grid_search = GridSearchCV(estimator=model, param_grid=params_grid, n_jobs=-1, cv=cv, scoring='f1_micro')
search_results = grid_search.fit(X_train, y_train)


# In[29]:


search_results.best_estimator_


# In[31]:


best_model.get_params()


# In[30]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# # **CONFUSION MATRIX**

# In[32]:


cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
cmd = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
cmd.plot(ax=ax, xticks_rotation='vertical');

plt.savefig('confusionmatrix_gradientboosting.png', dpi=600)


# # **FEATURE IMPORTANCE**

# In[33]:


importance = best_model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[34]:


plt.figure(figsize=(25,15))
plt.bar( range(len(best_model.feature_importances_)), best_model.feature_importances_)
plt.xticks(range(len(best_model.feature_importances_)), X.columns)
plt.savefig('featureimportance_gradientboosting.png', dpi=600)
plt.show()
'''
