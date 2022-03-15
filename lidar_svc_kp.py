#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from sklearn.svm import LinearSVC, SVC

from scipy.stats import uniform, loguniform

import warnings
warnings.filterwarnings("ignore")


# # **IMPORT DATA, TEST/TRAIN SPLIT, SCALE**

# In[6]:


#features = pd.read_csv("../data/paris_lille/geometric_features.csv")
features = pd.read_csv("geometric_features.csv")



X = features.drop(['class'], axis=1)
y = features[['class']].values.flatten()

# print(X[:50].shape)
# print(y[:50].shape)

# X=X[:100]
# y=y[:100]



X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[8]:


print('len of X_train',len(X_train))


# # **BASELINE SVC**

# In[9]:


svc_model = SVC(C=1, class_weight='balanced', degree=4, gamma=91.13708634647797, kernel='poly')
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)

print('classification_report',classification_report(y_test, y_pred, digits=3))
print('F1 score of y_test and y_pred',f1_score(y_test, y_pred, average='micro'))

    
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
    
    main_row=svc_model.classes_.tolist()
    main_row.insert( 0, 'Name')
#     print(main_row)
    # write a row to the csv file
    writer.writerow(main_row)

    # close the file
    f.close()
 
for i, value in enumerate(svc_model.classes_):
    try:
        f1_score=cls_report[value]['f1-score']
        class_name=value
        df = pd.read_csv("test.csv")
        df.loc[11,'Name'] = 'SVC'
        df.loc[11,class_name] = f1_score
        df.to_csv("test.csv", index=False)
    except:
        continue

########

'''
# # **HYPERPARAMETER TUNING**

# In[10]:


model = SVC()

#params_grid = [
#  {'C': [1], 'class_weight': ['balanced'], 'decision_function_shape' : ['ovo', 'ovr'], 'kernel': ['linear']},
#  {'C': [1], 'class_weight': ['balanced'], 'decision_function_shape' : ['ovo', 'ovr'], 'gamma': uniform(10, 100), 'kernel': ['rbf']},
#  {'C': [1], 'class_weight': ['balanced'], 'decision_function_shape' : ['ovo', 'ovr'], 'gamma': uniform(10, 100), 'kernel': ['poly'], 'degree': [2, 3, 4, 5]}, 
#  {'C': [1], 'class_weight': ['balanced'], 'decision_function_shape' : ['ovo', 'ovr'], 'gamma': uniform(10, 100), 'kernel': ['sigmoid']}
#]

params_grid = [
  {'C': [1], 'class_weight': ['balanced'],'kernel': ['linear']},
  {'C': [1], 'class_weight': ['balanced'],'gamma': uniform(50, 100), 'kernel': ['rbf']},
  {'C': [1], 'class_weight': ['balanced'],'gamma': uniform(50, 100), 'kernel': ['poly'], 'degree': [3, 4, 5]}, 
  {'C': [1], 'class_weight': ['balanced'],'gamma': uniform(50, 100), 'kernel': ['sigmoid']}
]


# In[11]:


#cv = RepeatedStratifiedKFold(n_split=5, n_repeats=2, random_state=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
random_search = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=cv, param_distributions=params_grid, scoring='f1_micro')
search_results = random_search.fit(X_train, y_train)


# In[ ]:


search_results.best_estimator_


# In[ ]:


best_model = search_results.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print('classification report y_test and y_pred after CV',classification_report(y_test, y_pred, digits=3))
print('f1 score y_test and y_pred after CV',f1_score(y_test, y_pred, average='micro'))


# In[ ]:


model = SVC(C=1, class_weight='balanced', degree=4, gamma=91.13708634647797, kernel='poly')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
print(f1_score(y_test, y_pred, average='micro'))


# #print(type(cls_report))
# print('all_classes',best_model.classes_)
# #print('all_scores',cls_report[:,'f1-score'])
# for i, value in enumerate(best_model.classes_):
#     #print('value',cls_report[value])
#     print(value,cls_report[value]['f1-score'])
    


# # **CONFUSION MATRIX**

# In[ ]:


cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
#print('confusion matrix',cm)
#print('labels',best_model.classes_)

# all_accuracy={}
# for i, value in enumerate(best_model.classes_):
#     temp=cm[i][i]/sum(cm[:,i])
#     all_accuracy[value]=temp
# print(all_accuracy)

cmd = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
cmd.plot(ax=ax, xticks_rotation='vertical');

plt.savefig('confusionmatrix_svm.png', dpi=600)


# # **FEATURE IMPORTANCE**

# In[ ]:


results = permutation_importance(best_model, X_test, y_test, scoring='f1_micro')
importance = results.importances_mean
for i,v in enumerate(importance):
    print('Feature: %0d, Score %.5f' % (i,v))


# In[ ]:


plt.figure(figsize=(25,15))
plt.bar(range(len(results.importances_mean)), results.importances_mean)
plt.xticks(range(len(results.importances_mean)), X.columns)
plt.savefig('featureimportance_svm.png', dpi=600)
plt.show()
'''
