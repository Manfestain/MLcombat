# *_* coding:utf-8 _*_

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV

# display data sets
train = pd.read_csv('F:/Download/train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'
#print(train['Disbursed'].value_counts())

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']
#print(x_columns)

# use default parameters
def dafaultGBDT():
    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(X, y)
    y_pred = gbm0.predict(X)
    y_predprob = gbm0.predict_proba(X)[:, 1]
    print('Accuracy : % .4g' % metrics.accuracy_score(y.values, y_pred))
    print('AUC Score : %f' % metrics.roc_auc_score(y, y_predprob))

# grid search
def gridSearch():
    # n_estimators
    param_test1 = {'n_estimators': range(20, 81, 10)}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                 min_samples_leaf=20,
                                                                 max_depth=8,
                                                                 max_features='sqrt',
                                                                 subsample=0.8,
                                                                 random_state=10),
                            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    gsearch1.fit(X, y)
    print(gsearch1.best_params_, gsearch1.best_score_)

    # max_depth and min_samples_split
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 800, 200)}
    gsearch2 = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20,
                                                       max_features='sqrt', subsample=0.8,
                                                       random_state=10),
                            param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(X, y)
    print(gsearch2.best_params_, gsearch2.best_score_)

def setGBDT1():
    gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, min_samples_leaf=60,
                                      min_samples_split=1200, max_features='sqrt', subsample=0.8,
                                      random_state=10)
    gbm1.fit(X, y)
    y_pred = gbm1.predict(X)
    y_predprob = gbm1.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score : %f" % metrics.roc_auc_score(y, y_predprob))

def setGBDT2():
    gbm2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, max_depth=7, min_samples_leaf=60,
                                      min_samples_split=1200, max_features=9, subsample=0.7,
                                      random_state=10)
    gbm2.fit(X, y)
    y_pred = gbm2.predict(X)
    y_predprob = gbm2.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score : %f" % metrics.roc_auc_score(y, y_predprob))

if __name__ == '__main__':
    #gridSearch()
    dafaultGBDT()
    print('--------------------')
    setGBDT1()
    print('--------------------')
    setGBDT2()
