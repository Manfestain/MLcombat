# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

train = pd.read_csv('F:/Download/train_modified.csv')
target = 'Disbursed'
Idcol = 'ID'
#print(train['Disbursed'].value_counts())

x_columes = [x for x in train.columns if x not in [target, Idcol]]
X = train[x_columes]
y = train['Disbursed']

def randomForest0():
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(X, y)
    print(rf0.oob_score_)
    y_predprob = rf0.predict_proba(X)[:, 1]
    print('AUC Score : %f' % metrics.roc_auc_score(y, y_predprob))

def gridSearch0():
    param_test1 = {'n_estimators': range(10, 71, 10)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                             min_samples_leaf=20,
                                                             max_depth=8,
                                                             max_features='sqrt',
                                                             random_state=10),
                            param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(X, y)
    print(gsearch1.best_params_, gsearch1.best_score_)

    param_test2 = {'max_depth': range(3, 14, 2),
                   'min_samples_split': range(50, 201, 20)}
    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,
                                                             min_samples_leaf=20,
                                                             max_features='sqrt',
                                                             oob_score=True,
                                                             random_state=10),
                            param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(X, y)
    print(gsearch2.best_params_, gsearch2.best_score_)

def randomForest1():
    rf1 = RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=110,
                                 min_samples_leaf=20, max_features='sqrt', oob_score=True,
                                 random_state=10)
    rf1.fit(X, y)
    print(rf1.oob_score_)

def gridSearch1():
    param_test3 = {'min_samples_split': range(80, 150, 20),
                   'min_samples_leaf': range(10, 60, 10),
                   'max_features': range(3, 11, 2)}
    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,
                                                             max_depth=13,
                                                             oob_score=True,
                                                             random_state=10),
                            param_grid=param_test3, scoring='roc_auc', cv=5)
    gsearch3.fit(X, y)
    print(gsearch3.best_params_, gsearch3.best_score_)

def randomForest2():
    rf2 = RandomForestClassifier(n_estimators=60, max_features=7, min_samples_leaf=20,
                                 min_samples_split=120, max_depth=13, oob_score=True,
                                 random_state=10)
    rf2.fit(X, y)
    print(rf2.oob_score_)

if __name__ == '__main__':
    #randomForest0()
    #gridSearch0()
    #randomForest1()
    #gridSearch1()
    randomForest2()