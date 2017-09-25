# _*_ coding:utf-8 _*_

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def loadDataSet():
    news = fetch_20newsgroups(subset='all')
    X = news.data[:3000]
    y = news.target[:3000]
    return X, y

def single_gridSearch():
    X, y = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')),
                    ('svc', SVC())])
    parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
    # gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)
    gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)   # 使用并行搜索
    gs.fit(X_train, y_train)
    print(gs.best_params_, gs.best_score_)
    print('score:', gs.score(X_test, y_test))

if __name__ == '__main__':
    single_gridSearch()