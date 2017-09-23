# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection

def loadDataSet():
    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    y = titanic['survived']
    X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

    X['age'].fillna(X['age'].mean(), inplace=True)
    X.fillna('UNKNOW', inplace=True)
    return X, y

def main():
    X, y = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))
    print('dimension: ', len(vec.feature_names_))

    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train, y_train)
    print('None feature-selection: ', dt.score(X_test, y_test))

    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
    X_train_fs = fs.fit_transform(X_train, y_train)
    dt.fit(X_train_fs, y_train)
    X_test_fs = fs.transform(X_test)
    print('20% feature-selection: ', dt.score(X_test_fs, y_test))

    # 交叉验证，使用固定百分比进行特征筛选，并作图展示
    percentiles = range(1, 100, 2)
    results = []
    for i in percentiles:
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
        X_train_fs = fs.fit_transform(X_train, y_train)
        scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
        results = np.append(results, scores.mean())
    print('Result: \n', results)

    opt = np.where(results == results.max())[0][0]
    print(opt)
    print('Opeimal number of features %d ' % percentiles[opt])

    # 使用最佳筛选特征进行建模并评估
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
    X_train_fs = fs.fit_transform(X_train, y_train)
    dt.fit(X_train_fs, y_train)
    X_test_fs = fs.transform(X_test)
    s = dt.score(X_test_fs, y_test)
    print('The best selected: ', s)

    plt.plot(percentiles, results)
    plt.xlabel('percentiles of feature')
    plt.ylabel('accuracy')
    # plt.show()

if __name__ == '__main__':
    main()

