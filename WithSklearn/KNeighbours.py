# _*_ coding:utf-8 _*_

import numpy as np
from utils import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def loadDataSet():
    iris = load_iris()
    # print('the shape of iris: ', iris.data.shape)
    # print(iris.DESCR)   # 查看数据说明
    return iris

def KNN():
    iris = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    plotTrainingDataSet(X_train[:, 0], X_train[:, 3], y_train)

    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)

    print('Accuracy of KNClassifier: ', knc.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=iris.target_names))

if __name__ == '__main__':
    KNN()