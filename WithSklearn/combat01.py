# _*_ coding:utf-8 _*_

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing

def demoOne():
    iris = datasets.load_iris()
    print('Data shape: ', iris.data.shape)
    print('Target shape: ', iris.target.shape)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=10, random_state=0)
    print('TrainSet shape: ', X_train.shape)
    print('TestSet shape: ', X_test.shape)

def demoTwo():
    iris = datasets.load_iris()
    X_train = iris.data[0:5]
    print(X_train)
    X_scaled = preprocessing.scale(X_train)
    # print(X_scaled)
    print('x_scaler mean:', X_scaled.mean(axis=1))
    # print(X_scaled.std(axis=0))
    ss = preprocessing.StandardScaler()
    X_ss = ss.fit_transform(X_train)
    print(X_ss)
    print(X_ss - X_scaled)
    print('--------------------------')
    scaler = preprocessing.StandardScaler().fit(X_train)
    print(scaler.transform(X_train))
    print('scaler mean:', scaler.mean_)
    print('scaler scaler_:', scaler.scale_)
    print(ss.transform(X_train) - X_scaled)


if __name__ == '__main__':
    # demoOne()
    demoTwo()