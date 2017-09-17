# _*_ coding:utf-8 _*_

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

def loadDataSet():
    digits = load_digits()
    print(digits.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

def SuppertVectorMachine():
    X_train, X_test, y_train, y_test = loadDataSet()
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    lsvc = LinearSVC()
    lsvc.fit(X_train, y_train)
    lsvc_y_predict = lsvc.predict(X_test)

    print('Accuracy of LinearSVC: ', lsvc.score(X_test, y_test))
    print(classification_report(y_test, lsvc_y_predict))

if __name__ == '__main__':
    SuppertVectorMachine()