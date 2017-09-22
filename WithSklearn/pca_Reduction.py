# _*_ coding:utf-8 _*_

import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def loadDataSet():
    digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                               header=None)
    digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                              header=None)
    X_train = digits_train[np.arange(64)]
    y_train = digits_train[64]
    X_test = digits_test[np.arange(64)]
    y_test = digits_test[64]
    return X_train, X_test, y_train, y_test

def pca_decomposition():
    X_digits, y_digits = loadDataSet()
    estimator = PCA(n_components=2)
    estimator.fit(X_digits)
    X_pca = estimator.transform(X_digits)
    print(X_pca)
    return X_pca, y_digits


def plot_pca_scatter():
    X_pca, y_digits = pca_decomposition()
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Senond Principal Component')
    plt.show()

def contrast_Predict():
    X_train, X_test, y_train, y_test = loadDataSet()

    # 使用SVC进行预测
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    y_predict = svc.predict(X_test)

    # 使用PCA降维后用SVC进行预测
    estimator = PCA(n_components=20)
    pca_X_train = estimator.fit_transform(X_train)   # 先使用X_train进行模型训练然后再降维
    pca_X_test = estimator.transform(X_test)   # 因为已经训练过了，所以直接使用模型降维

    pca_svc = LinearSVC()
    pca_svc.fit(pca_X_train, y_train)
    pca_y_predict = pca_svc.predict(pca_X_test)

    print('Linear svc: ', svc.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))

    print('-----------------------------------------------')
    print('Linear svc after pca: ', pca_svc.score(pca_X_test, y_test))
    print(classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))


if __name__ == '__main__':
    # plot_pca_scatter()
    # pca_decomposition()
    contrast_Predict()
