# _*_ coding:utf-8 _*_

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Imputer

def loadData():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def dataProcessing():
    X, y = loadData()
    #   无量纲化
    dataS = StandardScaler().fit_transform(X)   # 标准化法
    dataM = MinMaxScaler().fit_transform(X)   # 区间缩放法
    print(dataS[:3])
    print('----------------------------------------')
    print(dataM[:3])

    #   哑编码
    print(y[:5])
    dataO = OneHotEncoder().fit_transform(y.reshape((-1, 1)))
    print(dataO[:5])

    # 缺失值
    data = np.vstack((np.array([np.NaN, np.NaN, np.NaN, np.NaN]), X))
    print(data[-5:])
    print('--------------------------')
    dataI = Imputer(strategy='mean').fit_transform(data)
    print(dataI[-5:])


if __name__ == '__main__':
    dataProcessing()