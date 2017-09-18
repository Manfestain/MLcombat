# _*_ coding:utf-8 _*_
'''
    对于回归问题，在性能评估时和分类问题不同
    不能使用预测值和真实值之间的误差进行衡量
    所以引入了平均绝对误差、均方误差和R-squared
'''

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def loadDataSet():
    boston = load_boston()
    # print(boston.DESCR)
    # print(boston.target)
    return boston

def lineaReg():
    boston = loadDataSet()
    X = boston.data
    y = boston.target.reshape((len(boston.target), 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    print('The max target value is: ', np.max(boston.target))
    print('The min target value is: ', np.min(boston.target))
    print('The average target value is: ', np.mean(boston.target))

    ss_x = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_y_predict = lr.predict(X_test)

    sgdr = SGDRegressor()
    sgdr.fit(X_train, y_train)
    sgdr_y_predcit = sgdr.predict(X_test)

    # 回归问题的评估方法
    print('dafault value of LR: ', lr.score(X_test, y_test))
    print('R-squared of LR: ', r2_score(y_test, lr_y_predict))
    print('Mean squared error of LR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                           ss_y.inverse_transform(lr_y_predict)))
    print('Mean absoluate of LR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                        ss_y.inverse_transform(lr_y_predict)))

    print('------------------------------------------------------------------')
    print('dafault value of SGDR: ', sgdr.score(X_test, y_test))
    print('R-squared of SGDR: ', r2_score(y_test, sgdr_y_predcit))
    print('Mean squared error of SGDR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                             ss_y.inverse_transform(sgdr_y_predcit)))
    print('Mean absoluate of SGDR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                          ss_y.inverse_transform(sgdr_y_predcit)))
    return None


if __name__ == '__main__':
    lineaReg()