# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

def regulation(regular, X_train, y_train, X_test, y_test):
    poly4 = regular
    poly4.fit(X_train, y_train)
    print(poly4.score(X_test, y_test))
    print(poly4.coef_)

def linearR_Fitting():
    # 使用线性回归模型
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    xx = np.linspace(0, 26, 100)
    xx = xx.reshape(xx.shape[0], 1)
    yy = regressor.predict(xx)
    print('Accuracy of LinearRegressor: ', regressor.score(X_train, y_train))
    print('Predict: ', regressor.score(X_test, y_test))

    # 使用二次多项式回归模型
    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train)
    X_test_poly2 = poly2.transform(X_test)
    regressor_poly2 = LinearRegression()
    regressor_poly2.fit(X_train_poly2, y_train)
    xx_poly2 = poly2.transform(xx)
    yy_poly2 = regressor_poly2.predict(xx_poly2)
    print('Accuracy of 2Degree LinearR: ', regressor_poly2.score(X_train_poly2, y_train))
    print('Predict: ', regressor_poly2.score(X_test_poly2, y_test))


    # 使用四次多项式进行线性拟合
    poly4 = PolynomialFeatures(degree=4)
    X_train_poly4 = poly4.fit_transform(X_train)
    X_test_poly4 = poly4.fit_transform(X_test)
    regressor_poly4 = LinearRegression()
    regressor_poly4.fit(X_train_poly4, y_train)
    xx_poly4 = poly4.transform(xx)
    yy_poly4 = regressor_poly4.predict(xx_poly4)
    print('Accuracy of 4Degree LinearR: ', regressor_poly4.score(X_train_poly4, y_train))
    print('Predict: ', regressor_poly4.score(X_test_poly4, y_test))

    print('使用L1正则化：')
    regular = Lasso()
    regulation(regular, X_train_poly4, y_train, X_test_poly4, y_test)

    print('使用L2正则化：')
    regular = Ridge()
    regulation(regular, X_train_poly4, y_train, X_test_poly4, y_test)

    # 画图
    plt.scatter(X_train, y_train)
    plt1, = plt.plot(xx, yy, label='Degree=1')
    plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
    plt4, = plt.plot(xx, yy_poly4, label='Degree=4')
    plt.axis([0, 25, 0, 25])
    plt.xlabel('Diaeter of Pizza')
    plt.ylabel('Price of Pizza')
    plt.legend(handles=[plt1, plt2, plt4])
    plt.show()

if __name__ == '__main__':
    linearR_Fitting()