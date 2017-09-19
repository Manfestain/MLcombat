# _*_ coding:utf-8 _*_

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def loadDataSet():
    boston = load_boston()
    X = boston.data
    y = boston.target.reshape((len(boston.target), 1))
    print(X.shape)
    print(y.shape)
    return X, y

def knnReg():
    X, y = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    ss_x = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)

    #   使用算数平均法计算回归
    uni_knr = KNeighborsRegressor(weights='uniform')
    uni_knr.fit(X_train, y_train)
    uni_knr_y_predict = uni_knr.predict(X_test)

    #   考虑距离差异进行加权平均计算回归
    dis_knr = KNeighborsRegressor(weights='distance')
    dis_knr.fit(X_train, y_train)
    dis_knr_y_predict = dis_knr.predict(X_test)

    print('R-squared of uniform KNNR: ', uni_knr.score(X_test, y_test))
    print('Mean squared error of uniform KNNR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                    ss_y.inverse_transform(uni_knr_y_predict)))
    print('Mean sbsoluate error of uniform KNNR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                        ss_y.inverse_transform(uni_knr_y_predict)))

    print('------------------------------------------------------------')
    print('R-squared of distance KNNR: ', dis_knr.score(X_test, y_test))
    print('Mean squared error of distance KNNR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                      ss_y.inverse_transform(dis_knr_y_predict)))
    print('Mean sbsoluate error of distance KNNR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                         ss_y.inverse_transform(dis_knr_y_predict)))


if __name__ == '__main__':
    knnReg()