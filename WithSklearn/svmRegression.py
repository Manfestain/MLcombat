# _*_ coding:utf-8 _*_

from SupportVectorClassifier import loadDataSet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def svmReg():
    X_train, X_test, y_train, y_test = loadDataSet()
    y_train = y_train.reshape((len(y_train), 1))
    y_test = y_test.reshape((len(y_test), 1))

    ss_x = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)

    #   线性核函数
    linear_svr = SVR(kernel='linear')
    linear_svr.fit(X_train, y_train)
    linear_svr_y_predict = linear_svr.predict(X_test)

    #   多项式核函数
    poly_svr = SVR(kernel='poly')
    poly_svr.fit(X_train, y_train)
    poly_svr_y_predict = poly_svr.predict(X_test)

    #   径向基核函数
    rbf_svr = SVR(kernel='rbf')
    rbf_svr.fit(X_train, y_train)
    rbf_svr_y_predict = rbf_svr.predict(X_test)

    print('R-squared of LSVR: ', linear_svr.score(X_test, y_test))
    print('Mean squared error of LSVR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                             ss_y.inverse_transform(linear_svr_y_predict)))
    print('Mean sbsoluate error of LSVR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                ss_y.inverse_transform(linear_svr_y_predict)))

    print('------------------------------------------------------------')
    print('R-squared of PolySVR: ', poly_svr.score(X_test, y_test))
    print('Mean squared error of PolySVR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                ss_y.inverse_transform(poly_svr_y_predict)))
    print('Mean sbsoluate error of PolySVR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                   ss_y.inverse_transform(poly_svr_y_predict)))

    print('------------------------------------------------------------')
    print('R-squared of RBFSVR: ', rbf_svr.score(X_test, y_test))
    print('Mean squared error of RBFSVR: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                               ss_y.inverse_transform(rbf_svr_y_predict)))
    print('Mean sbsoluate error of RBFSVR: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                  ss_y.inverse_transform(rbf_svr_y_predict)))


if __name__ == '__main__':
    svmReg()
