# _*_ coding:utf-8 _*_

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def loadDataSet():
    boston = load_boston()
    X = boston.data
    y = boston.target
    return X, y

def decisionTreeReg():
    X, y = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    ss_x = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)
    y_train = ss_y.fit_transform(y_train.reshape((len(y_train), 1)))
    y_test = ss_y.transform(y_test.reshape(len(y_test), 1))

    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    dtr_y_predicrt = dtr.predict(X_test)

    print('R-squared of decision tree: ', dtr.score(X_test, y_test))
    print('Mean squared error of decision tree: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                     ss_y.inverse_transform(dtr_y_predicrt)))
    print('Mean sbsoluate error of decision tree: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                        ss_y.inverse_transform(dtr_y_predicrt)))


if __name__ == '__main__':
    decisionTreeReg()