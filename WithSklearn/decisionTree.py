# _*_ coding:utf-8 _*_

import pandas as pd
from sklearn.model_selection import train_test_split

def loadDataSet():
    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    # print(titanic.head())
    # print(titanic.info())
    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    X['age'].fillna(X['age'].mean(), inplace=True)
    return X, y

def DTree():
    data, target = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)
    


if __name__ == '__main__':
    loadDataSet()