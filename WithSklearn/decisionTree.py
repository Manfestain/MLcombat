# _*_ coding:utf-8 _*_

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

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
    print(X_train[:10])
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    # print(vec.feature_names_)
    print(X_train[:10])
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_predict = dtc.predict(X_test)

    print('Accuracy of DTtee: ', dtc.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=['died', 'survived']))


if __name__ == '__main__':
    DTree()