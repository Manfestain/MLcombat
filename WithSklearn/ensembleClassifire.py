# _*_ coding:utf-8 _*_
'''
    使用随机森林分类器和梯度提升决策树
    二者属于集成模型，可以更好的做出决策分类
    通常情况下，随机森林分类器被作为基线系统和新提出的模型做比较
'''

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

def loadDataSet():
    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    X['age'].fillna(X['age'].mean(), inplace=True)
    return X, y

def contrastFunc():
    X, y = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc_y_predict = dtc.predict(X_test)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_y_predict = rfc.predict(X_test)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    gbc_y_predict = gbc.predict(X_test)

    print('Accuracy of DecisionTree: ', dtc.score(X_test, y_test))
    print(classification_report(dtc_y_predict, y_test))

    print('----------------------------------------------')
    print('Accuracy of RandomForest: ', rfc.score(X_test, y_test))
    print(classification_report(rfc_y_predict, y_test))

    print('-----------------------------------------------')
    print('Accuracy of Gradient Tree:', gbc.score(X_test, y_test))
    print(classification_report(gbc_y_predict, y_test))


if __name__ == '__main__':
    contrastFunc()