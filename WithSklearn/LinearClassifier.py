# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report

def loadDataSet():
    cloumn_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape', 'Marginal Adhesion','Single Epithelial Sell Size',
                    'Bare Nuclei', 'Bland Chromatin', 'Normal Nuceoli', 'Mistoses', 'Class']
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/' +
                       'breast-cancer-wisconsin.data', names=cloumn_names)
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')
    return data, cloumn_names

def LinerC(dataSet, column_names):
    X_train, X_test, y_train, y_test = train_test_split(dataSet[column_names[1:10]], dataSet[column_names[10]],test_size=0.25, random_state=33)
    # print(y_train.value_counts())
    # print(y_test.value_counts())
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)  # 标准化数据
    X_test = ss.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_y_predict = lr.predict(X_test)

    sgdc = SGDClassifier()
    sgdc.fit(X_train, y_train)
    sgdc_y_predict = sgdc.predict(X_test)

    print('Accuracy of LR Classifier: ', lr.score(X_test, y_test))
    print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

    print('Accuracy of SGS Classifier: ', sgdc.score(X_test, y_test))
    print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))


if __name__ == '__main__':
    dataSet, column_names = loadDataSet()
    LinerC(dataSet, column_names)