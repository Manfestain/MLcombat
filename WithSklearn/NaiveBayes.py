# _*_ coding:utf-8 _*_

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

def loadDataSet():
    news = fetch_20newsgroups(subset='all')
    print('Data Counts: ', len(news.data))
    # print(news.data[0])
    return news

def NBayes():
    news = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
    vector = CountVectorizer()
    X_train = vector.fit_transform(X_train)
    X_test = vector.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    y_predict = mnb.predict(X_test)

    print('Accuracy of Naive Bayes: ', mnb.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == '__main__':
    NBayes()