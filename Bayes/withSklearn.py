# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from disposeData import *
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def demoOne():
    X = np.array([[-1, -1],
                  [-2, -1],
                  [-3, -2],
                  [1, 1],
                  [2, 1],
                  [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = GaussianNB(priors=None)
    clf.fit(X, y)
    print(clf.predict([[-0.8, -1]]))
    print('predict_prob: ', clf.predict_proba([[-0.8, -1]]))
    print('predict_log_prob: ', clf.predict_log_proba([[-0.8, -1]]))
    print(clf.score([[-0.8, -1]], clf.predict([[-0.8, -1]])))
    print(clf.partial_fit(X, y, classes=np.unique(y)))
    print(clf.set_params())
    return X, y

def demoTwo():
    docList = []
    classList = []
    hamfilepath = 'F:\Download\machinelearninginaction\Ch04\email\ham/'
    spamfilepath = 'F:\Download\machinelearninginaction\Ch04\email\spam/'

    for i in range(1, 26):
        wordList = textParse(open(hamfilepath + '%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
        wordList = textParse(open(spamfilepath + '%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)

    vocabList = createVocabList(docList)
    trainSet = range(30)
    testSet = range(30, 50)
    trainMat = []
    trainClasses = []

    for docIndex in trainSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    errorCount = 0
    clf = BernoulliNB()
    clf.fit(trainMat, trainClasses)
    # print(clf)
    # print(clf.predict([bagOfWords2VecMN(vocabList, docList[40])]))

    for docIndex in testSet:
        wordVector = [bagOfWords2VecMN(vocabList, docList[docIndex])]
        if clf.predict(wordVector) != classList[docIndex]:
            errorCount += 1

    print('the error rate: ', float(errorCount/len(testSet)))

def demoThree():
    plt.figure()
    plt.title('Learning Curves (Navive Bayes)')
    plt.xlabel('Training example')
    plt.ylabel('Score')
    digits = load_digits()
    X, y = digits.data, digits.target

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = GaussianNB()
    n_jobs = 4
    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # demoOne()
    # demoTwo()
    demoThree()

