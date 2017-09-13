# _*_ coding:utf-8 _*_

import numpy as np
from logRegres import *

def loadDataSet():
    frTrain = open('F:\Download\machinelearninginaction\Ch05/horseColicTraining.txt')
    frTest = open('F:\Download\machinelearninginaction\Ch05/horseColicTest.txt')
    trainSet = [] ; trainLabels = []
    testSet = [] ; testLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(currLine[-1]))

    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))

    return trainSet, trainLabels, testSet, testLabels

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def validateData(trainMat, trainLabels, testMat, testLabels):
    train_weights, w_all = improved_stocGradAscent(trainMat, trainLabels)
    # train_weights = np.array(train_weights).reshape((len(train_weights), 1))
    errorCount = 0
    numTestVec = 0.0
    for i in range(testMat.shape[0]):
        if int(classifyVector(testMat[i, :], train_weights)) != int(testLabels[i]):
            errorCount += 1
    print('the error rate: ', float(errorCount/testMat.shape[0]))

if __name__ == '__main__':
    trainMat, trainLabels, testSet, testLabels = loadDataSet()
    validateData(np.array(trainMat), np.array(trainLabels).reshape((len(trainLabels), 1)),
                 np.array(testSet), np.array(testLabels).reshape((len(testLabels), 1)))

