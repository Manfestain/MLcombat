# _*_ coding:utf-8 _*_

import random
from disposeData import *
from bayes import *

def spamText():
    docList = []
    classList = []
    fullText = []
    hamfilename = 'F:\Download\machinelearninginaction\Ch04\email\ham/'
    spamfilename = 'F:\Download\machinelearninginaction\Ch04\email\spam/'

    for i in range(1, 26):
        wordList = textParse(open(hamfilename + '%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)   # 表示正常言论

        wordList = textParse(open(spamfilename + '%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)   # 表示侮辱性言论

    vocabList = createVocabList(docList)
    trainingSet = range(40)
    testSet = range(40, 50)
    print(trainingSet)
    print(testSet)

    # 随机选出十组测试集
    # for i in range(10):
    #     randIndex = int(random.uniform(0, len(trainingSet)))
    #     testSet.append(trainingSet[randIndex])
    #     del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    # 初始化训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 训练算法
    p0V, p1V, pAb = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    # 测试算法
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pAb) != classList[docIndex]:
            errorCount += 1

    print('the error rate is: ', float(errorCount/len(testSet)))

if __name__ == '__main__':
    spamText()
