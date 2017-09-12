# _*_ coding:utf-8 _*_

from math import log
from numpy import *
from disposeData import *

# 朴素贝叶斯分类器训练
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 所有文档中整条文档是侮辱文字的概率
    p0Denom = 2.0   # 记录总的所有文字出现的总次数
    p1Denom = 2.0
    p0Num = ones(numWords)   # 每行的数字表示了所有的字符，直接对应位相加就可得到每个字符出现的次数
    p1Num = ones(numWords)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 满足条件时，说明该行是侮辱性言论
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)   # 侮辱性言语中各个词你出现的总次数
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify * p1Vec) + log(pClass)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass)
    if p1 > p0:
        return 1
    else:
        return 0

def main():
    listOfPosts, listClasses = loadDataList()
    myVocabList = createVocabList(listOfPosts)
    # print(myVocabList)
    result = setOfWords2Vec(myVocabList, listOfPosts[0])
    # print(result)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    pV0, pV1, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(pV0)
    testEntry = ['I', 'do', 'a', 'wrong', 'thing']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(thisDoc)
    print('classified as: ', classifyNB(thisDoc, pV0, pV1, pAb))

if __name__ == '__main__':
    main()