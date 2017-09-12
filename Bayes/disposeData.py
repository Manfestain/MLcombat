# _*_ coding:utf-8 _*_

import re

def loadDataList():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'sute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 文本解析
def textParse(bigString):
    # mySent = 'This book is best book on python or M.L. i have ever laid'
    returnString = re.split(r'\W*', bigString)
    return [word.lower() for word in returnString if len(word) > 2]

# 文档中所有的不重复的词列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)

# 词集模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: (%s) is not in my Vocabulary!' % word)
    return returnVec

# 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 多次出现的词可能表达其它信息
        return returnVec

# if __name__ == '__main__':
#     # text = 'Today is a good day, but ,en...,I want to sleep!'
#     # print(textParse(text))
#     emailText = open('F:\Download\machinelearninginaction\Ch04\email\ham/6.txt').read()
#     print(textParse(emailText))