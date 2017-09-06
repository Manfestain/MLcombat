# _*_ coding:utf-8 _*_

import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import *


# 构建分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 拿到数据集的行数

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

# 处理数据格式
def file2matrix(filename):
    f = open(filename)
    arrayOfLines = f.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelvector = []
    index = 0   # 记录行数
    for line in arrayOfLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelvector.append(int(listFormLine[-1]))   # 得到标签数据
        index += 1
    f.close()
    return returnMat, classLabelvector

# 归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 绘图
def plot(dataSet, datingLabel):
    zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\STKAITI.TTF')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 1], dataSet[:, 0], 15.0 * array(datingLabel), 15.0 * array(datingLabel))
    ax.set_xlabel(u'每年获得的飞行常客里程数', fontproperties=zhfont1)
    ax.set_ylabel(u'玩视频游戏所消耗的时间比', fontproperties=zhfont1)
    plt.show()
    return None

# 测试数据
def datingClassMeasure():
    hoRatio = 0.10
    datingDataMat, datingLabel = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)   # 测试数据集
    errorCount = 0.0

    for i in range(numTestVecs):
        classifiesResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabel[numTestVecs:m], 3)
        print('预测结果为：%d, 实际类别：%d' % (classifiesResult, datingLabel[i]))
        if (classifiesResult != datingLabel[i]):
            errorCount += 1.0

    print('分类器总的错误率：%f' % (errorCount/float(numTestVecs)))

# 使用
def calssifyPerson():
    resultList = ['not at all', 'in small dose', 'in large does']
    game = float(input('percentage of time spend play games?'))
    fly = float(input('plane miles?'))
    ice = float(input('ice cream consumed?'))
    datingDataMat, datingLabel = file2matrix('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([fly, game, ice])
    classifierResult = classify0((inArr-minVals)/ranges, norMat, datingLabel, 3)
    print('你对这位先生的喜爱程度：', (resultList[classifierResult -1]))

if __name__ == '__main__':
    calssifyPerson()
