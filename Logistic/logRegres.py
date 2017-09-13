# _*_ coding:utf-8 _*_

import random
import numpy as np
import matplotlib.pyplot as plt

def loadDateSet():
    dataMat = []
    labelMat = []
    fr = open('F:\Download\machinelearninginaction\Ch05/testSet.txt')
    for line in fr.readlines():
        words = line.strip().split('\t')
        dataMat.append([1.0, float(words[0]), float(words[1])])  # 为了便于计算，将dataMat[0]填充1.0
        labelMat.append(int(words[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return float(1.0/(1 + np.exp(-inX)))

# 梯度上升算法
def gradAscent(dataMat, labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    m, n = dataMat.shape
    step = 0.001
    maxCycle = 300
    weights = np.ones((n, 1))
    for i in range(maxCycle):
        h = sigmoid(dataMat * weights)   # 当二者均为matrix时， 就是矩阵乘法，可以使用multipl做矩阵点乘
        error = labelMat - h
        weights = weights + step * dataMat.transpose() * error
    return weights

# 随机梯度上升算法
def stocGradSscent(dataMat, labelMat):
    m, n = np.shape(dataMat)
    step = 0.01
    w_all = []
    weights = np.ones(n)
    for j in range(200):
        for i in range(m):
            h = sigmoid(sum(dataMat[i] * weights))
            error = labelMat[i] - h
            weights = weights + step * error * dataMat[i]
            w_all.append(weights)
    return weights, w_all

# 改进的随机梯度上升算法
def improved_stocGradAscent(dataMat, labelMat, numIter=200):
    w_all = []
    m, n = np.shape(dataMat)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))   # python3.x的range返回的是一个对象，不能使用del()直接删除
        for i in range(m):
            step = 4/(1.0 + i + j) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = labelMat[randIndex] - h
            weights = weights + step * error * dataMat[randIndex]
            w_all.append(weights)
            del(dataIndex[randIndex])
    return weights, w_all

# 画出训练集和训练结果的最佳回归拟合
def plotDataSet(dataMat, labelMat, weights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 1], dataMat[:, 2], c=labelMat)
    ax.set_title('trainSet plot')

    x_min, x_max = dataMat[:, 1].min() - 1, dataMat[:, 1].max() + 1
    lx = np.arange(x_min, x_max, 0.01)
    ly = (-weights[0, 0] - lx * weights[1, 0]) / weights[2, 0]   # 此处需要一点推倒
    ax.scatter(lx, ly, c='r', marker='_')

    plt.show()

# 绘制回归算法的学习曲线
def plotLearnCurves():
    dataMat, labelMat = loadDateSet()
    # weights, w_all = stocGradSscent(np.array(dataMat), labelMat)
    weights, w_all = improved_stocGradAscent(np.array(dataMat), labelMat)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    w_all = np.array(w_all)
    x = np.arange(w_all.shape[0])
    ax1.plot(x, w_all[:, 0], linewidth=0.5)
    ax1.set_ylabel('x1')
    ax2 = fig.add_subplot(312)
    ax2.plot(x, w_all[:, 1], linewidth=0.5)
    ax2.set_ylabel('x2')
    ax3 = fig.add_subplot(313)
    ax3.plot(x, w_all[:, 2], linewidth=0.5)
    ax3.set_ylabel('x3')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDateSet()
    # weights = gradAscent(dataMat, labelMat)
    # weights = improved_stocGradAscent(np.array(dataMat), labelMat)
    # weights = np.array(weights).reshape(len(weights), 1)
    plotLearnCurves()
    # plotDataSet(np.array(dataMat), labelMat, weights)