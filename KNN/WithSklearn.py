# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from kNNdating import *
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification

def demoOne():
    # 随机生成数据集
    X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, n_classes=3)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
    # plt.show()

    # 使用KNN分类
    clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
    clf.fit(X, Y)

    # 可视化分类结果
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    print(xx.shape)
    print(yy.shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print(set(Z))
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = 15, weights = 'distance')")
    plt.show()

def demoKneighbors():
    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    neigh = neighbors.NearestNeighbors(n_neighbors=2)
    neigh.fit(samples)
    print(neigh.kneighbors([[1., 1., 1.]]))
    print(neigh.kneighbors_graph([[1., 1., 1.]]))

def demoThree():
    datingMat, labels = file2matrix('datingTestSet2.txt')
    trainMat, ranges, minVals = autoNorm(datingMat)
    neigh = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    neigh.fit(trainMat, labels)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(trainMat[:, 0], trainMat[:, 1], trainMat[:, 2], c=labels, marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()


    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = trainMat[0, :].min() - 1, trainMat[0, :].max() + 1
    y_min, y_max = trainMat[1, :].min() - 1, trainMat[1, :].max() + 1
    z_min, z_max = trainMat[2, :].min() - 1, trainMat[2, :].max() + 1

    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1),
                             np.arange(z_min, z_max, 0.1))

    result = neigh.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    result = result.reshape(xx.shape)

    fig = plt.figure()
    # plt.pcolormesh(xx, yy, result, cmap=cmap_light)
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(xx, yy, zz, c=result)
    plt.show()


if __name__ == '__main__':
    # demoOne()
    demoKneighbors()
    # demoThree()