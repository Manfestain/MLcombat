# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
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
    print('x_min', x_min)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1))
    print('xx', xx)
    print(xx.shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = 15, weights = 'distance')")
    plt.show()

if __name__ == '__main__':
    demoOne()