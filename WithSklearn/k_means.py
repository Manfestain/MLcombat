# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

def loadDataSet():
    digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
    digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

    X_train = digits_train[np.arange(64)]
    y_train = digits_train[64]
    X_test = digits_test[np.arange(64)]
    y_test = digits_test[64]

    return X_train, X_test, y_train, y_test

def kMeans_ARI():
    X_train, X_test, y_train, y_test = loadDataSet()

    km = KMeans(n_clusters=10)   # 设置初始聚类中心数量为10
    km.fit(X_train)
    y_pred = km.predict(X_test)
    print(km.labels_)

    # 使用ARI进行性能评估
    print('ARI: ', metrics.adjusted_rand_score(y_test, y_pred))

def analysisOfCS():
    plt.subplot(321)
    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    X = np.array(list(zip(x1, x2)))
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('Instances')
    plt.scatter(x1, x2)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

    clusters = [2, 3, 4, 5, 8]
    subplot_counter = 1
    sc_scores = []
    for t in clusters:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter)
        kmeans_model = KMeans(n_clusters=t).fit(X)

        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
            plt.xlim([0, 10])
            plt.ylim([0, 10])
            sc_score = metrics.silhouette_score(X, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
        plt.title('K=%s, silhouette coefficient=%0.03f' % (t, sc_score))

    plt.figure()
    plt.plot(clusters, sc_scores, '*-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient Score')

    plt.show()
    # print(KMeans(n_clusters=t).fit(X).labels_)

def elbowObserve():
    cluster1 = np.random.uniform(0.5, 1.5, (2, 10))   # 产生两行十列的随机矩阵
    cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
    cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
    X = np.hstack((cluster1, cluster2, cluster3)).T   # 水平堆叠然后再转置
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'),
                                   axis=1))/X.shape[0])
        # print('%d' % k, cdist(X, kmeans.cluster_centers_, 'euclidean'))

    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Average Dispersion')
    plt.title('Selecting K with the Elbow Method')
    plt.show()


if __name__ == '__main__':
    # analysisOfCS()
    # kMeans_ARI()
    elbowObserve()