# _*_ coding:utf-8 _*_

import graphviz
import numpy as np
import matplotlib.pyplot as plt
from createDataSet import *
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def getDataSet():
    dataSet = [[1, 1],
               [1, 1],
               [1, 0],
               [1, 0],
               [0, 1]]
    labels = ['yes', 'yes', 'no', 'no', 'no']
    return dataSet, labels

def treePlot(clf):
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("treeOne")

def demoOne():
    dataSet, labels = getDataSet()
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(dataSet, labels)
    treePlot(clf)
    print(clf.tree_.max_depth)
    print(clf.decision_path([[0, 0]]))
    print(clf.get_params())
    print(clf.predict_proba([[0, 0]]))

def demoTwo():
    iris = load_iris()
    clf = DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    print(graph)

def demoThree():
    n_classes = 3
    plot_color = 'bry'
    plot_step = 0.02

    X = load_iris().data[:, 1:3]
    print(X.shape)

    y = load_iris().target[:]
    print(set(y))
    clf = DecisionTreeClassifier().fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    result = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    result = result.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xx, yy, c=result)
    for i, color in zip(range(n_classes), plot_color):
        idx = np.where(y == i)   # 当y==i时，返回此时的坐标信息
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=load_iris().target_names[i],
                    cmap=plt.cm.Paired)
    plt.show()


if __name__ == '__main__':
    # demoOne()
    demoTwo()
    # demoThree()