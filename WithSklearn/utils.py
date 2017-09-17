# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

def plotTrainingDataSet(feature1, feature2, labels):
    ax.scatter(feature1, feature2, c=labels, marker='o')
    ax.set_xlabel('feature1')
    ax.set_ylabel('feature2')
    ax.set_title('Training DataSet')
    plt.show()

def plotPredict(feature1, feature2, labels, kernal):
    step = 0.01
    x_min, x_max = feature1.min() - 1, feature2.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = kernal.predict(xx, yy)
