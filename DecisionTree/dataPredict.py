# _*_ coding:utf-8 _*_

from trees import *
from treePlotter import *
from createDataSet import *

# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def main():
    dataSet, labels = getDataSetTwo()
    myTree = createTree(dataSet, labels)
    print(myTree)

    # dataSet, labels = getDataSetOne()
    # myTree = createTree(dataSet, labels)
    # featLabels = ['no surfacing', 'flippers']
    # predict = classify(myTree, featLabels, [1, 1])
    # print(predict)


if __name__ == '__main__':
    main()