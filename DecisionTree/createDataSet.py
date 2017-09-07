# _*_ coding:utf-8 _*_

# 处理数据集
def getDataSetOne():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [1, 0, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def getDataSetTwo():
    f = open('lenses.txt')
    lenses = [inst.split('\t') for inst in f.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lensesLabels