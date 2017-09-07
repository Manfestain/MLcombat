# _*_ coding:utf-8 _*_

import pickle
from trees import *
from dataPredict import *

def storeTree(myTree, filename):
    fw = open(filename, 'w')
    pickle.dump(myTree, fw)
    fw.close()

def grabTree(filaname):
    fr = open(filaname)
    return pickle.load(fr)

def main():
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    inputTree = str(myTree)
    storeTree(inputTree, 'classifierStorage.txt')

if __name__ == '__main__':
    main()