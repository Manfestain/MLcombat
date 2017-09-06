# _*_ coding:utf-8 _*_

from kNNdating import *
from os import listdir

# 图片处理
def img2vector(filename):
    returnVect = zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])   #returnVect只有一行，所以在第0行放置
    f.close()
    return returnVect

# 数字测试代码
def handwritingClassMeasure():
    hwLabels = []
    trainingFileList  = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 处理测试数据集
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        calssifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('预测结果为：%d, 实际结果为：%d' % (calssifierResult, classNumStr))
        if (calssifierResult != classNumStr):
            errorCount += 1.0

    print('总的错误数目：%f' % errorCount)
    print('总的错误率：%f' % (errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassMeasure()