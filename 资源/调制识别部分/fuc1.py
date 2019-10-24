import numpy as np
import operator
from collections import defaultdict
import collections
import matplotlib.pyplot as plt

#KNN的具体实现
def classify0(inX, dataset, labels, k):
    # 计算输入数据和已有所有数据的距离
    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
    distances = sqDistances ** 0.5
    # 排序
    sortedDistIndex = distances.argsort()  # argsort将数据从小到大排列，并返回其索引值
    # 选择距离最小的k个点
    classCount = {}  # 字典类型
    for i in range(k):
        votelabel = labels[sortedDistIndex[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedClasscount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasscount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,9))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:9]
        classLabelVector.append(float(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def BPSKModulationClassTest():
    sampleDataMat,sampleLabels = file2matrix('data/sample.dat')       #load data setfrom file
    SNR = [2*x for x in range(-2,6)]
    accuracy = defaultdict(list)
    correctCount = defaultdict(list)
    classifierResult = defaultdict(list)
    for snr in SNR :
        testDataMat,testLabels = file2matrix('data/testBPSK-'+str(snr)+'.dat')       #load data setfrom file
        numTestVecs = testDataMat.shape[0]
        correctCount[snr] = 0.0
        for i in range(numTestVecs):
            classifierResult[snr] = classify0(testDataMat[i,:],sampleDataMat[:,:],sampleLabels[:],10)
            #print ("the classifier came back with: %d, the real answer is: 2" % (classifierResult[snr]))
            if (classifierResult[snr] == 2): correctCount[snr] += 1.0
        accuracy[snr] = correctCount[snr]/(numTestVecs)
        print ("the total correct rate on %d dB SNR is:" % (snr), (accuracy[snr]))
        print (correctCount[snr])
    accuracy = collections.OrderedDict(sorted(accuracy.items()))  # sort by ascending SNR values

#画图
    print (accuracy)
    plt.style.use('classic')
    plt.figure(figsize=(8, 5), dpi=100)
    x = SNR
    y = list(accuracy.values())
    plt.plot(x, y, marker="o", linewidth=2.0, linestyle='dashed', color='royalblue')
    plt.axis([0, 10, 0, 1])
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.yticks(np.arange(0, 1, 0.10))

    ttl = plt.title('SNR vs Accuracy - BPSK', fontsize=16)
    ttl.set_weight('bold')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Test accuracy', fontsize=14)
    plt.grid()

    plt.show()

def QPSKModulationClassTest():
    sampleDataMat,sampleLabels = file2matrix('data/sample.dat')       #load data setfrom file
    SNR = [2*x for x in range(-2,6)]
    accuracy = defaultdict(list)
    correctCount = defaultdict(list)
    classifierResult = defaultdict(list)
    for snr in SNR :
        testDataMat,testLabels = file2matrix('data/testQPSK-'+str(snr)+'.dat')       #load data setfrom file
        numTestVecs = testDataMat.shape[0]
        correctCount[snr] = 0.0
        for i in range(numTestVecs):
            classifierResult[snr] = classify0(testDataMat[i,:],sampleDataMat[:,:],sampleLabels[:],10)
            #print ("the classifier came back with: %d, the real answer is: 4" % (classifierResult[snr]))
            if (classifierResult[snr] == 4): correctCount[snr] += 1.0
        accuracy[snr] = correctCount[snr]/(numTestVecs)
        print ("the total correct rate on %d dB SNR is:" % (snr), (accuracy[snr]))
        print (correctCount[snr])
    accuracy = collections.OrderedDict(sorted(accuracy.items()))  # sort by ascending SNR values

    print (accuracy)
    plt.style.use('classic')
    plt.figure(figsize=(8, 5), dpi=100)
    x = SNR
    y = list(accuracy.values())
    plt.plot(x, y, marker="o", linewidth=2.0, linestyle='dashed', color='#800080')
    plt.axis([0, 10, 0, 1])
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.yticks(np.arange(0, 1, 0.10))

    ttl = plt.title('SNR vs Accuracy - QPSK', fontsize=16)
    ttl.set_weight('bold')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Test accuracy', fontsize=14)
    plt.grid()

    plt.show()

