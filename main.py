from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn import tree

import graphviz 

def SampleDataSet(dataSeq, labelSeq):
    trainData = []
    trainLabels = []
    for i in range(len(dataSeq)):
        if i % 2 == 0:
            trainData.append(dataSeq[i])
            trainLabels.append(labelSeq[i])
    return trainData, trainLabels

def treeTester(X, Y):
    trainData, trainLabels = SampleDataSet(X, Y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainData, trainLabels)
    errCnt = 0
    for i, pred in enumerate(clf.predict(X)):
        if not (pred == Y[i]):
            errCnt += 1
    print("Error Rate: {0} %".format(100*errCnt/len(irisDataSet["data"])))
    graphData = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(graphData)
    graph.render()
    return

if __name__ == '__main__':
    irisDataSet = load_iris()
    treeTester(irisDataSet["data"], irisDataSet["target"])

    # trainData, trainLabels = SampleDataSet(irisDataSet["data"], irisDataSet["target"])

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(trainData, trainLabels)
    # errCnt = 0
    # for i, pred in enumerate(clf.predict(irisDataSet["data"])):
    #     if not (pred == irisDataSet["target"][i]):
    #         errCnt += 1
    #     #print("{0}: {1} - {2} -- {3}".format(i, pred, irisDataSet["target"][i], pred == irisDataSet["target"][i]))
    # print("Error Rate: {0} %".format(100*errCnt/len(irisDataSet["data"])))
