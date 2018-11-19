from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

#import graphviz 

def SampleDataSet(dataSeq, labelSeq):
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    for i in range(len(dataSeq)):
        if i % 2 == 0:
            trainData.append(dataSeq[i])
            trainLabels.append(labelSeq[i])
        else:
            testData.append(dataSeq[i])
            testLabels.append(labelSeq[i])
    return trainData, trainLabels, #testData, testLabels

def treeTester(X, Y):
    # todo: add boosting
    trainData, trainLabels = SampleDataSet(X, Y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainData, trainLabels)
    errCnt = 0
    for i, pred in enumerate(clf.predict(X)):
        if not (pred == Y[i]):
            errCnt += 1
    printErrRate(errCnt, len(X))
    # graphData = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(graphData)
    # graph.render()
    return

def kNNTester(X,Y):
    trainData, trainLabels = SampleDataSet(X, Y)
    clf = KNeighborsClassifier(n_neighbors=len(set(Y))+1)
    clf.fit(trainData, trainLabels)
    errCnt = 0
    for i, pred in enumerate(clf.predict(X)):
        if not (pred == Y[i]):
            errCnt += 1
    printErrRate(errCnt, len(X))

def MLPTester(X,Y):
    trainData, trainLabels = SampleDataSet(X, Y)
    clf = MLPClassifier(
        solver='lbfgs', 
        alpha=1e-5, 
        hidden_layer_sizes=(3,8), 
        random_state=1)
    clf.fit(trainData, trainLabels)
    errCnt = 0
    for i, pred in enumerate(clf.predict(X)):
        if not (pred == Y[i]):
            errCnt += 1
    printErrRate(errCnt, len(X))

def SVMTester(X,Y):
    # todo: try to add two kernals
    trainData, trainLabels = SampleDataSet(X, Y)
    clf = svm.SVC()
    clf.fit(trainData, trainLabels)
    errCnt = 0
    for i, pred in enumerate(clf.predict(X)):
        if not (pred == Y[i]):
            errCnt += 1
    printErrRate(errCnt, len(X))

def GBCTester(X,Y):
    trainData, trainLabels = SampleDataSet(X, Y)
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=1.0,
        max_depth=1, 
        random_state=0
    )
    clf.fit(trainData, trainLabels)
    errCnt = 0
    for i, pred in enumerate(clf.predict(X)):
        if not (pred == Y[i]):
            errCnt += 1
    printErrRate(errCnt, len(X))

def printErrRate(errCnt, n):
    print("Error Rate: {0} %".format(100*errCnt/n))

if __name__ == '__main__':
    irisDataSet = load_iris()
    print("Tree")
    treeTester(irisDataSet["data"], irisDataSet["target"])
    print("kNN")
    kNNTester(irisDataSet["data"], irisDataSet["target"])
    print("MLP")
    MLPTester(irisDataSet["data"], irisDataSet["target"])
    print("SVM / SVC")
    SVMTester(irisDataSet["data"], irisDataSet["target"])
    print("GBC")
    GBCTester(irisDataSet["data"], irisDataSet["target"])
