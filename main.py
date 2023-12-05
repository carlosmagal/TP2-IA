from knn import KNN
from kmeans import KMeans
import pandas as pd
import sys


def runKNN(train, test):
    xTrain = train.drop('TARGET_5Yrs', axis=1).values
    yTrain = train['TARGET_5Yrs'].values
    xTest = test.drop('TARGET_5Yrs', axis=1).values
    yTest = test['TARGET_5Yrs'].values

    knn = KNN(xTrain, yTrain, xTest, yTest)
    kValues = [2, 5, 10, 50, 100]

    for k in kValues:
        predictions = knn.predict(k)
        matrix, accuracy, precision, recall, f1 = knn.getMetrics(
            predictions)

        knn.printMetrics(k, matrix, accuracy, precision, recall, f1)


def runKMeans(train, test):

    X = train.drop(['TARGET_5Yrs'], axis=1).values

    normalized = (X - X.mean()) / X.std()

    kmeans = KMeans(k=2)
    kmeans.train(normalized)
    kmeans.visualizeClusters(normalized)

    xTest = test.drop(['TARGET_5Yrs'], axis=1).values
    testNormalized = (xTest - xTest.mean()) / xTest.std()
    kmeans.clusterAcc(testNormalized, test['TARGET_5Yrs'])
    
    # kmeans.visualize_clusters(normalized)


def main():
    train = pd.read_csv('nba_treino.csv')
    test = pd.read_csv('nba_teste.csv')

    if len(sys.argv) > 1:
        if sys.argv[1] == 'KNN':
            runKNN(train, test)
        elif sys.argv[1] == 'KMEANS':
            runKMeans(train, test)


if __name__ == "__main__":
    main()
