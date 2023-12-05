import numpy as np
from math import sqrt
from sklearn.metrics import confusion_matrix


class KNN:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest

    def _calculateDist(p1, p2):
        distance = 0.0
        # for i in range(len(p1)):
        for i in range(len(p1) - 1):
            distance += (p1[i] - p2[i])**2
            # distance += (point1[i] - point2[i])
        return sqrt(distance)

    @staticmethod
    def Knn(xTrain, yTrain, xTest, k):
        distances = [KNN._calculateDist(xTest, x) for x in xTrain]
        # distancesArr = getEuclidianDist(xTest)
        kIndex = np.argsort(distances)[:k]
        Klabels = [yTrain[i] for i in kIndex]
        # Klabels = list(map(lambda i: yTrain[i], kIndex))

        return Klabels

    @staticmethod
    def majorityVote(labels):
        l, counts = np.unique(labels, return_counts=True)
        return l[np.argmax(counts)]

    def predict(self, k):
        predictions = [self.majorityVote(self.Knn(
            self.xTrain, self.yTrain, x, k)) for x in self.xTest]
        return predictions

    @staticmethod
    def calcAcc(matrix):
        tn, fp, fn, tp = matrix.ravel()

        accDenominator = tp + \
            tn + fp + fn
        pDenominator = tp + fp
        rDenominator = tp + fn

        accuracy = (tp + tn) / \
            accDenominator if accDenominator != 0 else 0
        precision = tp / pDenominator if pDenominator != 0 else 0
        recall = tp / rDenominator if rDenominator != 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) != 0 else 0

        return accuracy, precision, recall, f1

    def getMetrics(self, pred):
        matrix = confusion_matrix(self.yTest, pred)
        accuracy, precision, recall, f1 = KNN.calcAcc(
            matrix)
        return matrix, accuracy, precision, recall, f1

    @staticmethod
    def printMetrics(k, matrix, accuracy, precision, recall, f1):
        print("\nk = ", k)
        print("Matriz de Confusão:\n", matrix)
        print("Acurácia: ", accuracy * 100)
        print("Precisão: ", precision * 100)
        print("Revocação: ", recall * 100)
        print("F1: ", f1 * 100)
