from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


class KMeans:
    def __init__(self, k=2, tolerance=1e-4):
        self.k = k
        self.iterations = 100
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def _calculateDist(self, p1, p2):
        distance = 0.0
        # for i in range(len(p1)):
        for i in range(len(p1) - 1):
            distance += (p1[i] - p2[i])**2
            # distance += (point1[i] - point2[i])
        return sqrt(distance)

    def _initializeCentroids(self, data):
        np.random.seed(42)
        indices = np.random.choice(len(data), self.k, replace=False)
        return data[indices]

    def _assignDataToClusters(self, data):
        clusters = np.zeros(len(data))
        for i, point in enumerate(data):
            distances = [self._calculateDist(
                point, centroid) for centroid in self.centroids]
            clusters[i] = np.argmin(distances)
        return clusters

    def _updateCentroids(self, data):
        newCentroids = np.zeros((self.k, data.shape[1]))
        for c in range(self.k):
            cluster = data[self.labels == c]
            if len(cluster) > 0:
                newCentroids[c] = np.mean(cluster, axis=0)
        return newCentroids

    def _converged(self, previousCentroids):
        return np.sum(np.abs(self.centroids - previousCentroids)) < self.tolerance

    def train(self, data):
        self.centroids = self._initializeCentroids(data)

        for _ in range(self.iterations):
            aux = self.centroids.copy()
            self.labels = self._assignDataToClusters(data)
            self.centroids = self._updateCentroids(data)
            if self._converged(aux):
                break

    def predict(self, data):
        return self._assignDataToClusters(data)

    def visualizeClusters(self, data):
        plt.scatter(data[:, 0], data[:, 1], c=self.labels,
                    cmap='Set1', alpha=0.7)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    marker='*', s=100, c='blue', label='Centroides')
        plt.title(f'Agrupamento para k = {self.k}')
        plt.legend()
        plt.show()

    # def visualizeHist(self, data):

    #     labels = self.predict(data)

    #     df = pd.DataFrame({'Data': data[:, 0], 'Cluster': labels})

    #     sns.set(style="whitegrid")

    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(df, x='Data', hue='Cluster', bins=20,
    #                  palette='Set1', multiple='stack', edgecolor='black')
    #     plt.title('Clusters')
    #     plt.xlabel('Valores')
    #     plt.ylabel('Frequência')
    #     plt.show()

    def clusterAcc(self, tData, trueLabels):
        tLabels = self.predict(tData)
        df = pd.DataFrame(
            {'Cluster': tLabels, 'TARGET_5Yrs': trueLabels})
        stats = df.groupby(
            'Cluster')['TARGET_5Yrs'].value_counts(normalize=True).unstack()
        print(stats)
