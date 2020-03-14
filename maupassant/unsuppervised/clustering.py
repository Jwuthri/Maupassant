import pickle

import matplotlib.pyplot as plt

from hdbscan import HDBSCAN

from sklearn.cluster import KMeans, AgglomerativeClustering

from maupassant.utils import timer


class Clustering:

    def __int__(self, model_name='KMEANS'):
        self.model_name = model_name.upper()
        self.model = self.get_clustering

    @property
    def get_clustering(self):
        assert self.model_name in ['KMEANS', 'HDBSCAN', 'AHC']
        if self.model_name == 'KMEANS':
            return KMeans
        elif self.model_name == 'AHC':
            return AgglomerativeClustering
        elif self.model_name == "HDBSCAN":
            return HDBSCAN

    @timer
    def fit(self, x, n_clusters=5, **kwargs):
        if self.model_name == "HDBSCAN":
            self.model = self.model(**kwargs)
        else:
            self.model = self.model(n_clusters=n_clusters, init='k-means++', **kwargs).fit(x)

    @timer
    def predict(self, x):
        if self.model_name == "HDBSCAN":
            return self.model.fit_predict(x=x)
        else:
            return self.model.predict(x=x)

    def save(self, filename):
        pickle.dump(self.model, open(filename, "wb"))


class Elbow:

    def __int__(self):
        self.wcss = list()

    def get_optimal_n_clusters(self):
        max_clusters = len(self.wcss)
        for cluster in range(1, max_clusters - 1):
            lcurve = self.wcss[cluster + 1] + self.wcss[cluster - 1] - 2 * self.wcss[cluster]
            if lcurve < 0:
                return cluster

        return max_clusters

    @timer
    def fit(self, x, max_clusters=5):
        for i in range(max_clusters):
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(x)
            self.wcss.append(kmeans.inertia_)

        return self.wcss

    @timer
    def predict(self):
        return self.get_optimal_n_clusters()

    def plot_elbow(self):
        if bool(len(self.wcss)):
            plt.plot(len(self.wcss), self.wcss, 'bx-')
        else:
            raise Exception("Please run self.tensorboard(x) or provide wcss")
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal n_clusters')
        plt.show()


