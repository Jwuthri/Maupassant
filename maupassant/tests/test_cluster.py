from sklearn.datasets import make_classification

from maupassant.cluster.clustering import Clustering, Elbow


class TestClustering:

    def __init__(self):
        self.data = make_classification(n_samples=1000, n_classes=4, n_clusters_per_class=1, random_state=42)
        elbow = Elbow()
        elbow.fit(self.data[0])
        self.n_clusters = elbow.predict()

    def test_elbow(self):
        assert self.n_clusters in [4, 5, 6]


    def test_kmeans(self):
        c = Clustering(model_name='kmeans')

    def test_hdbscan(self):
        pass

    def test_ahc(self):
        pass
