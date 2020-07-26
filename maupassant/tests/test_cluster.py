from sklearn.datasets import make_classification

from maupassant.cluster.clustering import Clustering, Elbow


nsamples = 500
data = make_classification(n_samples=nsamples, n_classes=8, n_informative=5, n_clusters_per_class=1, random_state=42)
x = data[0]
elbow = Elbow()
elbow.fit_model(x)
n_clusters = elbow.predict()


class TestClustering(object):

    def test_elbow(self):
        assert n_clusters < 14

    def test_kmeans(self):
        c = Clustering(model_name='kmeans')
        c.fit_model(x, n_clusters)
        assert nsamples == len(c.predict(x))

    def test_hdbscan(self):
        c = Clustering(model_name='hdbscan')
        c.fit_model(x, n_clusters)
        assert nsamples == len(c.predict(x))

    def test_ahc(self):
        c = Clustering(model_name='ahc')
        c.fit_model(x, n_clusters)
        assert nsamples == len(c.predict(x))
