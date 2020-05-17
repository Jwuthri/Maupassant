import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn import preprocessing
from umap import UMAP

from maupassant.utils import timer


class Decomposition(object):

    def __init__(self, model='QDA', n_components=2):
        self.model = model.upper()
        self.n_components = n_components
        self.decomposition = self.get_decomposition(n_components=n_components)

    def get_decomposition(self, n_components):
        assert self.model in ["LDA", "PCA", "SVD", "SPCA", "UMAP"]
        if self.model == "LDA":
            return LinearDiscriminantAnalysis(n_components=n_components)
        elif self.model == "PCA":
            return PCA(n_components=n_components)
        elif self.model == "SVD":
            return TruncatedSVD(n_components=n_components)
        elif self.model == "SPCA":
            return SparsePCA(n_components=n_components)
        elif self.model == "UMAP":
            return UMAP(n_components=n_components)

    @timer
    def fit(self, x, y=None):
        if self.model == "UMAP":
            if y is not None:
                le = preprocessing.LabelEncoder()
                le.fit(y)
                y = le.transform(y)
            self.decomposition.fit(X=x, y=y)
        else:
            self.decomposition.fit(X=x, y=y)

    @timer
    def transform(self, x, y=None):
        if self.model == "UMAP":
            if y is not None:
                le = preprocessing.LabelEncoder()
                le.fit(y)
                y = le.transform(y)
            return self.decomposition.fit_transform(X=x, y=y)
        else:
            return self.decomposition.transform(X=x)

    def save(self, filename):
        pickle.dump(self.decomposition, open(filename, "wb"))
