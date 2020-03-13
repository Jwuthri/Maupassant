import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from umap import UMAP

from ttk.utils import timer


class Decomposition:

    def __int__(self, model='QDA'):
        self.model = model.upper()
        self.decomposition = self.get_decomposition()

    @property  
    def get_decomposition(self):
        assert self.model in ["LDA", "QDA", "PCA", "SVD", "SPCA", "UMAP"]
        if self.model == "LDA":
            return LinearDiscriminantAnalysis
        elif self.model == "QDA":
            return QuadraticDiscriminantAnalysis
        elif self.model == "PCA":
            return PCA
        elif self.model == "SVD":
            return TruncatedSVD
        elif self.model == "SPCA":
            return SparsePCA
        elif self.model == "UMAP":
            return UMAP

    @timer
    def fit(self, x, y=None, n_components=0.95):
        if self.model == "LDA":
            self.decomposition = self.decomposition(n_components=n_components)
            self.decomposition.fit(X=x, y=y)
        elif self.model == "UMAP":
            self.decomposition = self.decomposition(n_components=n_components)
            self.decomposition.fit(X=x, y=y)
        elif self.model == "QDA":
            self.decomposition = self.decomposition()
            self.decomposition.fit(X=x, y=y)
        else:
            self.decomposition = self.decomposition(n_components=n_components)
            self.decomposition.fit(X=x)

    @timer
    def transform(self, x):
        return self.decomposition.tranform(X=x)

    def save(self, filename):
        pickle.dump(self.decomposition, open(filename, "wb"))
