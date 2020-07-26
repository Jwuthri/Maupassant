from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn import preprocessing
from umap import UMAP

from maupassant.settings import MODEL_PATH
from maupassant.utils import timer


class Decomposition(object):

    def __init__(self, model='QDA', n_components=2, base_path=MODEL_PATH, name="dim_reduction", model_load=False):
        super().__init__(base_path, name, model_load)
        self.model = model.upper()
        self.n_components = n_components
        self.model = self.get_model(n_components=n_components)
        self.encoder = preprocessing.LabelEncoder()

    def get_model(self, n_components):
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
    def fit_model(self, x, y=None):
        if self.model == "UMAP":
            if y is not None:
                self.encoder.fit(y)
                y = self.encoder.transform(y)
            self.model.fit(X=x, y=y)
        else:
            self.model.fit(X=x, y=y)

    @timer
    def transform(self, x, y=None):
        if self.model == "UMAP":
            if y is not None:
                self.encoder.fit(y)
                y = self.encoder.transform(y)
            return self.model.fit_transform(X=x, y=y)
        else:
            return self.model.transform(X=x)
