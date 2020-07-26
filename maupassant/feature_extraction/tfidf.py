import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from maupassant.utils import timer
from maupassant.settings import MODEL_PATH
from maupassant.utils import ModelSaverLoader


class Tfidf(ModelSaverLoader):

	def __init__(
			self, bigrams=False, unigrams=True, analyzer='word',
			base_path=MODEL_PATH, name="tf_idf", model_load=False):
		super().__init__(base_path, name, model_load)
		self.unigrams = unigrams
		self.bigrams = bigrams
		self.analyzer = analyzer
		self.ngram_range = self._get_ngrams_range()
		self.tfidf = self.get_model()

	def _get_ngrams_range(self):
		contains_one_of = self.unigrams or self.bigrams
		assert contains_one_of is True
		if self.unigrams:
			if self.bigrams:
				return 1, 2
			else:
				return 1, 1
		else:
			return 2, 2

	def get_model(self):
		return TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range)

	@timer
	def fit_model(self, documents):
		assert type(documents) in [list, np.ndarray]
		self.tfidf.fit(documents)

	@timer
	def transform(self, document):
		if isinstance(document, str):
			document = list(document)

		return self.tfidf.transform(document).toarray()

	@property
	def get_features_name(self):
		return self.tfidf.get_feature_names()
