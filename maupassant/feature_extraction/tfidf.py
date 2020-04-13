import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from maupassant.utils import timer


class Tfidf(object):

	def __init__(self, bigrams=False, unigrams=True, analyzer='word'):
		self.unigrams = unigrams
		self.bigrams = bigrams
		self.analyzer = analyzer
		self.ngram_range = self.get_ngrams_range
		self.tfidf = self.get_tfidf

	@property
	def get_ngrams_range(self):
		contains_one_of = self.unigrams or self.bigrams
		assert contains_one_of is True
		if self.unigrams:
			if self.bigrams:
				return 1, 2
			else:
				return 1, 1
		else:
			return 2, 2

	@property
	def get_tfidf(self):
		return TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range)

	@timer
	def fit(self, documents):
		assert type(documents) in [list, np.ndarray]
		self.tfidf.fit_model(documents)

	@timer
	def transform(self, document):
		if isinstance(document, str):
			document = list(document)

		return self.tfidf.transform(document).toarray()

	@property
	def get_features_name(self):
		return self.tfidf.get_feature_names()

	def save(self, filename):
		pickle.dump(self.tfidf, open(filename, "wb"))