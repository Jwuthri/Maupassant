import numpy as np
import pandas as pd

from maupassant.feature_extraction.tfidf import Tfidf


class TestTfIdf(object):

    def test_transform(self):
        tfidf = Tfidf()
        docs = ['where you from', 'where are you']
        tfidf.fit_model(documents=docs)
        data = tfidf.transform(document=docs)
        t_data = np.array(
            [[0., 0.70490949, 0.50154891, 0.50154891], [0.70490949, 0., 0.50154891, 0.50154891]]
        )
        t_data = [[round(x, 3) for x in xx] for xx in t_data]
        data = [[round(x, 3) for x in xx] for xx in data]

        assert data == t_data

    def test_get_features_name(self):
        tfidf = Tfidf()
        docs = ['where you from', 'where are you']
        tfidf.fit_model(documents=docs)
        columns = tfidf.get_features_name
        t_columns = ['are', 'from', 'where', 'you']

        assert columns == t_columns

    def test_end_to_end(self):
        tfidf = Tfidf()
        docs = ['where you from', 'where are you']
        tfidf.fit_model(documents=docs)
        data = tfidf.transform(document=docs)
        columns = tfidf.get_features_name
        data = [[round(x, 3) for x in xx] for xx in data]
        df = pd.DataFrame(data)
        df.columns = columns
        t_data = np.array(
            [[0., 0.70490949, 0.50154891, 0.50154891], [0.70490949, 0., 0.50154891, 0.50154891]]
        )
        t_columns = ['are', 'from', 'where', 'you']
        t_data = [[round(x, 3) for x in xx] for xx in t_data]
        t_df = pd.DataFrame(t_data)
        t_df.columns = t_columns

        assert data == t_data
        assert columns == t_columns
        assert df.equals(t_df)
