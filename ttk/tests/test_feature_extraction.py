import numpy as np
import pandas as pd

from ttk.feature_extraction.tfidf import Tfidf


class TestTfIdf:

    def test_transform(self):
        tfidf = Tfidf()
        docs = ['where you from', 'where are you']
        tfidf.fit(documents=docs)
        data = tfidf.transform(document=docs)
        t_data = np.array(
            [[0., 0.53404633, 0.53404633, 0.53404633, 0.37997836, 0.], [0.6316672, 0., 0., 0., 0.44943642, 0.6316672]]
        )

        assert data == t_data

    def test_get_features_name(self):
        tfidf = Tfidf()
        docs = ['where you from', 'where are you']
        tfidf.fit(documents=docs)
        columns = tfidf.get_features_name
        t_columns = ['are', 'form', 'where', 'you']

        assert columns == t_columns

    def test_end_to_end(self):
        tfidf = Tfidf()
        docs = ['where you from', 'where are you']
        tfidf.fit(documents=docs)
        data = tfidf.transform(document=docs)
        columns = tfidf.get_features_name
        df = pd.DataFrame(data)
        df.columns = columns
        t_data = np.array(
            [[0., 0.53404633, 0.53404633, 0.53404633, 0.37997836, 0.], [0.6316672, 0., 0., 0., 0.44943642, 0.6316672]]
        )
        t_columns = ['are', 'form', 'where', 'you']
        t_df = pd.DataFrame(t_data)
        t_df.columns = t_columns

        assert data == t_data
        assert columns == t_columns
        assert df.equals(t_df)
