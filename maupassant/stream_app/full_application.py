import pandas as pd

import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split

from maupassant.feature_decomposition.dimension_reduction import Decomposition
from maupassant.feature_extraction.pretrained_embedding import PretrainedEmbedding
from maupassant.feature_extraction.tfidf import Tfidf
from maupassant.dataset.pandas import remove_rows_contains_null


class TextApplication(object):

    def __init__(self):
        self.models = ["", "LDA", "PCA", "SVD", "SPCA", "UMAP"]
        self.embedding = PretrainedEmbedding()
        self.max_len = 10000

    def create_feature(self, data, feature):
        new_data = None
        featuring = st.sidebar.selectbox("How to generate features?", ['Embedding', 'TfIdf'])
        if featuring == 'Embedding':
            embeddings = self.embedding.predict_one(x=[data[feature].values])
            new_data = pd.DataFrame([[float(x) for x in d] for d in embeddings])
        elif featuring == "TfIdf":
            grams = st.sidebar.multiselect("Use bigrams? unigrams? or both?", ['unigrams', 'bigrams'])
            if grams:
                if "unigrams" in grams and "bigrams" in grams:
                    tfidf = Tfidf(bigrams=True, unigrams=True)
                elif "bigrams" in grams:
                    tfidf = Tfidf(bigrams=True, unigrams=False)
                else:
                    tfidf = Tfidf(bigrams=False, unigrams=True)
                tfidf.fit_model(documents=data[feature].values)
                new_data = pd.DataFrame(tfidf.transform(document=data[feature].values))
                new_data.columns = tfidf.get_features_name
            else:
                st.sidebar.warning("Please select the ngrams.")

        return new_data

    def subset_dataset(self, data, label, feature):
        data = remove_rows_contains_null(data, label)
        data = remove_rows_contains_null(data, feature)
        if len(data) > self.max_len:
            st.warning(f"The dataset is too big, will only select {self.max_len} rows keeping classes distribution")
            test_size = 1 - (self.max_len / len(data))
            data, _, y_train, _ = train_test_split(data[feature], data[label], test_size=test_size)
            data[label] = y_train
        st.plotly_chart(px.histogram(data, x=label))

        return data

    @staticmethod
    def reduce_dimension(labels, embeddings_data, dim_reduction_model, n_components):
        model = Decomposition(model=dim_reduction_model, n_components=n_components)
        model.fit_model(embeddings_data.values, labels.values)
        reduce_data = pd.DataFrame(model.transform(embeddings_data.values, labels.values))
        reduce_data['label'] = labels
        if n_components == 3:
            reduce_data.columns = ['x', 'y', 'z', 'label']
            fig = px.scatter_3d(reduce_data, x="x", y="y", z="z", color="label")
        else:
            reduce_data.columns = ['x', 'y', 'label']
            fig = px.scatter(reduce_data, x="x", y="y", color="label")
        st.dataframe(reduce_data.head())
        st.plotly_chart(fig)

        return reduce_data

    def main(self):
        file = st.file_uploader("Upload a dataset", type=["csv"])
        if not file:
            st.warning("Please upload a CSV file.")
            return
        data = pd.read_csv(file)
        file.close()
        st.dataframe(data.head())
        feature = st.selectbox("Select the text column", [''] + list(data.columns))
        label = st.selectbox("Select the label column", [''] + list(data.columns))
        if not label and not feature:
            st.info("Please select the label and feature columns.")
            return
        data = self.subset_dataset(data, label, feature)
        embeddings_data = self.create_feature(data, feature)
        if embeddings_data is None:
            return
        st.dataframe(embeddings_data.head())
        dim_reduction_model = st.selectbox("Select the dimensional reduction algorithm", self.models)
        n_components = st.selectbox("2D or 3D?", [2, 3])
        if not dim_reduction_model:
            st.warning("Please select a dimensional reduction algorithm")
            return
        reduce_data = self.reduce_dimension(data[label], embeddings_data, dim_reduction_model, n_components)
