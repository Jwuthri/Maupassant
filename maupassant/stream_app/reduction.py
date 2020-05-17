import pandas as pd

import streamlit as st
import plotly.express as px

from maupassant.feature_decomposition.dimension_reduction import Decomposition
from maupassant.feature_extraction.embedding import Embedding


class DimensionReduction(object):

    def __init__(self):
        self.models = ["LDA", "PCA", "SVD", "SPCA", "UMAP"]
        self.embedding = Embedding()

    def read_file(self, file):
        return pd.read_csv(file)

    def create_feature(self, data, feature):
        embeddings = self.embedding.predict_one(x=[data[feature].values])
        embeddings_data = pd.DataFrame([[float(x) for x in d] for d in embeddings])

        return embeddings_data

    def main(self):
        file = st.file_uploader("Upload a dataset", type=["csv"])
        if not file:
            st.info("Please upload a CSV file.")
            return
        data = self.read_file(file)
        st.dataframe(data.head())
        file.close()
        feature = st.multiselect("Select the text column", data.columns)
        label = st.multiselect("Select the label column", data.columns)
        if label and feature:
            feature = feature[0]
            label = label[0]
            embeddings_data = self.create_feature(data, feature)
            st.dataframe(embeddings_data.head())
            decomposition_model = st.selectbox("Select the decomposition algorithm", self.models)
            n_components = st.selectbox("2D or 3D?", [2, 3])
            if decomposition_model:
                model = Decomposition(model=decomposition_model, n_components=n_components)
                model.fit(embeddings_data.values, data[label].values)
                decomposed_data = pd.DataFrame(model.transform(embeddings_data.values, data[label].values))
                decomposed_data['labels'] = data[label]
                st.dataframe(decomposed_data)
                if n_components == 3:
                    decomposed_data.columns = ['x', 'y', 'z', 'label']
                    fig = px.scatter_3d(decomposed_data, x="x", y="y", z="z", color="label")
                else:
                    decomposed_data.columns = ['x', 'y', 'label']
                    fig = px.scatter(decomposed_data, x="x", y="y", color="label")
                st.plotly_chart(fig)
        else:
            st.info("Please select the label and feature columns.")
