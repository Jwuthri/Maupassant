import pandas as pd
import streamlit as st

from maupassant.feature_extraction.pretrained_embedding import PretrainedEmbedding
from maupassant.feature_extraction.tfidf import Tfidf


class Featuring(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.tfidf = Tfidf()
        self.embedding = PretrainedEmbedding()

    def main(self):
        text = st.text_area("Enter the text to normalize here:")
        if text:
            if self.model_name == "MultilangEmbedding":
                data = self.embedding.predict_one(x=text)
                data = [float(x) for x in data[0]]
                st.write(pd.DataFrame([data]))
            else:
                bigrams = st.checkbox('Use bigrams?')
                self.tfidf = Tfidf(bigrams=bigrams)
                self.tfidf.fit_model(documents=[text])
                data = self.tfidf.transform(document=[text])
                cols = self.tfidf.get_features_name
                df = pd.DataFrame(data)
                df.columns = cols
                st.write(df)
