import streamlit as st

from maupassant.text_summarization.predict import Predictor


class Summarizer(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.set_model(self.model_name)

    @staticmethod
    def set_model(model_name, keywords=[], min_threshold=1.35):
        return Predictor(model_name=model_name, keywords=keywords, min_threshold=min_threshold)

    def main(self):
        if self.model_name == "TfIdf":
            min_threshold = st.slider("Set the minimum sentences threshold", 1.0, 5.0, 1.35)
            keywords = st.text_input('Enter your keywords here separate by "," example', "refund, return")
            keywords = keywords.split(",")
            keywords = [x.strip() for x in keywords]
            self.model.model.set_keywords(keywords)
            self.model.model.min_threshold = min_threshold

        text = st.text_area("Enter the text to summarize here:")
        if text:
            if self.model_name == "TfIdf":
                summarized, ranked = self.model.predict(text)
                st.markdown(summarized, unsafe_allow_html=True)
                st.json(ranked)
            else:
                summarized = self.model.predict(text)
                st.markdown(summarized, unsafe_allow_html=True)
