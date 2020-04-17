import streamlit as st

from maupassant.classifier.predict import Predictor


class TryModel(object):

    def __init__(self, path):
        self.path = path
        self.predictor = self.set_predictor()

    @st.cache(allow_output_mutation=True)
    def set_predictor(self):
        return Predictor(self.path)

    def predict(self):
        st.title("Inference")
        text_2_predict = st.text_area("enter text predict")
        prediction = self.predictor.predict_one(text_2_predict, threshold=0.5)
        st.json(prediction)
