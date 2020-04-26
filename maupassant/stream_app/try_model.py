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
        if text_2_predict != "":
            threshold = st.slider("threshold", 0.0, 1.0)
            prediction = self.predictor.predict_one(text_2_predict, threshold=threshold)
            prediction = {str(k): v for k, v in prediction.items()}
            st.json(prediction)
