import streamlit as st

from maupassant.inference.one_one import Predictor


class Inference(object):

    def __init__(self, path):
        self.path = path
        self.prediction = self.set_predictor()

    @st.cache(allow_output_mutation=True)
    def set_predictor(self):
        return Predictor(self.path)

    def main(self):
        st.title("Inference")
        text_2_predict = st.text_area("enter text predict")
        prediction = self.prediction.predict_classes(text_2_predict)
        st.markdown(prediction, unsafe_allow_html=True)
