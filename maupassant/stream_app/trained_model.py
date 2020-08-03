import os
import json
import glob
import pickle

import streamlit as st

import tensorflow as tf

from maupassant.text_classification.predict import Predictor
from maupassant.text_classification.model import TensorflowModel
from maupassant.utils import timer
from maupassant.utils import predict_format


class TrainedModel(object):

    def __init__(self, path):
        self.path = path
        self.predictor = self.set_predictor()

    @st.cache(allow_output_mutation=True)
    def set_predictor(self):
        return Predictor(self.path)

    def predict(self):
        st.title("Inference")
        text_2_predict = st.text_area("enter text predict")
        if text_2_predict:
            threshold = st.slider("threshold", 0.0, 1.0, 0.5)
            prediction = self.predictor.predict_one(text_2_predict, threshold=threshold)
            prediction = {str(k): v for k, v in prediction.items()}
            st.json(prediction)


class TrainedModelV2(object):

    def __init__(self, path):
        self.path = path
        self.info = self.load_info()
        self.model = self.get_model(self.info)
        self.encoders = self.load_encoder()

    def load_info(self):
        info_path = os.path.join(self.path, "model.json")
        with open(info_path) as json_file:
            info = json.load(json_file)

        return info

    def get_model(self, info):
        tf_model = TensorflowModel(**info)
        tf_model.set_model()
        model = tf_model.model
        latest = tf.train.latest_checkpoint(self.path)
        model.load_weights(latest)

        return model

    def load_encoder(self):
        path = os.path.join(self.path, "*encoder.pkl")
        encoders_files = sorted(glob.glob(path))
        encoders = dict()
        for file in encoders_files:
            encoder = pickle.load(open(file, "rb"))
            encoder_name = os.path.split(file)[1].split('.')[0]
            encoders[encoder_name] = dict(enumerate(encoder.classes_))

        return encoders

    @predict_format
    def predict_probabilities(self, x):
        return self.model.predict(x)

    def predict_classes(self, prediction, threshold=0.5):
        classes = self.encoders[prediction[0]]
        results = [(classes[label], float(th)) for label, th in enumerate(prediction[1]) if float(th) >= threshold]

        return dict(results)

    @timer
    def predict_one(self, x, threshold=0):
        probabilities = self.predict_probabilities(x=x)
        predictions = list(zip(self.encoders, probabilities))

        return self.predict_classes(predictions[0], threshold)

    @timer
    def predict_batch(self, x, threshold=0):
        probabilities = self.predict_probabilities(x)
        results = []
        for probability in probabilities:
            predictions = list(zip(self.encoders, probability))
            results.append([self.predict_classes(prediction, threshold) for prediction in predictions])

        return results

    def predict(self, x, threshold=0):
        return self.predict_one(x, threshold)

    def predict3(self):
        st.title("Inference")
        text_2_predict = st.text_area("enter text predict")
        if text_2_predict:
            threshold = st.slider("threshold", 0.0, 1.0, 0.5)
            prediction = self.predict_one(text_2_predict, threshold=threshold)
            prediction = {str(k): v for k, v in prediction.items()}
            st.json(prediction)
