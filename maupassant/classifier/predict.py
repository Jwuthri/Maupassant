import os
import glob
import json
import pickle

import tensorflow as tf
import tensorflow_hub as hub

from maupassant.utils import timer
from maupassant.utils import predict_format


class Predictor(object):
    """Tool to predict through the model."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.info = self.load_info()
        self.encoders = self.load_encoder()
        self.model = self.load_model()

    def load_info(self):
        info_path = os.path.join(self.model_path, "model.json")
        with open(info_path) as json_file:
            info = json.load(json_file)

        return info

    def load_model(self):
        return tf.keras.experimental.load_from_saved_model(
            os.path.join(self.model_path, "model"),
            custom_objects={'KerasLayer': hub.KerasLayer}
        )

    def load_encoder(self):
        path = os.path.join(self.model_path, "*encoder.pkl")
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
