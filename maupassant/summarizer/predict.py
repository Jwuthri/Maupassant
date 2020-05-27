import os
import glob
import json
import pickle

import numpy as np
import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from maupassant.utils import timer
from maupassant.utils import predict_format
from maupassant.summarizer.model import WeightedTfIdf, GoogleT5


class Predictor(object):

    def __init__(self, model_name='GoogleT5', keywords=[], min_threshold=2.0):
        self.model_name = model_name
        self.keywords = keywords
        self.min_threshold = min_threshold
        self.model = self.set_model()

    def set_model(self):
        if self.model_name == "GoogleT5":
            return GoogleT5()
        else:
            return WeightedTfIdf(self.keywords, self.min_threshold)

    @timer
    def predict(self, text):
        return self.model.predict(text)


class TensorflowPredictor(object):
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


if __name__ == '__main__':
    # p = Predictor(model_name='tfidf', min_threshold=1.5)
    # summarized = p.predict('''
    # Hello,I came across your Instagram and I absolutely love your clothing, they look absolutely amazing!
    # I’m a lifestyle/ fashion/ beauty content creator based in Perth, Western Australia and
    # I am very interested in collaborating with you. I think a collaboration would greatly benefit both myself and
    # your company in many ways! The age range in my audience is 58% 18-24 and 26% 25-34 year olds.
    # The top two countries my followers are located are 38% Australia and 10% United States.
    # This would be great for your company to get more exposure other then Australia!
    # This collaboration would benefit myself greatly to help myself grow and help myself get exposure.
    # ''')
    # print(summarized)
    model_path = "/home/jwuthri/Documents/GitHub/Maupassant/maupassant/models/binary-label_is_relevant_hard_2020_05_27_11_20_12"
    pred = TensorflowPredictor(model_path)
    data = [np.asarray(['My order number is 62767']), np.asarray(['hello. where is my order? My order number is 62767.'])]
    print(pred.predict_one(x=data))
    data = [np.asarray(['hello']),
            np.asarray(['hello. where is my order? My order number is 62767.'])]
    print(pred.predict_one(x=data))
