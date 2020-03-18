import os
import glob
import pickle
import numpy as np

import tensorflow as tf
import tensorflow_text

from maupassant.utils import timer
from maupassant.feature_extraction.embedding import BertEmbedding


class Predictor(object):

    def __init__(self, model_dir):
        self.encoders = self.load_encoder(model_dir)
        self.model = self.set_model()
        self.set_weights(model_dir)

    def load_encoder(self, model_dir):
        encoders_files = glob.glob(model_dir + "/*encoder.pkl")
        encoders = {}
        for file in encoders_files:
            encoder = pickle.load(open(file, "rb"))
            encoder_name = os.path.split(file)[1].split('.')[0]
            encoders[encoder_name] = dict(enumerate(encoder.classes_))

        return encoders

    @staticmethod
    def set_output_layer(classification_type, label, nb_classes):
        if classification_type == "binary":
            output = tf.keras.layers.Dense(1, activation="sigmoid", name=label)
        elif classification_type == "multi":
            output = tf.keras.layers.Dense(nb_classes, activation="sigmoid", name=label)
        else:
            output = tf.keras.layers.Dense(nb_classes, activation="softmax", name=label)

        return output

    def set_model(self):
        input_text = tf.keras.Input((), dtype=tf.string, name='input_text')
        embedding = BertEmbedding().get_embedding(multi_output=True)(input_text)
        dense = tf.keras.layers.Dense(512, activation="relu", name="hidden_layer")(embedding)
        outputs = []
        for k, v in self.encoders.items():
            _, classification, label, _ = k.split("_")
            layer = self.set_output_layer(classification, label, len(v))(dense)
            outputs.append(layer)

        return tf.keras.models.Model(inputs=input_text, outputs=outputs)

    def set_weights(self, path):
        path = os.path.join(path, 'variables')
        latest = tf.train.latest_checkpoint(path)
        self.model.load_weights(latest)

    @staticmethod
    def predict_format(x):
        if isinstance(x, str):
            x = np.asarray([x])
        if isinstance(x, list):
            x = np.asarray(x)

        return x

    def predict_proba(self, x):
        x = self.predict_format(x)

        return self.model.predict(x)

    @timer
    def predict_classes(self, x):
        probas = self.predict_proba(x)
        preds = []
        for proba in probas:
            size = len(proba[0])
            for k, v in self.encoders.items():
                if len(v) == size:
                    preds.append([(v[label], th) for label, th in enumerate(proba[0]) if th >= 0.5])

        return preds

    def predict_batch(self, x):
        raise NotImplemented


if __name__ == '__main__':
    path = '/home/jwuthri/Documents/GitHub/Maupassant/maupassant/models/one_to_many_2020_03_18_00_31_29'
    predictor = Predictor(path)
    predictor.predict_proba(['where is my order?'])
    print(predictor.predict_proba(['where is my order?']))
    res = predictor.predict_classes(['where is my order?'])
    print(res)
    print(predictor.predict_classes(['I want a refund']))
