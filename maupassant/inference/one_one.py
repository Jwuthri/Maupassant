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
        embed_module = BertEmbedding().get_embedding(multi_output=True)
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        embedding_layer = embed_module(input_layer)
        reshape_layer = tf.keras.layers.Reshape(target_shape=(1, 512))(embedding_layer)
        conv_layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(reshape_layer)
        gpooling_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
        flatten_layer = tf.keras.layers.Flatten()(gpooling_layer)
        dense_layer = tf.keras.layers.Dense(250, activation="relu")(flatten_layer)
        dropout_layer = tf.keras.layers.Dropout(0.25)(dense_layer)
        outputs = []
        for k, v in self.encoders.items():
            _, classification, label, _ = k.split("_")
            layer = self.set_output_layer(classification, label, len(v))(dropout_layer)
            outputs.append(layer)

        return tf.keras.models.Model(inputs=input_layer, outputs=outputs)

    def set_weights(self, path):
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
        proba = self.predict_proba(x)[0]
        for k, v in self.encoders.items():
            preds = [(v[label], th) for label, th in enumerate(proba) if th >= 0.5]

        return preds

    def predict_batch(self, x):
        raise NotImplemented


if __name__ == '__main__':
    path = '/home/jwuthri/Documents/GitHub/Maupassant/maupassant/models/one_to_one_2020_03_24_13_04_50'
    predictor = Predictor(path)
    print(predictor.predict_classes(['I want a refund']))
    print(predictor.predict_classes(['aller vous faire foutre']))
    print(predictor.predict_classes(['I am very angry about your services']))
    print(predictor.predict_classes(['I am not happy about your services']))