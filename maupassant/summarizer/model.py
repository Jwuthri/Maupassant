import os
import glob
import json
import pickle

import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from maupassant.feature_extraction.embedding import Embedding
from maupassant.utils import timer


class TensorflowModel(object):

    def __init__(self, architecture, embedding_type, text):
        self.architecture = architecture
        self.embedding_type = embedding_type
        self.text = text
        self.number_label = 2 if self.text else 1
        self.label_type = "binary-label"
        self.model = tf.keras.Sequential()
        self.info = {
            "label_type": self.label_type, "architecture": self.architecture,
            "number_labels": 1, "embedding_type": self.embedding_type, "number_label": self.number_label
        }

    def get_sub_model(self, architecture, name="input_sentences"):
        embed_module = Embedding(model_type=self.embedding_type, name="keras_" + name)
        input_layer = tf.keras.Input((), dtype=tf.string, name=name)
        layer = embed_module.model(input_layer)
        layer = tf.keras.layers.Reshape(target_shape=(1, 512))(layer)
        if architecture == "CNN_GRU":
            layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.GRU(128, activation='relu')(layer)
        elif architecture == "GRU":
            layer = tf.keras.layers.GRU(256, activation='relu')(layer)
        elif architecture == "CNN":
            layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.GlobalMaxPooling1D()(layer)
        elif architecture == "CNN_LSTM":
            layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
        else:
            layer = tf.keras.layers.GlobalMaxPooling1D()(layer)

        return layer, input_layer

    def set_model(self):
        if self.text:
            model1, input_model1 = self.get_sub_model(architecture=self.architecture, name="input_sentences")
            model2, input_model2 = self.get_sub_model(architecture=self.architecture, name="input_text")
            model = tf.keras.layers.concatenate([model1, model2])
            if self.architecture == "CNN_LSTM":
                model = tf.keras.layers.LSTM(128, activation='relu')(model)
            model = tf.keras.layers.Dropout(0.2)(model)
            model = tf.keras.layers.Dense(64, activation="relu")(model)
            output = tf.keras.layers.Dense(units=1, activation="sigmoid", name="is_relevant")(model)
            self.model = tf.keras.models.Model(inputs=[input_model1, input_model2], outputs=output)
        else:
            model, input_model = self.get_sub_model(architecture=self.architecture, name="input_sentences")
            if self.architecture == "CNN_LSTM":
                model = tf.keras.layers.LSTM(128, activation='relu')(model)
            model = tf.keras.layers.Dropout(0.2)(model)
            model = tf.keras.layers.Dense(64, activation="relu")(model)
            output = tf.keras.layers.Dense(units=1, activation="sigmoid", name="is_relevant")(model)
            self.model = tf.keras.models.Model(inputs=input_model, outputs=output)


class TensorflowPredictorHelper(object):

    def __init__(self, model_path=None):
        self.model_path = model_path
        if self.model_path:
            self.info = self.load_info()
            self.encoders = self.load_encoder()
            self.model = self.load_model()
        else:
            print("Be careful no model loaded, please provide one.")

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

    def predict(self, x, threshold=0):
        return self.predict_one(x, threshold)
