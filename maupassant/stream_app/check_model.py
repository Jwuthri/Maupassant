# import os
# import glob
# import pickle
#
# import numpy as np
# import streamlit as st
#
# import tensorflow_text
# import tensorflow as tf
# import tensorflow_hub as hub
#
#
# class PredictModel(object):
#     """Predict the sentiment for all languages."""
#
#     def __init__(self, model_path):
#         self.model_path = model_path
#         self.encoders = self.load_encoder()
#         self.model = self.set_model()
#         self.load_model()
#
#     def load_encoder(self):
#         encoders_files = glob.glob(self.model_path + "/*encoder.pkl")
#         encoders = {}
#         for file in encoders_files:
#             encoder = pickle.load(open(file, "rb"))
#             encoder_name = os.path.split(file)[1].split('.')[0]
#             encoders[encoder_name] = dict(enumerate(encoder.classes_))
#
#         return encoders
#
#     def set_model(self):
#         input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
#         bert_module = hub.KerasLayer(
#             "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
#             input_shape=[], dtype=tf.string, trainable=False, name='multilingual_embed')
#         embedding_layer = bert_module(input_layer)
#         reshape_layer = tf.keras.layers.Reshape(target_shape=(1, 512))(embedding_layer)
#         conv_layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(reshape_layer)
#         gpooling_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
#         flatten_layer = tf.keras.layers.Flatten()(gpooling_layer)
#         dense_layer = tf.keras.layers.Dense(250, activation="relu")(flatten_layer)
#         dropout_layer = tf.keras.layers.Dropout(0.25)(dense_layer)
#         for k, v in self.encoders.items():
#             _, classification, label, _ = k.split("_")
#             layer = tf.keras.layers.Dense(len(v), activation="sigmoid", name=label)(dropout_layer)
#
#         return tf.keras.models.Model(inputs=input_layer, outputs=layer)
#
#     def load_model(self):
#         latest = tf.train.latest_checkpoint(self.model_path)
#         self.model.load_weights(latest)
#
#     @staticmethod
#     def predict_format(x):
#         if isinstance(x, str):
#             x = np.asarray([x])
#         if isinstance(x, list):
#             x = np.asarray(x)
#
#         return x
#
#     def predict(self, x):
#         x = self.predict_format(x)
#         proba = self.model.predict(x)[0]
#         for k, v in self.encoders.items():
#             preds = [(v[label], th) for label, th in enumerate(proba) if th >= 0.5]
#
#         return preds
