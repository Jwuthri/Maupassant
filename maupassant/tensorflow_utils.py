import os
import json
import glob
import pickle
import datetime

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from maupassant.utils import timer
from maupassant.utils import predict_format
from maupassant.settings import MODEL_PATH
from maupassant.feature_extraction.embedding import Embedding


@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels)."""
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1
    macro_cost = tf.reduce_mean(cost)

    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)"""
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)

    return macro_f1


def hamming_score(y_true, y_pred):
    """Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case"""
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    return np.mean(acc_list)


def learning_curves(history):
    """Plot the learning curves of loss and macro f1 score"""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    macro_f1 = history.history["macro_f1"]
    val_macro_f1 = history.history["val_macro_f1"]

    return loss, val_loss, macro_f1, val_macro_f1


class Model(object):
    """Setup the model."""

    def __init__(self, label_type, architecture, number_labels, embedding_type):
        self.label_type = label_type
        self.architecture = architecture
        self.number_labels = number_labels
        self.embedding_type = embedding_type
        self.model = tf.keras.Sequential()
        self.info = {
            "label_type": self.label_type, "architecture": self.architecture,
            "number_labels": self.number_labels, "embedding_type": self.embedding_type
        }

    def get_output_layer(self):
        if self.label_type == "binary-label":
            output = tf.keras.layers.Dense(units=1, activation="sigmoid", name="output_layer")
        elif self.label_type == "multi-label":
            output = tf.keras.layers.Dense(units=self.number_labels, activation="sigmoid", name="output_layer")
        else:
            output = tf.keras.layers.Dense(units=self.number_labels, activation="softmax", name="output_layer")

        return output

    def set_model(self):
        embed_module = Embedding(model_type=self.embedding_type)
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        layer = embed_module.model(input_layer)
        layer = tf.keras.layers.Reshape(target_shape=(1, 512))(layer)

        if self.architecture in ['CNN_NN', 'CNN_GRU_NN']:
            layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(layer)
            if self.architecture == 'CNN_GRU_NN':
                layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
                layer = tf.keras.layers.GRU(128, activation='relu')(layer)
            else:
                layer = tf.keras.layers.GlobalMaxPooling1D()(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(128, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.2)(layer)
        layer = self.get_output_layer()(layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=layer)


class TrainerHelper(Model):
    """Tool to train model."""

    def __init__(self, label_type, architecture, number_labels, embedding_type):
        self.model = tf.keras.Sequential()
        super().__init__(label_type, architecture, number_labels, embedding_type)

    def compile_model(self):
        if self.info['label_type'] == "binary-label":
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[macro_f1, "accuracy"])
        elif self.info['label_type'] == "multi-label":
            self.model.compile(optimizer="adam", loss=macro_soft_f1, metrics=[macro_f1, "accuracy"])
        else:
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[macro_f1, "accuracy"])

    @staticmethod
    def callback_func(checkpoint_path, tensorboard_dir=None):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, period=5)
        if tensorboard_dir:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            return [tensorboard, checkpoint]
        else:
            return [checkpoint]

    @timer
    def fit_model(self, train_dataset, val_dataset, epochs=30, callbacks=[]):
        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

    @staticmethod
    def define_paths(classifier, label):
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        name = f"{classifier}_{label}_{date}"
        base_dir = os.path.join(MODEL_PATH, name)

        return {
            "path": base_dir,
            "model_path": os.path.join(base_dir, 'model'),
            "model_plot":  os.path.join(base_dir, "model.jpg"),
            "model_info": os.path.join(base_dir, "model.json"),
            "metrics_path": os.path.join(base_dir, "metrics.json"),
            "tensorboard_path": os.path.join(base_dir, "tensorboard"),
            "checkpoint_path": os.path.join(base_dir, "checkpoint"),
        }

    @staticmethod
    def export_model_plot(path, model):
        tf.keras.utils.plot_model(model, to_file=path)

    @staticmethod
    def export_model(path, model):
        tf.keras.experimental.export_saved_model(model, path)
        print(f"Model has been exported here => {path}")

    @staticmethod
    def export_encoder(directory, label_data):
        for k, v in label_data.items():
            path = os.path.join(directory, f"{v['id']}_{v['label_type']}_{k}_encoder.pkl")
            pickle.dump(v['encoder'], open(path, "wb"))
            print(f"{k} encoder has been exported here => {path}")

    @staticmethod
    def export_info(path, info):
        with open(path, 'w') as outfile:
            json.dump(info, outfile)
            print(f"Model information have been exported here => {path}")

    @staticmethod
    def export_metrics(path, metrics):
        with open(path, 'w') as outfile:
            json.dump(metrics, outfile)
            print(f"Model metrics have been exported here => {path}")


class PredictHelper(object):
    """Tool to predict through the model."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.info = self.load_info()
        self.encoders = self.load_encoder()
        self.model = self.load_model()

    def load_info(self):
        info_path = os.path.join(self.model_path, "info.json")
        with open(info_path) as json_file:
            info = json.load(json_file)

        return info

    def load_model(self):
        return tf.keras.experimental.load_from_saved_model(
            self.model_path,
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

    def predict_classes(self, prediction):
        classes = self.encoders[prediction[0]]
        results = [
            [(classes[label], float(th)) for label, th in enumerate(pred)]
            for pred in prediction[1]
        ]

        return [dict(result) for result in results][0]

    @timer
    def predict_one(self, x):
        probabilities = self.predict_probabilities(x)
        predictions = list(zip(self.encoders, probabilities))

        return [self.predict_classes(prediction) for prediction in predictions]

    @timer
    def predict_batch(self, x):
        probabilities = self.predict_probabilities(x)
        results = []
        for probability in probabilities:
            predictions = list(zip(self.encoders, probability))
            results.append([self.predict_classes(prediction) for prediction in predictions])

        return results
