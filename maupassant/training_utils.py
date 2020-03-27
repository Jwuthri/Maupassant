import os
import pickle
from collections import Counter

import numpy as np

import tensorflow as tf


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())

    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}


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


class TrainerHelper(object):

    def __init__(self, type, nb_classes):
        self.model = tf.keras.Sequential()
        self.type = type
        self.nb_classes = nb_classes

    def compile_model(self):
        if self.type == "binary":
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[macro_f1, "accuracy"])
        elif self.type == "multi":
            self.model.compile(optimizer="adam", loss=macro_soft_f1, metrics=[macro_f1, "accuracy"])
        else:
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[macro_f1, "accuracy"])

    def get_output_layer(self):
        if self.type == "binary":
            output = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
        elif self.type == "multi":
            output = tf.keras.layers.Dense(self.nb_classes, activation="sigmoid", name="output_layer")
        else:
            output = tf.keras.layers.Dense(self.nb_classes, activation="softmax", name="output_layer")

        return output

    @staticmethod
    def callback_func(checkpoint_path, tensorboard_dir=None):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, period=5)
        if tensorboard_dir:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            return [tensorboard, checkpoint]
        else:
            return [checkpoint]

    def get_summary(self):
        print(self.model.summary())

    def export_model(self, model_path):
        self.model.save_weights(model_path)
        print(f"Model was exported in this path: {model_path}")

    def plot_model(self, filename):
        tf.keras.utils.plot_model(self.model, to_file=filename)

    def export_encoder(self, model_dir, label_data):
        for k in label_data.keys():
            le = label_data[k]['encoder']
            classification = label_data[k]['classification']
            id = label_data[k]['id']
            filename = os.path.join(model_dir, f"{id}_{classification}_{k}_encoder.pkl")
            pickle.dump(le, open(filename, "wb"))
