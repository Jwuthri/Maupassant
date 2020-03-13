import numpy as np

import tensorflow_text
import tensorflow as tf

from ttk.utils import timer, text_format
from ttk.dataset.tensorflow import TensorflowDataset
from ttk.training_utils import macro_f1, macro_soft_f1
from ttk.feature_extraction.embedding import BertEmbedding


class TensorflowClassifier(TensorflowDataset):

    def __init__(self, text='text', label='label', clf_type="mutli", batch_size=512, buffer_size=1024, epochs=30):
        super().__init__(text, label, True, batch_size, buffer_size)
        self.classification_type = clf_type
        self.text = text
        self.label = label
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.model = tf.keras.Sequential()

    def set_model(self):
        if self.classification_type == "binary":
            output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")
        elif self.classification_type == "multi":
            output = tf.keras.layers.Dense(self.nb_classes, activation="sigmoid", name="output")
        else:
            output = tf.keras.layers.Dense(self.nb_classes, activation="softmax", name="output")

        self.model = tf.keras.Sequential(
            [
                BertEmbedding().embedding,
                tf.keras.layers.Dense(512, activation="relu", name="hidden_layer"),
                output,
            ]
        )

    def get_summary(self):
        print(self.model.summary())

    def compile_model(self):
        if self.classification_type == "binary":
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[macro_f1, "accuracy"])
        elif self.classification_type == "multi":
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
    def train(self, train_dataset, val_dataset, epochs=30, callbacks=[]):
        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

    @staticmethod
    def predict_format(x):
        if isinstance(x, str):
            x = np.asarray([x])
        if isinstance(x, list):
            x = np.asarray(x)

        return x

    @timer
    def predict_proba(self, x):
        x = self.predict_format(x)

        return self.model.predict(x)

    @timer
    def predict_classes(self, x):
        if self.classification_type == "multi":
            preds = self.predict_proba(x)
            return [[self.classes_mapping[i] for i, j in enumerate(pred) if j >= 0.5] for pred in preds]
        else:
            x = self.predict_format(x)
            return self.model.predict_classes(x)

    def export_model(self, model_path):
        f_blue = text_format(txt_color='blue')
        b_black = text_format(txt_color='black', bg_color='green')
        end = text_format(end=True)
        tf.keras.experimental.export_saved_model(self.model, model_path)
        print(f"{f_blue}Model was exported in this path: {b_black}{model_path}{end}")

    def load_model(self, model_path):
        latest = tf.train.latest_checkpoint(model_path)
        self.model.load_weights(latest)
