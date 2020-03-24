import numpy as np

import tensorflow_text
import tensorflow as tf

from maupassant.utils import timer, text_format
from maupassant.dataset.tensorflow import TensorflowDataset
from maupassant.training_utils import macro_f1, macro_soft_f1
from maupassant.feature_extraction.embedding import BertEmbedding


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
            output = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
        elif self.classification_type == "multi":
            output = tf.keras.layers.Dense(self.nb_classes, activation="sigmoid", name="output_layer")
        else:
            output = tf.keras.layers.Dense(self.nb_classes, activation="softmax", name="output_layer")

        embed_module = BertEmbedding().get_embedding(multi_output=True)
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        embedding_layer = embed_module(input_layer)
        reshape_layer = tf.keras.layers.Reshape(target_shape=(1, 512))(embedding_layer)
        conv_layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(reshape_layer)
        gpooling_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
        flatten_layer = tf.keras.layers.Flatten()(gpooling_layer)
        dense_layer = tf.keras.layers.Dense(250, activation="relu")(flatten_layer)
        dropout_layer = tf.keras.layers.Dropout(0.25)(dense_layer)
        output_layer = output(dropout_layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

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

    def plot_model(self, filename):
        tf.keras.utils.plot_model(self.model, to_file=filename)
