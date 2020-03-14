import numpy as np

import tensorflow_text
import tensorflow as tf

from maupassant.utils import timer, text_format
from maupassant.training_utils import macro_f1, macro_soft_f1
from maupassant.feature_extraction.embedding import BertEmbedding


class TensorflowClassifier(object):

    def __init__(self, labels=dict(), batch_size=1024, epochs=30):
        self.labels = labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = tf.keras.Sequential()

    @staticmethod
    def set_output_layer(classification_type, label, nb_classes):
        if classification_type == "binary":
            output = tf.keras.layers.Dense(1, activation="sigmoid", name=label)
        elif classification_type == "multi":
            output = tf.keras.layers.Dense(nb_classes, activation="sigmoid", name=label)
        else:
            output = tf.keras.layers.Dense(nb_classes, activation="softmax", name=label)

        return output

    def set_model(self, label_data):
        input_text = tf.keras.Input((), dtype=tf.string, name='input_text')
        embedding = BertEmbedding().get_embedding(multi_output=True)(input_text)
        dense = tf.keras.layers.Dense(512, activation="relu", name="hidden_layer")(embedding)
        outputs = []
        for k, v in label_data.items():
            dense_2 = tf.keras.layers.Dense(512, activation="relu", name=f"hidden_{k}")(dense)
            layer = self.set_output_layer(v["classification"], k, len(v['encoder'].classes_))(dense_2)
            outputs.append(layer)
        self.model = tf.keras.models.Model(inputs=input_text, outputs=outputs)

    def get_summary(self):
        print(self.model.summary())

    def compile_model(self):
        loss, metrics = {}, {}
        for k, v in self.labels.items():
            if v == "binary":
                loss[k] = "binary_crossentropy"
                metrics[k] = [macro_f1]
            elif v == "multi":
                loss[k] = macro_soft_f1
                metrics[k] = [macro_f1]
            else:
                loss[k] = "sparse_categorical_crossentropy"
                metrics[k] = [macro_f1]
        self.model.compile(optimizer='adam', loss=loss, metrics=metrics)

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
        return self.model.fit(
            x=train_dataset[0], y=train_dataset[1], validation_data=val_dataset,
            batch_size=self.batch_size, epochs=epochs, callbacks=callbacks)

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