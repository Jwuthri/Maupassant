import tensorflow as tf

from maupassant.tensorflow_utils import macro_f1


class TensorflowModel(object):

    def __init__(self, label_type, architecture, number_labels, vocab_size):
        self.label_type = label_type
        self.architecture = architecture
        self.number_labels = number_labels
        self.vocab_size = vocab_size
        self.embedding_type = "basic"
        self.model = tf.keras.Sequential()
        self.info = {
            "label_type": self.label_type, "architecture": self.architecture, "vocab_size": self.vocab_size,
            "number_labels": self.number_labels, "embedding_type": self.embedding_type
        }

    def set_model(self):
        input_layer = tf.keras.Input((128), name="input_layer")
        layer = tf.keras.layers.Embedding(self.vocab_size, 128)(input_layer)
        if self.architecture == "CNN_GRU":
            layer = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.GRU(256, activation='relu')(layer)
        elif self.architecture == "GRU":
            layer = tf.keras.layers.GRU(256, activation='relu')(layer)
        elif self.architecture == "CNN":
            layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.GlobalMaxPooling1D()(layer)
        elif self.architecture == "CNN_LSTM":
            layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.LSTM(256, activation='relu')(layer)
        else:
            layer = tf.keras.layers.GlobalMaxPooling1D()(layer)

        layer = tf.keras.layers.Dense(512, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.2)(layer)
        layer = tf.keras.layers.Dense(units=self.number_labels, activation="softmax")(layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=layer)

    def compile_model(self):
        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy",
            metrics=[macro_f1, "sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy", "accuracy"])
