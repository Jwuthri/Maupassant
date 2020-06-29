import tensorflow as tf


class TensorflowModel(object):

    def __init__(self, label_type, architecture, number_labels, vocab_size):
        self.label_type = label_type
        self.architecture = architecture
        self.number_labels = number_labels
        self.vocab_size = vocab_size
        self.embedding_size = 128
        self.embedding_type = "basic"
        self.model = tf.keras.Sequential()
        self.info = {
            "label_type": self.label_type, "architecture": self.architecture, "vocab_size": self.vocab_size,
            "number_labels": self.number_labels, "embedding_type": self.embedding_type
        }

    def set_model_api(self, pretrained_embedding):
        input_layer = tf.keras.Input((128), name="input_layer")
        layer = pretrained_embedding(input_layer)
        # layer = tf.keras.layers.Reshape(target_shape=(1, 128))(layer)
        if self.architecture == "CNN_GRU":
            layer = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.GRU(256, activation='relu')(layer)
        elif self.architecture == "GRU":
            layer = tf.keras.layers.GRU(256, activation='relu')(layer)
        elif self.architecture == "CNN":
            layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.GlobalMaxPooling1D()(layer)
        elif self.architecture == "CNN_LSTM":
            layer = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu', strides=1)(layer)
            layer = tf.keras.layers.LSTM(256, activation='relu')(layer)
        else:
            layer = tf.keras.layers.GlobalMaxPooling1D()(layer)

        layer = tf.keras.layers.Dense(512, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.2)(layer)
        layer = tf.keras.layers.Dense(units=self.number_labels, activation="softmax")(layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=layer)

    def set_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.embedding_size))
        if self.architecture == "LSTM":
            self.model.add(tf.keras.layers.LSTM(256))
        elif self.architecture == "GRU":
            self.model.add(tf.keras.layers.GRU(256))
        elif self.architecture == "CNN_LSTM":
            self.model.add(tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1))
            self.model.add(tf.keras.layers.LSTM(256, activation='relu'))
        elif self.architecture == "CNN_GRU":
            self.model.add(tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1))
            self.model.add(tf.keras.layers.GRU(256, activation='relu'))
        else:
            self.model.add(tf.keras.layers.GlobalMaxPooling1D())
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units=self.number_labels, activation='softmax'))

    def compile_model(self):
        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"])
