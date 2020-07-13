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

    def set_model(self):
        input = tf.keras.Input((64), name="input_layer")
        embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)(input)
        if self.architecture == "LSTM":
            rnn = tf.keras.layers.LSTM(256, activation='relu')(embedding)
        elif self.architecture == "GRU":
            rnn = tf.keras.layers.GRU(256, activation='relu')(embedding)
        else:
            rnn = tf.keras.layers.SimpleRNN(256, activation='relu')(embedding)
        dense = tf.keras.layers.Dense(512, activation="relu")(rnn)
        dropout = tf.keras.layers.Dropout(0.2)(dense)
        output = tf.keras.layers.Dense(units=self.number_labels, activation="softmax")(dropout)
        self.model = tf.keras.models.Model(inputs=input, outputs=output)

    def compile_model(self):
        self.model.compile(
            optimizer="Nadam", loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]
        )
