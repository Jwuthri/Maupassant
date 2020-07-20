import tensorflow as tf

from maupassant.feature_extraction.pretrainedembedding import PretrainedEmbedding


class TensorflowModel(object):

    def __init__(self, architecture, embedding_type, text):
        self.architecture = architecture
        self.embedding_type = embedding_type
        self.text = text
        self.number_output = 2 if self.text else 1
        self.label_type = "binary-label"
        self.model = tf.keras.Sequential()
        self.info = {
            "label_type": self.label_type, "architecture": self.architecture,
            "number_labels": 1, "embedding_type": self.embedding_type, "number_output": self.number_output
        }

    def get_sub_model(self, architecture, name="input_sentences"):
        embed_module = PretrainedEmbedding(model_type=self.embedding_type, name="keras_" + name)
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
