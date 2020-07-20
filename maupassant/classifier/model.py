import tensorflow as tf

from maupassant.feature_extraction.pretrainedembedding import PretrainedEmbedding


class TensorflowModel(object):

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
        embed_module = PretrainedEmbedding(model_type=self.embedding_type)
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        layer = embed_module.model(input_layer)
        layer = tf.keras.layers.Reshape(target_shape=(1, 512))(layer)

        if self.architecture in ['CNN', 'CNN_GRU']:
            layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(layer)
            if self.architecture == 'CNN_GRU':
                layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
                layer = tf.keras.layers.GRU(128, activation='relu')(layer)
            else:
                layer = tf.keras.layers.GlobalMaxPooling1D()(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(128, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.2)(layer)
        layer = self.get_output_layer()(layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=layer)
