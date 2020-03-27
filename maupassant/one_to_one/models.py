import tensorflow_text
import tensorflow as tf

from maupassant.feature_extraction.embedding import BertEmbedding
from maupassant.training_utils import TrainerHelper


class BasicModel(TrainerHelper):

    def __init__(self, type, nb_classes=1):
        super().__init__(type, nb_classes)

    def basic_model(self):
        embed_module = BertEmbedding().get_embedding()
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        embedding_layer = embed_module(input_layer)

        dense_layer = tf.keras.layers.Dense(512, activation="relu")(embedding_layer)
        dropout_layer = tf.keras.layers.Dropout(0.05)(dense_layer)
        output_layer = self.get_output_layer()(dropout_layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    def intermediate_model(self):
        embed_module = BertEmbedding().get_embedding()
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        embedding_layer = embed_module(input_layer)
        reshape_layer = tf.keras.layers.Reshape(target_shape=(1, 512))(embedding_layer)

        conv_layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(reshape_layer)
        gpooling_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
        dropout_layer = tf.keras.layers.Dropout(0.05)(gpooling_layer)

        flatten_layer = tf.keras.layers.Flatten()(dropout_layer)
        dense_layer = tf.keras.layers.Dense(250, activation="relu")(flatten_layer)
        dropout_layer1 = tf.keras.layers.Dropout(0.15)(dense_layer)
        output_layer = self.get_output_layer()(dropout_layer1)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    def advanced_model(self):
        embed_module = BertEmbedding().get_embedding()
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        embedding_layer = embed_module(input_layer)
        reshape_layer = tf.keras.layers.Reshape(target_shape=(1, 512))(embedding_layer)

        conv_layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(reshape_layer)
        gpooling_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
        dropout_layer = tf.keras.layers.Dropout(0.05)(gpooling_layer)

        conv_layer1 = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(dropout_layer)
        mpooling_layer1 = tf.keras.layers.MaxPooling1D()(conv_layer1)
        gru_layer = tf.keras.layers.GRU(128, activation='relu')(mpooling_layer1)
        dropout_layer1 = tf.keras.layers.Dropout(0.15)(gru_layer)

        flatten_layer = tf.keras.layers.Flatten()(dropout_layer1)
        dense_layer = tf.keras.layers.Dense(64, activation="relu")(flatten_layer)
        dropout_layer2 = tf.keras.layers.Dropout(0.25)(dense_layer)
        output_layer = self.get_output_layer()(dropout_layer2)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    def train(self, train_dataset, val_dataset, epochs=30, callbacks=[]):
        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
