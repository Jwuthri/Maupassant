import tensorflow_text
import tensorflow as tf

from maupassant.feature_extraction.embedding import Embedding
from maupassant.tensorflow_utils import TrainerHelper


class Model(TrainerHelper):

    def __init__(self, type, nb_classes=1):
        super().__init__(type, nb_classes)

    def __model__(self, how='NN'):
        assert how in ['NN', 'CNN_NN', 'CNN_CNN_GRU']
        embed_module = Embedding().get_embedding()
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        layer = embed_module(input_layer)
        layer = tf.keras.layers.Reshape(target_shape=(1, 512))(layer)

        if how in ['CNN_NN', 'CNN_CNN_GRU']:
            layer = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu', strides=1)(layer)
            if how == 'CNN_CNN_GRU':
                layer = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1)(layer)
                layer = tf.keras.layers.GRU(128, activation='relu')(layer)
            else:
                layer = tf.keras.layers.GlobalMaxPooling1D()(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(250, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.2)(layer)
        layer = self.get_output_layer()(layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=layer)

    def fit(self, train_dataset, val_dataset, epochs=30, callbacks=[]):
        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
