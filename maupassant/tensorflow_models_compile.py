import os
import tensorflow as tf

from maupassant.feature_extraction.pretrained_embedding import PretrainedEmbedding
from maupassant.tensorflow_metric_loss_optimizer import f1_score, f1_loss
from maupassant.settings import MODEL_PATH
from maupassant.utils import ModelSaverLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class BaseTensorflowModel(ModelSaverLoader):

    def __init__(
            self, label_type, architecture, number_labels, pretrained_embedding,
            base_path=MODEL_PATH, name="text_classification", model_load=False):
        super().__init__(base_path, name, model_load)
        self.label_type = label_type
        self.architecture = architecture
        self.number_labels = number_labels
        self.pretrained_embedding = pretrained_embedding
        self.model = tf.keras.Sequential()
        self.model_info = {
            "architecture": architecture, "label_type": label_type,
            "pretrained_embedding": pretrained_embedding, "number_labels": number_labels,
        }

    def get_input_layer(self, input_size, embedding_size, vocab_size, name="input_layer"):
        if self.pretrained_embedding:
            input_layer = tf.keras.Input((), dtype=tf.string, name=name)
            layer = PretrainedEmbedding(name="embedding_layer").model(input_layer)
            layer = tf.keras.layers.Reshape(target_shape=(1, 512))(layer)
            self.model_info['embedding_size'] = 512
        else:
            input_layer = tf.keras.Input((input_size), name=name)
            layer = tf.keras.layers.Embedding(vocab_size, embedding_size, name="embedding_layer")(input_layer)
            self.model_info['input_size'] = input_size
            self.model_info['vocab_size'] = vocab_size
            self.model_info['embedding_size'] = embedding_size

        return input_layer, layer

    def get_output_layer(self, name="output_layer"):
        if self.label_type == "binary-class":
            output = tf.keras.layers.Dense(units=1, activation="sigmoid", name=name)
        elif self.label_type == "multi-label":
            output = tf.keras.layers.Dense(units=self.number_labels, activation="sigmoid", name=name)
        elif self.label_type == "multi-class":
            output = tf.keras.layers.Dense(units=self.number_labels, activation="softmax", name=name)
        else:
            raise(Exception("Please provide a 'label_type' in ['binary-class', 'multi-label', 'multi-class']"))

        return output

    def build_model(self, input_size=None, embedding_size=None, vocab_size=None):
        input_layer, layer = self.get_input_layer(input_size, embedding_size, vocab_size)
        for block, unit in self.architecture:
            if block == "CNN":
                layer = tf.keras.layers.Conv1D(unit, kernel_size=3, strides=1, padding='same', activation='relu')(layer)
            elif block == "LSTM":
                layer = tf.keras.layers.LSTM(unit, activation='relu')(layer)
            elif block == "GRU":
                layer = tf.keras.layers.GRU(unit, activation='relu')(layer)
            elif block == "RNN":
                layer = tf.keras.layers.SimpleRNN(unit, activation='relu')(layer)
            elif block == "DENSE":
                layer = tf.keras.layers.Dense(unit, activation="relu")(layer)
            elif block == "FLATTEN":
                layer = tf.keras.layers.Flatten()(layer)
            elif block == "DROPOUT":
                layer = tf.keras.layers.Dropout(unit)(layer)
            elif block == "GLOBAL_POOL":
                layer = tf.keras.layers.GlobalMaxPooling1D()(layer)
            elif block == "MAX_POOL":
                layer = tf.keras.layers.MaxPool1D()(layer)
        output_layer = self.get_output_layer()(layer)

        return tf.keras.models.Model(inputs=input_layer, outputs=[output_layer])

    def compile_model(self):
        if self.label_type == "binary-class":
            self.model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=[f1_score, "binary_accuracy"])
        elif self.label_type == "multi-label":
            self.model.compile(
                optimizer="adam", loss=f1_loss,
                metrics=[f1_score, "categorical_accuracy", "top_k_categorical_accuracy"])
        elif self.label_type == "multi-class":
            self.model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy",
                metrics=[f1_score, "sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"])
        else:
            raise(Exception("Please provide a 'label_type' in ['binary-class', 'multi-label', 'multi-class']"))

    @staticmethod
    def callback_func(checkpoint_path, tensorboard_dir=None):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1, period=1, save_weights_only=True)
        if tensorboard_dir:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            return [tensorboard, checkpoint]
        else:
            return [checkpoint]

    def fit_dataset(self, train_dataset, val_dataset, epochs=30, callbacks=None):
        callbacks = [] if not callbacks else callbacks

        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

    def fit_numpy(self, x, y, x_val, y_val, epochs=30, callbacks=None):
        callbacks = [] if not callbacks else callbacks

        return self.model.fit(x, y, epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks)
