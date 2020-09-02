import os
import glob
import json
import pickle
import shutil
import datetime

import tensorflow as tf

from maupassant.feature_extraction.pretrained_embedding import PretrainedEmbedding
from maupassant.tensorflow_metric_loss_optimizer import f1_score, f1_loss
from maupassant.settings import MODEL_PATH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ModelSaverLoader(object):

    def __init__(self, base_path, name, model_load):
        self.base_path = base_path
        self.name = name
        self.model_load = model_load
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.paths = self.define_paths()

    def define_paths(self):
        if self.model_load:
            base_dir = os.path.join(self.base_path, self.name)
        else:
            base_dir = os.path.join(self.base_path, f"{self.date}_{self.name}")

        return {
            "path": base_dir,
            "weights_path":  os.path.join(base_dir, 'weights'),
            "model_path": os.path.join(base_dir, 'model.pkl'),
            "model_plot_path":  os.path.join(base_dir, "model.jpg"),
            "model_info_path": os.path.join(base_dir, "model.json"),
            "metrics_path": os.path.join(base_dir, "metrics.json"),
            "label_encoder_path": os.path.join(base_dir, "label_encoder.pkl"),
            "tokenizer_path": os.path.join(base_dir, "tokenizer.pkl"),
            "tensorboard_path": os.path.join(base_dir, "tensorboard"),
            "checkpoint_path": os.path.join(base_dir, "checkpoint"),
        }

    def export_tf_model_plot(self, model):
        tf.keras.utils.plot_model(model, to_file=self.paths['model_plot_path'], show_shapes=True)

    def export_weights(self, model):
        model.save_weights(self.paths['weights_path'])
        print(f"Model has been exported here => {self.paths['weights_path']}")

    def load_weights(self, model):
        latest = tf.train.latest_checkpoint(self.paths['path'])
        model.load_weights(latest)

        return model

    def export_encoder(self, label_encoder):
        pickle.dump(label_encoder, open(self.paths['label_encoder_path'], "wb"))
        print(f"Label Encoder has been exported here => {self.paths['label_encoder_path']}")

    def load_encoder(self):
        encoders_paths = glob.glob(os.path.join(self.paths['path'], "label_encoder*"))
        encoders = []
        for encoder_path in encoders_paths:
            encoder = pickle.load(open(encoder_path, "rb"))
            encoders.append(encoder)

        return encoders

    def export_info(self, info):
        with open(self.paths['model_info_path'], 'w') as outfile:
            json.dump(info, outfile)
            print(f"Model information have been exported here => {self.paths['model_info_path']}")

    def load_info(self):
        with open(self.paths['model_info_path'], "rb") as json_file:
            info = json.load(json_file)

        return info

    def export_metrics(self, metrics):
        with open(self.paths['metrics_path'], 'w') as outfile:
            json.dump(metrics, outfile)
            print(f"Model metrics have been exported here => {self.paths['metrics_path']}")

    def load_metrics(self):
        with open(self.paths['metrics_path'], "rb") as json_file:
            metrics = json.load(json_file)

        return metrics

    def export_tokenizer(self, tokenizer):
        pickle.dump(tokenizer, open(self.paths['tokenizer_path'], "wb"))
        print(f"Tokenizer has been exported here => {self.paths['tokenizer_path']}")

    def load_tokenizer(self):
        tokenizer = pickle.load(open(self.paths['tokenizer_path'], "rb"))

        return tokenizer

    def export_model(self, model):
        pickle.dump(model, open(self.paths['model_path'], "wb"))
        print(f"Model has been exported here => {self.paths['model_path']}")

    def load_model(self):
        model = pickle.load(open(self.paths['model_path'], "rb"))

        return model

    def zip_model(self):
        zip_path = shutil.make_archive(
            self.paths['path'], "zip", os.path.dirname(self.paths['path']), os.path.basename(self.paths['path'])
        )

        return zip_path


class BaseTensorflowModel(ModelSaverLoader):

    def __init__(self, label_type, architecture, number_labels, pretrained_embedding, base_path=MODEL_PATH, name="text_classification", model_load=False):
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

    def get_input_layer(self, input_size, embedding_size, vocab_size, embedding_layer, name="input_layer"):
        if self.pretrained_embedding:
            input_layer = tf.keras.Input((), dtype=tf.string, name=name)
            if embedding_layer:
                layer = embedding_layer(input_layer)
            else:
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

    def build_model(self, input_size=None, embedding_size=None, vocab_size=None, embedding_layer=None):
        input_layer, layer = self.get_input_layer(input_size, embedding_size, vocab_size, embedding_layer)
        for block, unit in self.architecture:
            if block == "CNN":
                layer = tf.keras.layers.Conv1D(unit, kernel_size=3, strides=1, padding='same', activation='relu')(layer)
            elif block == "LCNN":
                layer = tf.keras.layers.LocallyConnected1D(
                    unit, kernel_size=3, strides=1, padding='valid', activation='relu')(layer)
            elif block == "BiLSTM":
                layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unit, activation="relu"))(layer)
            elif block == "BiGRU":
                layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(unit, activation="relu"))(layer)
            elif block == "BiRNN":
                layer = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(unit, activation="relu"))(layer)
            elif block == "CudaLSTM":
                layer = tf.compat.v1.keras.layers.CuDNNLSTM(unit)(layer)
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
            elif block == "RESHAPE":
                layer = tf.keras.layers.Reshape(target_shape=unit)(layer)
        output_layer = self.get_output_layer()(layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self):
        if self.label_type == "binary-class":
            self.model.compile(
                optimizer="adam", loss="binary_crossentropy",
                metrics=[f1_score, "binary_accuracy", "Recall", "Precision"])
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
    def callback_func(checkpoint_path: str, tensorboard_dir: str=None):
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


class MergeModels(BaseTensorflowModel):

    def __init__(self, model1_path, model2_path, base_path, name, model_load):
        super().__init__("multi-class", architecture, self.number_labels, False, base_path, name, False)
        self.model1_path = model1_path
        self.model2_path = model2_path

    def merge(self, model1_layer_start=3, model2_layer_start=3, model1_layer_end=None, model2_layer_end=None):
        input_layer = tf.keras.Input((), dtype=tf.string, name="input_layer")
        layer_input = PretrainedEmbedding(name="embedding_layer").model(input_layer)
        layer_input = tf.keras.layers.Reshape(target_shape=(1, 512))(layer_input)
        lst = [(self.model1, model1_layer_start, model1_layer_end), (self.model2, model2_layer_start, model2_layer_end)]
        outputs = []
        for model, start_layer, end_layer in lst:
            layer = model.layers()[start_layer](layer_input)
            for block in model.layers()[model1_layer_start + 1:end_layer]:
                layer = block(layer)
            outputs.append(layer)
        merged_model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)

        return merged_model
