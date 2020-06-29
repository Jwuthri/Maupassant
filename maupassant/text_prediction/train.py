import os
import ast
import json
import pickle
import shutil
import datetime

from comet_ml import Experiment
import tensorflow as tf

from maupassant.utils import timer
from maupassant.text_prediction.model import TensorflowModel
from maupassant.dataset.pandas import remove_rows_contains_null
from maupassant.text_prediction.dataset import DatasetGenerator
from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE, MODEL_PATH

tf.compat.v1.disable_eager_execution()


class TrainerHelper(TensorflowModel):
    """Tool to train model."""

    def __init__(self, label_type, architecture, number_labels, vocab_size):
        self.model = tf.keras.Sequential()
        super().__init__(label_type, architecture, number_labels, vocab_size)

    @staticmethod
    def callback_func(checkpoint_path, tensorboard_dir=None):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, period=2)
        if tensorboard_dir:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            return [tensorboard, checkpoint]
        else:
            return [checkpoint]

    @timer
    def fit_model(self, train_dataset, val_dataset, epochs=30, callbacks=[]):
        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

    @staticmethod
    def define_paths(classifier, label):
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        name = f"{classifier}_{label}_{date}"
        base_dir = os.path.join(MODEL_PATH, name)

        return {
            "path": base_dir,
            "model_path": os.path.join(base_dir, 'model'),
            "model_plot":  os.path.join(base_dir, "model.jpg"),
            "model_info": os.path.join(base_dir, "model.json"),
            "metrics_path": os.path.join(base_dir, "metrics.json"),
            "tensorboard_path": os.path.join(base_dir, "tensorboard"),
            "checkpoint_path": os.path.join(base_dir, "checkpoint"),
        }

    @staticmethod
    def export_model_plot(path, model):
        tf.keras.utils.plot_model(model, to_file=path, show_shapes=True)

    @staticmethod
    def export_model(path, model):
        model.save_weights(os.path.join(path, "model_weights"))
        print(f"Model has been exported here => {path}")

    @staticmethod
    def export_encoder(directory, label_data):
        for k, v in label_data.items():
            path = os.path.join(directory, f"{v['id']}_{v['label_type']}_{k}_encoder.pkl")
            pickle.dump(v['encoder'], open(path, "wb"))
            print(f"{k} encoder has been exported here => {path}")

    @staticmethod
    def export_pickle(directory, pkle, name):
        path = os.path.join(directory, f"{name}.pkl")
        pickle.dump(pkle, open(path, "wb"))

    @staticmethod
    def export_info(path, info):
        with open(path, 'w') as outfile:
            json.dump(info, outfile)
            print(f"Model information have been exported here => {path}")

    @staticmethod
    def export_metrics(path, metrics):
        with open(path, 'w') as outfile:
            json.dump(metrics, outfile)
            print(f"Model metrics have been exported here => {path}")


class Trainer(TrainerHelper):

    def __init__(
            self, dataset, architecture, feature, lang="english", words_predict=1, use_comet=True, epochs=10,
            batch_size=32, api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE):
        self.words_predict = words_predict
        self.lang = lang
        self.epochs = epochs
        self.label_type = "single-label"
        self.API_KEY = api_key
        self.PROJECT_NAME = project_name
        self.WORKSPACE = workspace
        self.use_comet = use_comet
        self.feature = feature
        self.__dataset__ = DatasetGenerator(batch_size=batch_size, words_to_predict=words_predict)
        self.train_dataset, self.test_dataset, self.val_dataset = self.__dataset__.generate(dataset[feature])
        self.label_encoder = {
            self.feature: {"encoder": self.__dataset__.le, "label_type": self.label_type, "id": self.words_predict}}
        super().__init__(self.label_type, architecture, self.__dataset__.max_labels, self.__dataset__.vocab_size)

    def main(self, pretrained_embedding=None):
        if pretrained_embedding:
            self.set_model_api(pretrained_embedding)
        else:
            self.set_model()
        self.compile_model()
        paths = self.define_paths(self.label_type, self.feature)
        os.mkdir(paths['path'])
        experiment = None

        if self.use_comet and self.API_KEY:
            experiment = Experiment(api_key=self.API_KEY, project_name=self.PROJECT_NAME, workspace=self.WORKSPACE)
            experiment.log_dataset_hash(self.train_dataset)
            experiment.add_tags(
                [
                    'tensorflow', self.feature, self.architecture, self.embedding_type,
                    self.lang, "words_prediction", self.words_predict
                ]
            )
            experiment.log_parameters(dict(enumerate(self.__dataset__.le.classes_)))
            with experiment.train():
                history = self.model.fit(
                    self.train_dataset[0], self.train_dataset[1], validation_data=self.val_dataset, epochs=self.epochs)
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(
                tensorboard_dir=paths['tensorboard_path'], checkpoint_path=paths['checkpoint_path']
            )
            history = self.model.fit(
                self.train_dataset[0], self.train_dataset[1],
                validation_data=self.val_dataset, epochs=self.epochs, callbacks=callbacks)
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        macro_f1 = history.history["sparse_categorical_accuracy"]
        val_macro_f1 = history.history["sparse_categorical_accuracy"]
        metrics = {"loss": loss, "val_loss": val_loss, "macro_f1": macro_f1, "val_macro_f1": val_macro_f1}
        metrics = {metric: [round(float(value), 5) for value in values] for metric, values in metrics.items()}

        self.export_model(paths['model_path'], self.model)
        self.export_encoder(paths['path'], self.label_encoder)
        self.export_model_plot(paths['model_plot'], self.model)
        self.export_info(paths["model_info"], self.info)
        self.export_metrics(paths["metrics_path"], metrics)
        zip_model = shutil.make_archive(
            paths['path'], "zip", os.path.dirname(paths['path']), os.path.basename(paths['path'])
        )
        if self.use_comet:
            experiment.log_image(paths['model_plot'])
            experiment.log_asset(zip_model)
            experiment.end()

        return self.model, self.__dataset__.le, self.__dataset__.tokenizer


def train(dataset, architecture, feature, max_words_pred, epochs):
    dict_models = {}
    dict_le = {}
    max_words_pred = min(5, max_words_pred)
    dataset = remove_rows_contains_null(dataset, feature)
    train = Trainer(dataset, architecture, feature, words_predict=1, epochs=epochs)
    model, le, tokenizer = train.main()
    dict_models[1] = model
    dict_le[1] = le
    embedding = model.layers[0]
    embedding.trainable = False
    if max_words_pred > 1:
        for word_predict in range(2, max_words_pred + 1):
            train = Trainer(dataset, architecture, feature, words_predict=word_predict, epochs=epochs)
            model, le, _ = train.main(embedding)
            dict_models[word_predict] = model
            dict_le[1] = le

    input_layer = tf.keras.Input((), name="input_layer")
    embedding_layer = embedding(input_layer)
    embedding_layer = tf.keras.layers.Reshape(target_shape=(1, 128))(embedding_layer)
    outputs = []
    for k, v in dict_models.items():
        layers = v.layers
        start = 1 if k == 1 else 2
        layer = layers[start](embedding_layer)
        for layer_ in layers[start + 1:]:
            layer = layer_(layer)
        outputs.append(layer)
    final_model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
    info = {
        "label_type": "single-label", "architecture": architecture, "vocab_size": len(tokenizer.word_index) + 1
    }
    paths = TrainerHelper.define_paths("single-label", feature)
    TrainerHelper.export_model(paths['model_path'], final_model)
    TrainerHelper.export_pickle(paths['path'], dict_le, 'encoder')
    TrainerHelper.export_pickle(paths['path'], tokenizer, 'tokenizer')
    TrainerHelper.export_info(paths["model_info"], info)

    return final_model


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from maupassant.settings import DATASET_PATH

    dataset_path = os.path.join(DATASET_PATH, "french_phrase.csv")
    dataset = pd.read_csv(dataset_path, nrows=100)
    final_model = train(dataset, "LSTM", 'agent_text', 2, 2)
    # x = np.asarray([0] * 127 + [24])
    # preds = final_model.predict(x)
    # for k, v in models_word_to_predict.items():
    #     k = k - 1
    #     le = v['label_encoder']
    #     tok = v['tokenizer']
    #     pred = preds[k][-1]
    #     best = np.argmax(pred)
    #     inv = le.inverse_transform([best])
    #     print(inv)
    #     print(inv[0])
    #     vals = ast.literal_eval(inv[0])
    #     for val in vals:
    #         print(tok.index_word[val])
