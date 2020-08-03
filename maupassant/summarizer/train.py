import os
import json
import pickle
import shutil
import datetime

from comet_ml import Experiment
import tensorflow as tf

from maupassant.utils import timer
from maupassant.tensorflow_metric_loss_optimizer import f1_score
from maupassant.dataset.label_encoder import LabelEncoding
from maupassant.summarizer.model import TensorflowModel
from maupassant.dataset.pandas import remove_rows_contains_null
from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE, MODEL_PATH


class TrainerHelper(TensorflowModel):

    def __init__(self, architecture, embedding_type, text=None):
        self.model = tf.keras.Sequential()
        super().__init__(architecture, embedding_type, text)

    def compile_model(self):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[f1_score, "binary_accuracy"])

    @staticmethod
    def callback_func(checkpoint_path, tensorboard_dir=None):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, period=5)
        if tensorboard_dir:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            return [tensorboard, checkpoint]
        else:
            return [checkpoint]

    @timer
    def fit_model(self, train_dataset, val_dataset, epochs=30, callbacks=[]):
        return self.model.fit(x=train_dataset[0], y=train_dataset[1], epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

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
        tf.keras.experimental.export_saved_model(model, path)
        print(f"Model has been exported here => {path}")

    @staticmethod
    def export_encoder(directory, label_data):
        for k, v in label_data.items():
            path = os.path.join(directory, f"{v['id']}_{v['label_type']}_{k}_encoder.pkl")
            pickle.dump(v['encoder'], open(path, "wb"))
            print(f"{k} encoder has been exported here => {path}")

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
        self, train_df, test_df, val_df, architecture, feature, label, text=None,
        api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE, use_comet=True,
        batch_size=512, buffer_size=512, embedding_type="multilingual", epochs=30
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.epochs = epochs
        self.API_KEY = api_key
        self.PROJECT_NAME = project_name
        self.WORKSPACE = workspace
        self.use_comet = use_comet
        self.label = label
        self.feature = feature
        self.text = text
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        super().__init__(architecture, embedding_type, text)

    def get_dataset(self, feature, label, text=None):
        train_dataset = remove_rows_contains_null(self.train_df, feature)
        train_dataset = remove_rows_contains_null(train_dataset, label)
        if self.text:
            train_dataset = remove_rows_contains_null(train_dataset, text)
        test_dataset = remove_rows_contains_null(self.test_df, feature)
        test_dataset = remove_rows_contains_null(test_dataset, label)
        if self.text:
            test_dataset = remove_rows_contains_null(test_dataset, text)
        val_dataset = remove_rows_contains_null(self.val_df, feature)
        val_dataset = remove_rows_contains_null(val_dataset, label)
        if self.text:
            val_dataset = remove_rows_contains_null(val_dataset, text)

        return train_dataset, test_dataset, val_dataset

    def main(self):
        self.set_model()
        self.compile_model()
        paths = self.define_paths(self.label_type, self.label)
        os.mkdir(paths['path'])
        experiment = None
        train_dataset, test_dataset, val_dataset = self.get_dataset(self.label, self.feature, self.text)
        le = LabelEncoding(False)
        le.fit_encoder(train_dataset[self.label].values)
        label_encoder = {self.label: {"encoder": le.encoder, "label_type": "binary-label", "id": 0}}

        if self.text:
            train_dataset = (
                {"input_sentences": train_dataset[self.feature].values, "input_text": train_dataset[self.text].values},
                {"is_relevant": train_dataset[self.label].values}
            )
            val_dataset = (
                {"input_sentences": val_dataset[self.feature].values, "input_text": val_dataset[self.text].values},
                {"is_relevant": val_dataset[self.label].values}
            )
        else:
            train_dataset = (
                {"input_sentences": train_dataset[self.feature].values},
                {"is_relevant": train_dataset[self.label].values}
            )
            val_dataset = (
                {"input_sentences": val_dataset[self.feature].values},
                {"is_relevant": val_dataset[self.label].values}
            )
        del self.train_df
        del self.test_df
        del self.val_df

        if self.use_comet and self.API_KEY:
            experiment = Experiment(api_key=self.API_KEY, project_name=self.PROJECT_NAME, workspace=self.WORKSPACE)
            experiment.log_dataset_hash(train_dataset)
            experiment.add_tags(['tensorflow', self.label, self.architecture, self.embedding_type])
            with experiment.train():
                history = self.fit_model(train_dataset, val_dataset, epochs=self.epochs)
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(
                tensorboard_dir=paths['tensorboard_path'], checkpoint_path=paths['checkpoint_path']
            )
            history = self.fit_model(train_dataset, val_dataset, epochs=self.epochs, callbacks=callbacks)
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        macro_f1 = history.history["macro_f1"]
        val_macro_f1 = history.history["val_macro_f1"]
        metrics = {"loss": loss, "val_loss": val_loss, "macro_f1": macro_f1, "val_macro_f1": val_macro_f1}
        metrics = {metric: [round(float(value), 5) for value in values] for metric, values in metrics.items()}
        self.info['label'] = self.label
        self.info['first_input'] = self.feature
        self.info['second_label'] = self.text

        self.export_model(paths['model_path'], self.model)
        self.export_encoder(paths['path'], label_encoder)
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

        return zip_model


if __name__ == '__main__':
    import pandas as pd
    from sklearn.utils import shuffle
    from maupassant.settings import DATASET_PATH

    train_path = os.path.join(DATASET_PATH, "one_to_one", "train_summarization.csv")
    test_path = os.path.join(DATASET_PATH, "one_to_one", "val_summarization.csv")
    val_path = os.path.join(DATASET_PATH, "one_to_one", "val_summarization.csv")

    train_df = shuffle(pd.read_csv(train_path))
    test_df = shuffle(pd.read_csv(test_path))
    val_df = shuffle(pd.read_csv(val_path))

    train = Trainer(
        train_df, test_df, val_df, architecture="CNN_LSTM", feature="sentences", label="is_relevant", text=None,
        epochs=2, batch_size=1024, buffer_size=1024
    )
    model_path = train.main()
