import os

import colorful as cf
from comet_ml import Experiment

from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE
from maupassant.tensorflow_helper.metrics_helper import get_metrics
from maupassant.tensorflow_helper.dataset_helper import TensorflowDataset
from maupassant.tensorflow_helper.callbacks_helper import checkpoint_callback, tensorboard_callback


class TensorflowTrainer(TensorflowDataset):
    """Module to train model."""

    def __init__(self, label_type, name, architecture, **kwargs):
        self.epochs = kwargs.get('epochs', 10)
        self.use_comet = kwargs.get('use_comet', True)
        self.do_zip_model = kwargs.get('do_zip_model', True)
        self.api_key = kwargs.get('api_key', API_KEY)
        self.project_name = kwargs.get('project_name', PROJECT_NAME)
        self.workspace = kwargs.get('workspace', WORKSPACE)
        self.metrics = dict()
        super().__init__(architecture, label_type, name, **kwargs)

    def fit_dataset(self, train_dataset, val_dataset, callbacks=None):
        callbacks = [] if not callbacks else callbacks

        return self.model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset, callbacks=callbacks)

    def fit_numpy(self, x, y, x_val, y_val, callbacks=None):
        callbacks = [] if not callbacks else callbacks

        return self.model.fit(x, y, epochs=self.epochs, validation_data=(x_val, y_val), callbacks=callbacks)

    def _train_with_comet(self, train_dataset, val_dataset):
        experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace)
        experiment.log_dataset_hash(train_dataset)
        experiment.add_tags([str(self.architecture), self.name, f"nb_labels_{self.label_encoder_classes_number}"])
        with experiment.train():
            hist = self.fit_dataset(train_dataset, val_dataset)
        experiment.end()

        return hist

    def _train_with_tensorboard(self, train_dataset, val_dataset):
        callbacks = [
            tensorboard_callback(self.paths['tensorboard_path']),
            checkpoint_callback(self.paths['checkpoint_path'])
        ]
        print(f"{cf.bold_magenta}tensorboard --logdir {self.paths['tensorboard_path']}{cf.reset}")
        hist = self.fit_dataset(train_dataset, val_dataset, callbacks)

        return hist

    @property
    def which_metrics_used(self):
        if self.label_type == "binary-class":
            metric = "binary_accuracy"
        elif self.label_type == "multi-label":
            metric = "f1_score"
        else:
            metric = "categorical_accuracy"

        return metric

    def train(self, data, x_col, y_col):
        os.mkdir(self.paths['path'])
        train_dataset, val_dataset = self.generate_dataset(data, x_col, y_col)
        self.build_model()
        self.compile_model()

        if self.use_comet and self.api_key and self.project_name and self.workspace:
            hist = self._train_with_comet(train_dataset, val_dataset)
        elif not self.use_comet:
            hist = self._train_with_tensorboard(train_dataset, val_dataset)
        else:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")

        self.metrics = get_metrics(hist, self.which_metrics_used)
        self.export_weights(self.model)
        self.export_info(self.model_info)
        self.export_metrics(self.metrics)
        self.export_label_encoder(self.label_encoder)
        if self.do_zip_model:
            self.zip_model()
