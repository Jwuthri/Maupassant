import os

from comet_ml import Experiment

from maupassant.tensorflow_helper.dataset_helper import TensorflowDataset
from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE


class TensorflowTrainer(TensorflowDataset):
    """Module to train model."""

    def __init__(self, label_type, name, architecture, **kwargs):
        self.use_multilingual_embedding = kwargs.get('use_multilingual_embedding', True)
        self.embedding_size = kwargs.get('embedding_size', 256)
        self.input_size = kwargs.get('input_size', 128)
        self.epochs = kwargs.get('epochs', 10)
        self.use_comet = kwargs.get('use_comet', True)
        self.do_zip_model = kwargs.get('do_zip_model', True)
        self.api_key = kwargs.get('api_key', API_KEY)
        self.project_name = kwargs.get('project_name', PROJECT_NAME)
        self.workspace = kwargs.get('workspace', WORKSPACE)
        self.metrics = dict()
        super().__init__(architecture, label_type, name, **kwargs)

    def _train_with_comet(self, train_dataset, val_dataset):
        experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace)
        experiment.log_dataset_hash(train_dataset)
        experiment.add_tags([str(self.architecture), self.name, f"nb_labels_{self.label_encoder_classes_number}"])
        # with experiment.train():
        #     # hist = self.fit_dataset(train_dataset, val_dataset, self.epochs)
        #     pass
        # experiment.end()
        #
        # # return hist
    #
    # def _train_with_tensorboard(self,train_dataset, val_dataset):
    #     callbacks = checkpoint_tensorboard_callback(self.paths['checkpoint_path'], self.paths['tensorboard_path'])
    #     print(f"tensorboard --logdir {self.paths['tensorboard_path']}")
    #     # hist = self.fit_dataset(train_dataset, val_dataset, self.epochs, callbacks)
    #
    #     # return hist

    def train(self, data, x_col, y_col):
        os.mkdir(self.paths['path'])
        train_dataset, val_dataset = self.generate_dataset(data, x_col, y_col)
        if self.use_comet and self.api_key and self.project_name and self.workspace:
            self._train_with_comet(train_dataset, val_dataset)
        elif not self.use_comet:
            # self._train_with_tensorboard(train_dataset, val_dataset)
            pass
        else:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")