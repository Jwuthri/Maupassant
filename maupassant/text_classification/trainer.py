import os

from comet_ml import Experiment

from maupassant.text_classification.dataset import BuildDataset
from maupassant.tensorflow_metric_loss_optimizer import get_metrics
from maupassant.tensorflow_models_compile import BaseTensorflowModel
from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE, MODEL_PATH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Trainer(BaseTensorflowModel):

    def __init__(
        self, data, x, y, label_type, architecture, buffer_size=512, pretrained_embedding=True, test_size=0.1,
        batch_size=512, base_path=MODEL_PATH, name="text_classification", input_shape=64, embedding_size=128, epochs=30,
        api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE, use_comet=True, do_zip_model=True,
    ):
        dataset_generator = BuildDataset(label_type=label_type, batch_size=batch_size, buffer_size=buffer_size)
        self.train_dataset, self.val_dataset = dataset_generator.generate(data, x, y, test_size=test_size)
        self.encoder = dataset_generator.encoder
        self.classes_mapping = dataset_generator.classes_mapping
        self.number_labels = dataset_generator.number_labels
        self.name = name
        self.pretrained_embedding = pretrained_embedding
        self.embedding_size = embedding_size
        self.input_shape = input_shape
        self.use_comet = use_comet
        self.api_key = api_key
        self.project_name = project_name
        self.do_zip_model = do_zip_model
        self.workspace = workspace
        self.epochs = epochs
        self.metrics = {}
        super().__init__(
            label_type, architecture, self.number_labels, self.pretrained_embedding, base_path, name, model_load=False)
        self.build_model()
        self.compile_model()

    def train(self):
        os.mkdir(self.paths['path'])
        if self.use_comet and self.api_key and self.project_name and self.workspace:
            experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace)
            experiment.log_dataset_hash(self.train_dataset)
            experiment.add_tags([str(self.architecture), self.name, f"nb_labels_{self.number_labels}"])
            with experiment.train():
                hist = self.fit_dataset(self.train_dataset, self.val_dataset, self.epochs)
            experiment.end()
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(
                tensorboard_dir=self.paths['tensorboard_path'], checkpoint_path=self.paths['checkpoint_path']
            )
            print(f"tensorboard --logdir {self.paths['tensorboard_path']}")
            hist = self.fit_dataset(self.train_dataset, self.val_dataset, self.epochs, callbacks)

        if self.label_type == "binary-class":
            metric = "binary_accuracy"
        elif self.label_type == "multi-label":
            metric = "f1_score"
        else:
            metric = "sparse_categorical_accuracy"

        self.metrics = get_metrics(hist, metric)
        self.export_weights(self.model)
        self.export_info(self.model_info)
        self.export_metrics(self.metrics)
        self.export_encoder(self.encoder)
        if self.do_zip_model:
            self.zip_model()


if __name__ == '__main__':
    import pandas as pd
    from maupassant.settings import DATASET_PATH
    from maupassant.dataset.pandas import remove_rows_contains_null

    dataset_path = os.path.join(DATASET_PATH, "sentiment.csv")
    dataset = pd.read_csv(dataset_path, nrows=10000)
    # ['binary-class', 'multi-label', 'multi-class']
    x, y, label_type, epochs = "feature", "multi", "multi-label", 2
    dataset = remove_rows_contains_null(dataset, x)
    dataset = remove_rows_contains_null(dataset, y)
    architecture = [('LSTM', 512), ("DROPOUT", 0.2), ('DENSE', 1024)]
    trainer = Trainer(dataset, x, y, label_type, architecture, epochs=epochs, use_comet=True)
    trainer.train()
