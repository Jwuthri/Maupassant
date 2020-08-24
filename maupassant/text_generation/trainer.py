import os

from comet_ml import Experiment
import tensorflow as tf

from maupassant.text_generation.dataset import BuildDataset
from maupassant.tensorflow_metric_loss_optimizer import get_metrics
from maupassant.tensorflow_models_compile import BaseTensorflowModel
from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE, MODEL_PATH

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()


class Trainer(BaseTensorflowModel):

    def __init__(
        self, architecture, number_labels_max, data,
        batch_size=512, base_path=MODEL_PATH, name="text_generation", input_shape=64, embedding_size=128, epochs=30,
        api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE, use_comet=True
    ):
        dataset_generator = BuildDataset(batch_size=batch_size, input_shape=input_shape, max_labels=number_labels_max)
        self.train_dataset, self.val_dataset = dataset_generator.generate(data)
        self.tokenizer = dataset_generator.tokenizer
        self.vocab_size = dataset_generator.vocab_size
        self.number_labels = dataset_generator.number_labels
        self.embedding_size = embedding_size
        self.input_shape = input_shape
        self.use_comet = use_comet
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace
        self.epochs = epochs
        super().__init__("multi-class", architecture, self.number_labels, False, base_path, name, False)
        self.model = self.build_model(self.input_shape, embedding_size=self.embedding_size, vocab_size=self.vocab_size)
        self.compile_model()

    def train(self):
        os.mkdir(self.paths['path'])
        if self.use_comet and self.api_key and self.project_name and self.workspace:
            experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace)
            experiment.log_dataset_hash(self.train_dataset)
            experiment.add_tags([str(self.architecture), "text_generation", f"nb_labels_{self.number_labels}"])
            with experiment.train():
                hist = self.model.fit_dataset(self.train_dataset, self.val_dataset, self.epochs)
            experiment.end()
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(
                tensorboard_dir=self.paths['tensorboard_path'], checkpoint_path=self.paths['checkpoint_path']
            )
            hist = self.model.fit_dataset(self.train_dataset, self.val_dataset, self.epochs, callbacks)

        metrics = get_metrics(hist, "sparse_categorical_crossentropy")
        self.export_weights(self.model)
        self.export_info(self.model_info)
        self.export_metrics(metrics)
        self.export_tokenizer(self.tokenizer)
        self.zip_model()


if __name__ == '__main__':
    import pandas as pd
    from maupassant.settings import DATASET_PATH
    from maupassant.dataset.pandas import remove_rows_contains_null

    dataset_path = os.path.join(DATASET_PATH, "french_phrase_300k.csv")
    dataset = pd.read_csv(dataset_path)
    dataset = remove_rows_contains_null(dataset, "cleaned_agent")
    architecture = [('RNN', 512), ('DENSE', 1024)]
    number_labels_max = 5000
    data = dataset['cleaned_agent'].values
    trainer = Trainer(architecture, number_labels_max, data)
    trainer.train()
