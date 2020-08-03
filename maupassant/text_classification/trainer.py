import os

from comet_ml import Experiment
import tensorflow as tf

from maupassant.dataset.dataset import TensorflowDataset
from maupassant.utils import text_format
from maupassant.tensorflow_metric_loss_optimizer import get_metrics
from maupassant.tensorflow_models_compile import BaseTensorflowModel
from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE, MODEL_PATH

try:
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except:
    pass


class Trainer(BaseTensorflowModel):

    def __init__(
        self, architecture, data, feature, label, buffer_size=512, label_type="multi-class",
        batch_size=512, base_path=MODEL_PATH, name="text_classification", epochs=30, keep_emot=True,
        api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE, use_comet=True
    ):
        multi_label = True if label_type == "multi-label" else False
        dataset_generator = TensorflowDataset(feature, label, multi_label, batch_size, buffer_size, keep_emot=keep_emot)
        self.train_dataset, self.test_dataset, self.val_dataset = dataset_generator.generate(data)
        self.encoder = dataset_generator.encoder
        self.number_labels = dataset_generator.number_labels
        self.use_comet = use_comet
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace
        self.epochs = epochs
        super().__init__(label_type, architecture, self.number_labels, True, base_path, name, False)
        self.model = self.build_model()
        self.compile_model()

    def train(self):
        os.mkdir(self.paths['path'])
        bg_fw = text_format(txt_color='white', bg_color='green', txt_style='bold')
        end_formatting = text_format(end=True)
        if self.use_comet and self.api_key and self.project_name and self.workspace:
            experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace)
            print(f"{bg_fw} Please go on the url: {experiment.url} {end_formatting}")
            experiment.log_dataset_hash(self.train_dataset)
            experiment.add_tags([str(self.architecture), "text_classification", f"nb_labels_{self.number_labels}"])
            with experiment.train():
                hist = self.fit_dataset(self.train_dataset, self.val_dataset, self.epochs)
            experiment.end()
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(
                tensorboard_dir=self.paths['tensorboard_path'], checkpoint_path=self.paths['checkpoint_path']
            )
            print(f"{bg_fw} tensorboard --logdir {self.paths['tensorboard_path']} {end_formatting}")
            hist = self.fit_dataset(self.train_dataset, self.val_dataset, self.epochs, callbacks)

        metrics = get_metrics(hist, "f1_score")
        self.export_weights(self.model)
        self.export_info(self.model_info)
        self.export_metrics(metrics)
        self.export_encoder(self.encoder)
        zip_model = self.zip_model()

        return zip_model


if __name__ == '__main__':
    import pandas as pd
    from maupassant.settings import DATASET_PATH

    dataset_path = os.path.join(DATASET_PATH, "train_intent.csv")
    data = pd.read_csv(dataset_path, nrows=10000)
    architecture = [('CNN', 512), ('LSTM', 512), ("FLATTEN", 0), ("DROPOUT", 0.2), ('DENSE', 1024)]
    feature, label, epochs, label_type = "feature", "intent", 2, "multi-label"
    trainer = Trainer(architecture, data, feature, label, label_type=label_type, epochs=epochs)
    trainer.train()
