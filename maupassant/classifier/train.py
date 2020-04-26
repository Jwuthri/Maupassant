import os
import shutil

from comet_ml import Experiment

from maupassant.settings import API_KEY, PROJECT_NAME, WORKSPACE
from maupassant.tensorflow_utils import TrainerHelper
from maupassant.dataset.tensorflow import TensorflowDataset


class Trainer(TrainerHelper):

    def __init__(
            self, train_df, test_df, val_df, label_type, architecture, feature, label,
            api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE, use_comet=True, epochs=30,
            multi_label=True, batch_size=512, buffer_size=512, embedding_type="multilingual"
    ):
        self.epochs = epochs
        self.label_type = label_type
        self.API_KEY = api_key
        self.PROJECT_NAME = project_name
        self.WORKSPACE = workspace
        self.use_comet = use_comet
        self.label = label
        self.feature = feature
        self.__dataset__ = TensorflowDataset(feature, label, multi_label, batch_size, buffer_size)
        self.train_dataset, self.test_dataset, self.val_dataset = self.__dataset__.main(train_df, test_df, val_df)
        self.label_encoder = {self.label: {"encoder": self.__dataset__.lb, "label_type": self.label_type, "id": 0}}
        super().__init__(label_type, architecture, self.__dataset__.nb_classes, embedding_type)

    def main(self):
        self.set_model()
        self.compile_model()
        paths = self.define_paths(self.label_type, self.label)
        os.mkdir(paths['path'])
        experiment = None

        if self.use_comet and self.API_KEY:
            experiment = Experiment(api_key=self.API_KEY, project_name=self.PROJECT_NAME, workspace=self.WORKSPACE)
            experiment.log_dataset_hash(self.train_dataset)
            experiment.add_tags(['tensorflow', self.label, self.architecture, self.embedding_type])
            experiment.log_parameters(dict(enumerate(self.__dataset__.lb.classes_)))
            with experiment.train():
                history = self.fit_model(self.train_dataset, self.val_dataset, epochs=self.epochs)
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(
                tensorboard_dir=paths['tensorboard_path'], checkpoint_path=paths['checkpoint_path']
            )
            history = self.fit_model(self.train_dataset, self.val_dataset, epochs=self.epochs, callbacks=callbacks)
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        macro_f1 = history.history["macro_f1"]
        val_macro_f1 = history.history["val_macro_f1"]
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

        return zip_model


if __name__ == '__main__':
    import pandas as pd
    from maupassant.settings import DATASET_PATH

    train_path = os.path.join(DATASET_PATH, "one_to_one", "train_intent.csv")
    test_path = os.path.join(DATASET_PATH, "one_to_one", "test_intent.csv")
    val_path = os.path.join(DATASET_PATH, "one_to_one", "val_intent.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)

    train = Trainer(
        train_df, test_df, val_df, "multi-label", "CNN_NN", "feature", "intent",
        epochs=10, multi_label=True, batch_size=300, buffer_size=512
    )
    model_path = train.main()
