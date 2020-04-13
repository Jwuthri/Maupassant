import shutil

from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from comet_ml import Experiment
import pandas as pd

from maupassant.settings import *
from maupassant.classifier.models import Model
from maupassant.dataset.tensorflow import TensorflowDataset


@dataclass()
class Trainer(Model):
    feature: str
    label: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: pd.DataFrame
    classification_type: str
    epochs: int = 30
    batch_size: int = 512
    use_comet: bool = False
    buffer_size: int = batch_size * 2
    model_architecture: str = "CNN_NN"
    lb: LabelEncoder = None
    nb_classes: int = 10
    api_key: str = API_KEY
    project_name: str = PROJECT_NAME
    workspace: str = WORKSPACE
    super().__init__(classification_type, nb_classes)

    def build_dataset(self):
        multi_label = True if self.classification_type == "multi" else False
        td = TensorflowDataset(self.feature, self.label, multi_label, self.batch_size, self.buffer_size)
        tf_train, tf_test, tf_val = td.tf_dataset(self.train_df, self.test_df, self.val_df)
        self.lb = td.lb
        self.nb_classes = td.nb_classes

        return tf_train, tf_test, tf_val

    def fit(self, tf_train, tf_val):
        label_data = {self.label: {"encoder": self.lb, "classification": self.classifier, "id": 0}}
        info = {self.label: {"model_type": self.model_type, "classification": self.classifier}}
        training_path = self.define_training_path(self.classification_type, self.label)
        model_dir, plot_path, info_path, lb_dir, tensorboard_dir, ckpt_path = training_path
        self.model = self.__model__(how=self.model_architecture)
        self.compile_model()
        self.plot_model(plot_path)

        if self.use_comet and self.api_key:
            callbacks = self.callback_func(checkpoint_path=tensorboard_dir)
            experiment = Experiment(api_key=self.api_key, project_name=self.project_name, workspace=self.workspace)
            experiment.log_dataset_hash(tf_train)
            experiment.add_tags(['multi_lang', 'tensorflow', self.label, self.classification_type])
            experiment.log_parameters(dict(enumerate(self.lb.classes_)))
            with experiment.train():
                _ = self.fit_model(tf_train, tf_val, epochs=self.epochs, callbacks=callbacks)
        elif self.use_comet:
            raise Exception("Please provide an api_key, project_name and workspace for comet_ml")
        else:
            callbacks = self.callback_func(tensorboard_dir=tensorboard_dir, checkpoint_path=tensorboard_dir)
            history = self.train(tf_train, tf_val, epochs=self.epochs, callbacks=callbacks)
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            macro_f1 = history.history["macro_f1"]
            val_macro_f1 = history.history["val_macro_f1"]
            metrics = {"loss": loss, "val_loss": val_loss, "macro_f1": macro_f1, "val_macro_f1": val_macro_f1}
            print(metrics)

        self.export_model(model_dir)
        self.export_encoder(lb_dir, label_data)
        self.export_info(info, info_path)
        zip_model = shutil.make_archive(lb_dir, "zip", os.path.dirname(lb_dir), os.path.basename(lb_dir))
        print(zip_model)

        if self.use_comet:
            experiment.log_image(plot_path)
            experiment.log_asset(zip_model)
            experiment.send_notification('Finished')
            experiment.end()

        return self.model

    def evaluate(self, tf_test):
        raise NotImplemented


if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(DATASET_PATH, "intent_data_multi_label_train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_PATH, "intent_data_multi_label_val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, "intent_data_multi_label_test.csv"))

    train = Trainer(
        feature='feature',
        label='sentiment',
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        classification_type='multi',
        epochs=30,
        batch_size=512,
        use_comet=True,
        model_architecture="CNN_GRU_NN"
    )
    tf_train, tf_test, tf_val = train.build_dataset()
    model = train.fit(tf_train, tf_val)
