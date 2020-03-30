import shutil
import datetime

import pandas as pd
from comet_ml import Experiment

from maupassant.settings import *
from maupassant.classifier.models import Model
from maupassant.dataset.tensorflow import TensorflowDataset


class TrainClassifier(TensorflowDataset, Model):

    def __init__(
            self, train_path, test_path, val_path, feature, label, use_comet=False,
            model_type='intermediate', batch_size=512, buffer_size=1024, epochs=30, classifier='multi'
    ):
        assert classifier in ['binary', 'single', 'multi']
        assert model_type in ['basic', 'intermediate', 'advanced']
        multi_label = True if classifier == 'multi' else False
        super().__init__(feature, label, multi_label, batch_size, buffer_size)
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.feature = feature
        self.label = label
        self.use_comet = use_comet
        self.model_type = model_type
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.classifier = classifier
        self.experiment = None
        self.model = None
        self.bm = None

    def define_path(self):
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        name = f"{self.classifier}_{self.label}_{date}"
        model_dir = os.path.join(MODEL_PATH, name, 'model')
        plot_path = os.path.join(MODEL_PATH, name, f"{name}.jpg")
        info_path = os.path.join(MODEL_PATH, name, "info.json")
        encoder_dir = os.path.join(MODEL_PATH, name)
        tensorboard_dir = os.path.join(LOGS_PATH, f"tensorboard/{name}")
        checkpoint_path = os.path.join(LOGS_PATH, f"checkpoint/{name}")

        return model_dir, plot_path, info_path, encoder_dir, tensorboard_dir, checkpoint_path

    def fit(self):
        test = self.clean_dataset(pd.read_csv(self.test_path, nrows=1000))
        val = self.clean_dataset(pd.read_csv(self.val_path, nrows=1000))
        train = self.clean_dataset(pd.read_csv(self.train_path, nrows=10000))

        x_test, y_test = self.split_x_y(test)
        x_val, y_val = self.split_x_y(val)
        x_train, y_train = self.split_x_y(train)

        self.fit_lb(y_train)
        y_val_encoded = self.transform_lb(y_val)
        y_test_encoded = self.transform_lb(y_test)
        y_train_encoded = self.transform_lb(y_train)

        val_dataset = self.to_tensorflow_dataset(x_val, y_val_encoded)
        train_dataset = self.to_tensorflow_dataset(x_train, y_train_encoded)
        test_dataset = self.to_tensorflow_dataset(x_test, y_test_encoded)

        label_data = {self.label: {"encoder": self.lb, "classification": self.classifier, "id": 0}}
        info = {self.label: {"model_type": self.model_type, "classification": self.classifier}}

        model_dir, plot_path, info_path, encoder_dir, tensorboard_dir, checkpoint_path = self.define_path()
        self.bm = Model(type=self.classifier, nb_classes=self.nb_classes)
        self.model = self.bm.set_model(how=self.model_type)
        self.bm.compile_model()
        self.bm.plot_model(plot_path)

        if self.use_comet:
            experiment = Experiment(api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE)
            experiment.log_dataset_hash(train)
            experiment.add_tags(['multi_lang', 'tensorflow', "one_to_one", self.label])
            experiment.log_parameters(dict(enumerate(self.lb.classes_)))
            with experiment.train():
                _ = self.bm.fit(train_dataset, val_dataset, epochs=self.epochs)
        else:
            callbacks = self.callback_func(tensorboard_dir=tensorboard_dir, checkpoint_path=checkpoint_path)
            history = self.train(train_dataset, val_dataset, epochs=self.epochs, callbacks=callbacks)
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            macro_f1 = history.history["macro_f1"]
            val_macro_f1 = history.history["val_macro_f1"]
            metrics = {"loss": loss, "val_loss": val_loss, "macro_f1": macro_f1, "val_macro_f1": val_macro_f1}
            print(metrics)

        self.bm.export_model(model_dir)
        self.bm.export_encoder(encoder_dir, label_data)
        self.bm.export_info(info, info_path)
        zip_model = shutil.make_archive(encoder_dir, "zip", os.path.dirname(encoder_dir), os.path.basename(encoder_dir))
        print(zip_model)

        if self.use_comet:
            experiment.log_image(plot_path)
            experiment.log_asset(zip_model)
            experiment.send_notification('Finished')
            experiment.end()


if __name__ == '__main__':
    train_path = os.path.join(DATASET_PATH, "intent_data_multi_label_train.csv")
    val_path = os.path.join(DATASET_PATH, "intent_data_multi_label_val.csv")
    test_path = os.path.join(DATASET_PATH, "intent_data_multi_label_test.csv")
    Train(
        train_path, test_path, val_path, "feature", "intent", use_comet=True, model_type='advanced',
        batch_size=512, buffer_size=1024, epochs=2, classifier='multi'
    ).fit()
