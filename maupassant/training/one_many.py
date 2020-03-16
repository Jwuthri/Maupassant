import pickle
import datetime

import pandas as pd
from comet_ml import Experiment

from maupassant.settings import *
from maupassant.dataset.labels import LabelEncoding
from maupassant.utils import mark_format, text_format
from maupassant.dataset.tensorflow import TensorflowDataset
from maupassant.settings import MODEL_PATH, DATASET_PATH, LOGS_PATH
from maupassant.suppervised.one_to_many_classifier import TensorflowClassifier


def define_path(classifier):
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = f"{classifier}_{date}"
    model_dir = os.path.join(MODEL_PATH, name)
    tensorboard_dir = os.path.join(LOGS_PATH, f"tensorboard/{name}")
    plot_file = os.path.join(LOGS_PATH, f"plot/{name}.jpg")
    checkpoint_path = os.path.join(LOGS_PATH, f"checkpoint/{name}")

    return model_dir, tensorboard_dir, checkpoint_path, plot_file


def train(train, test, val, experiment=None, text="", labels={}, model_export=None, **kwargs):
    model_dir, tensorboard_dir, checkpoint_path, plot_file = define_path("one_to_many")
    model_dir = model_export if model_export else model_dir

    for k, v in labels.items():
        multi_label = True if v == "multi" else False
        td = TensorflowDataset(text=text, label=k, multi_label=multi_label)
        test = td.clean_dataset(test)
        val = td.clean_dataset(val)
        train = td.clean_dataset(train)

    input_data = {"train": train[text].values, "val": val[text].values, "test": test[text].values}
    label_data = {}
    for k, v in labels.items():
        multi_label = True if v == "multi" else False
        le = LabelEncoding(multi_label)
        le.fit_lb(train[k].values)
        label_data[k] = {
            'train': le.transform_lb(train[k].values),
            'val': le.transform_lb(val[k].values),
            'test': le.transform_lb(test[k].values),
            'encoder': le.lb,
            "classification": v
        }

    self = TensorflowClassifier(labels, **kwargs)
    self.set_model(label_data)
    self.compile_model()
    self.plot_model(plot_file)
    train_dataset = (
        {"input_text": input_data["train"]},
        {k: label_data[k]['train'] for k, v in labels.items()}
    )
    val_dataset = (
        {"input_text": input_data["val"]},
        {k: label_data[k]['val'] for k, v in labels.items()}
    )

    if experiment:
        callbacks = self.callback_func(checkpoint_path=checkpoint_path)
        open_mark = mark_format()
        close_mark = mark_format(close=True)
        experiment.log_html("".join([open_mark, "train size", close_mark, str(len(train)), "<br><br>"]))
        experiment.log_html("".join([open_mark, "test size", close_mark, str(len(test)), "<br><br>"]))
        experiment.log_html("".join([open_mark, "val size", close_mark, str(len(val)), "<br><br>"]))
        experiment.log_parameters(kwargs)
        experiment.log_dataset_hash(train)
        experiment.add_tags(['multi_lang', 'tensorflow', "1_input_2_outputs"])
        experiment.add_tags([x for x in labels.keys()])
        experiment.log_image(plot_file)
        experiment.set_model_graph(self.model)
        with experiment.train():
            _ = self.train(train_dataset, val_dataset, epochs=self.epochs, callbacks=callbacks)
    else:
        callbacks = self.callback_func(tensorboard_dir=tensorboard_dir, checkpoint_path=checkpoint_path)
        start_color = text_format(txt_color='purple', txt_style='bold')
        end_format = text_format(end=True)
        print(start_color, "train size", end_format, train.shape)
        print(start_color, "test size", end_format, test.shape)
        print(start_color, "val size", end_format, val.shape)
        print(kwargs)
        history = self.train(train_dataset, val_dataset, epochs=self.epochs, callbacks=callbacks)
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        macro_f1 = history.history["macro_f1"]
        val_macro_f1 = history.history["val_macro_f1"]
        metrics = {"loss": loss, "val_loss": val_loss, "macro_f1": macro_f1, "val_macro_f1": val_macro_f1}
        print(metrics)

    self.export_model(model_dir)
    for k in label_data.keys():
        le = label_data[k]['encoder']
        filename = os.path.join(model_dir, f"{k}.pkl")
        pickle.dump(le, open(filename, "wb"))

    if experiment:
        experiment.log_asset(model_dir)
        experiment.send_notification('Finished')
        experiment.end()


if __name__ == '__main__':
    expe = Experiment(api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE)
    test_df = pd.read_csv(os.path.join(DATASET_PATH, "one_to_many", 'test.csv'))
    val_df = pd.read_csv(os.path.join(DATASET_PATH, "one_to_many", 'val.csv'))
    train_df = pd.read_csv(os.path.join(DATASET_PATH, "one_to_many", 'train.csv'))
    labels = {"intent": "multi", "sentiment": "multi"}
    train(
        train_df, val_df, test_df, experiment=expe, text='feature',
        labels=labels, model_export=None, batch_size=1024, epochs=30)
