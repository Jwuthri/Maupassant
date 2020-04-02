import datetime

import pandas as pd
from comet_ml import Experiment

from maupassant.settings import *
from maupassant.utils import mark_format, text_format
from maupassant.settings import MODEL_PATH, DATASET_PATH, LOGS_PATH
from maupassant.suppervised.one_to_one_classifier import TensorflowClassifier


def define_path(classifier, label):
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = f"one_to_one_{date}"
    model_dir = os.path.join(MODEL_PATH, name, 'model')
    le_path = os.path.join(MODEL_PATH, f"{name}/0_{classifier}_{label}_encoder.pkl")
    tensorboard_dir = os.path.join(LOGS_PATH, f"tensorboard/{name}")
    checkpoint_path = os.path.join(LOGS_PATH, f"checkpoint/{name}")
    plot_file = os.path.join(LOGS_PATH, f"plot/{name}.jpg")

    return model_dir, le_path, tensorboard_dir, checkpoint_path, plot_file


def train(train, test, val, classifier="binary", experiment=None, text="text", label='sentiment', fine_tune=False, update_model_path=None, **kwargs):
    assert classifier in ['binary', 'single', 'multi']
    model_dir, le_path, tensorboard_dir, checkpoint_path, plot_file = define_path(classifier, label)
    self = TensorflowClassifier(clf_type=classifier, text=text, label=label, **kwargs)
    cleaned_test = self.clean_dataset(test)
    cleaned_val = self.clean_dataset(val)
    cleaned_train = self.clean_dataset(train)

    x_test, y_test = self.split_x_y(cleaned_test)
    x_val, y_val = self.split_x_y(cleaned_val)
    x_train, y_train = self.split_x_y(cleaned_train)

    self.fit_lb(y_train)
    y_val_encoded = self.transform_lb(y_val)
    y_train_encoded = self.transform_lb(y_train)

    val_dataset = self.to_tensorflow_dataset(x_val, y_val_encoded)
    train_dataset = self.to_tensorflow_dataset(x_train, y_train_encoded)

    self.set_model()
    self.compile_model()
    self.plot_model(plot_file)
    if fine_tune:
        if update_model_path:
            import tensorflow as tf
            latest = tf.train.latest_checkpoint(update_model_path)
            self.model.load_weights(latest)
        else:
            raise Exception('Please provide the path of the model to fine_tune using the args => "update_model_path"')

    if experiment:
        callbacks = self.callback_func(checkpoint_path=checkpoint_path)
        open_mark = mark_format()
        close_mark = mark_format(close=True)
        experiment.log_html("".join([open_mark, "train size", close_mark, str(len(cleaned_train)), "<br><br>"]))
        experiment.log_html("".join([open_mark, "test size", close_mark, str(len(cleaned_test)), "<br><br>"]))
        experiment.log_html("".join([open_mark, "val size", close_mark, str(len(cleaned_val)), "<br><br>"]))
        experiment.log_parameters(kwargs)
        experiment.log_dataset_hash(train)
        experiment.add_tags(['multi_lang', 'tensorflow', "1_input_1_output", self.label])
        experiment.log_parameters(dict(enumerate(self.lb.classes_)))
        experiment.log_image(plot_file)
        with experiment.train():
            _ = self.train(train_dataset, val_dataset, epochs=self.epochs, callbacks=callbacks)

    else:
        callbacks = self.callback_func(tensorboard_dir=tensorboard_dir, checkpoint_path=checkpoint_path)
        start_color = text_format(txt_color='purple', txt_style='bold')
        end_format = text_format(end=True)
        print(start_color, "train size", end_format, cleaned_train.shape)
        print(start_color, "test size", end_format, cleaned_test.shape)
        print(start_color, "val size", end_format, cleaned_val.shape)
        print(kwargs)
        self.show_classes()
        history = self.train(train_dataset, val_dataset, epochs=self.epochs, callbacks=callbacks)
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        macro_f1 = history.history["macro_f1"]
        val_macro_f1 = history.history["val_macro_f1"]
        metrics = {"loss": loss, "val_loss": val_loss, "macro_f1": macro_f1, "val_macro_f1": val_macro_f1}
        print(metrics)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        metrics = {'test_loss': loss, 'test_accuracy': accuracy}
        print(metrics)

    self.export_model(model_dir)
    self.save_lb(le_path)
    if experiment:
        experiment.log_asset(model_dir)
        experiment.send_notification('Finished')
        experiment.end()


if __name__ == '__main__':
    expe = Experiment(api_key=API_KEY, project_name=PROJECT_NAME, workspace=WORKSPACE)
    test_df = pd.read_csv(os.path.join(DATASET_PATH, "one_to_one", 'test_sentiment.csv'))
    val_df = pd.read_csv(os.path.join(DATASET_PATH, "one_to_one", 'val_sentiment.csv'))
    train_df = pd.read_csv(os.path.join(DATASET_PATH, "one_to_one", 'train_sentiment.csv'))
    train(
        train_df, val_df, test_df, experiment=expe, text='feature', label='sentiment',
        batch_size=128, buffer_size=128, epochs=20, classifier='multi', fine_tune=False, update_model_path=None
    )
