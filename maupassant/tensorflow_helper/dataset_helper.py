import ast

from tqdm import tqdm

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from maupassant.utils import timer
from maupassant.preprocessing.normalization import TextNormalization
from maupassant.tensorflow_helper.model_helper import TensorflowModel


class TensorflowDataset(TensorflowModel):
    """Module to generate dataset."""

    def __init__(self, architecture, label_type, name, **kwargs):
        self.test_size = kwargs.get('test_size', 0.1)
        self.buffer_size = kwargs.get('buffer_size', 512)
        self.batch_size = kwargs.get('batch_size', 512)
        self.label_encoder = self.init_label_encoder(label_type)
        self.label_encoder_classes = dict()
        self.label_encoder_classes_number = 0
        super().__init__(architecture, label_type, name, **kwargs)

    @staticmethod
    def init_label_encoder(label_type):
        if label_type == "multi-label":
            return MultiLabelBinarizer()
        else:
            return LabelEncoder()

    def fit_encoder(self, y):
        self.label_encoder.fit(y)
        self.label_encoder_classes = dict(enumerate(self.label_encoder.classes_))
        self.label_encoder_classes_number = len(self.label_encoder.classes_)

    @staticmethod
    def clean_x(x):
        pbar = tqdm(total=len(x), desc="Cleaning x")
        cleaned_texts = []
        tn = TextNormalization()
        for text in x:
            text = tn.replace_char_rep(text=text)
            text = tn.replace_words_rep(text=text)
            text = tn.remove_multiple_spaces(text=text)
            text = text.strip()
            cleaned_texts.append(text)
            pbar.update(1)
        pbar.close()

        return cleaned_texts

    @staticmethod
    def clean_y(y):
        pbar = tqdm(total=len(y), desc="Cleaning y")
        cleaned_labels = []
        for label in y:
            try:
                label = ast.literal_eval(label)
            except ValueError:
                label = [label]
            cleaned_labels.append(label)
            pbar.update(1)
        pbar.close()

        return cleaned_labels

    def to_tensorflow_dataset(self, x, y, dataset_name="train", is_training=True):
        pbar = tqdm(totall=1, desc=f"Generating dataset {dataset_name}")
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        pbar.update(1)
        pbar.close()

        return dataset

    @timer
    def generate_dataset(self, data, x_column, y_column):
        data = data[data[x_column].notnull()]
        data = data[data[y_column].notnull()]
        x = self.clean_x(data[x_column])
        y = self.clean_y(data[y_column]) if self.label_type == "multi-label" else data[y_column]
        self.fit_encoder(y)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.test_size, random_state=42)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        train = self.to_tensorflow_dataset(x_train, y_train_encoded, dataset_name="train")
        val = self.to_tensorflow_dataset(x_val, y_val_encoded, dataset_name="val", is_training=False)

        return train, val
