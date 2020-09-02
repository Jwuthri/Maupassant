import ast
import tqdm

import tensorflow as tf

from sklearn.model_selection import train_test_split

from maupassant.utils import timer
from maupassant.settings import MODEL_PATH
from maupassant.dataset.labels import LabelEncoding
from maupassant.preprocessing.normalization import TextNormalization


class BuildDataset(LabelEncoding):

    def __init__(self, label_type="single_label", batch_size=512, buffer_size=512):
        multi_label = True if label_type == "multi-label" else False
        super().__init__(multi_label, base_path=MODEL_PATH, model_load=False)
        self.label_type = label_type
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def clean_labels(self, data, y):
        pbar = tqdm.tqdm(total=len(data), desc="Cleaning the labels")
        cleaned_labels = []
        for label in data[y].values:
            try:
                label = ast.literal_eval(label)
            except ValueError:
                label = [label]
            cleaned_labels.append(label)
            pbar.update(1)
        data['cleaned_labels'] = cleaned_labels

        return data

    def clean_texts(self, data, x):
        pbar = tqdm.tqdm(total=len(data), desc="Cleaning the texts")
        cleaned_texts = []
        tn = TextNormalization()
        for text in data[x].values:
            text = tn.replace_char_rep(text=text)
            text = tn.replace_words_rep(text=text)
            text = tn.remove_multiple_spaces(text=text)
            text = text.strip()
            cleaned_texts.append(text)
            pbar.update(1)
        pbar.close()
        data["cleaned_texts"] = cleaned_texts

        return data

    def to_tensorflow_dataset(self, x, y, dataset_name="train", is_training=True):
        print(f"creating the dataset: {dataset_name}")
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    @timer
    def generate(self, data, x, y, test_size=0.1):
        cleaned = self.clean_texts(data, x)
        if self.multi_label:
            cleaned = self.clean_labels(cleaned, y)
        else:
            cleaned['cleaned_labels'] = cleaned[y]
        X, y = cleaned["cleaned_texts"].values, cleaned["cleaned_labels"].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        self.fit_encoder(y)
        y_train_encoded = self.transform_encoder(y_train)
        y_val_encoded = self.transform_encoder(y_val)
        train_dataset = self.to_tensorflow_dataset(X_train, y_train_encoded, dataset_name="train")
        val_dataset = self.to_tensorflow_dataset(X_val, y_val_encoded, is_training=False, dataset_name="validation")

        return train_dataset, val_dataset
