import ast
import tqdm

import tensorflow as tf
from sklearn.model_selection import train_test_split

from maupassant.dataset.label_encoder import LabelEncoding
from maupassant.dataset.pandas import remove_rows_contains_null
from maupassant.preprocessing.normalization import TextNormalization


class TensorflowDataset(LabelEncoding):

    def __init__(self, feature, label, multi_label, batch_size, buffer_size, cleaning_func=None, keep_emot=False):
        super().__init__(multi_label)
        self.normalizer = TextNormalization()
        self.feature = feature
        self.label = label
        self.multi_label = multi_label
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cleaning_func = cleaning_func
        self.keep_emot = keep_emot

    @staticmethod
    def clean_labels(labels):
        label_cleaned = []
        for label in labels:
            try:
                label = ast.literal_eval(label)
            except ValueError:
                label = [label]
            label_cleaned.append(label)

        return label_cleaned

    def clean_dataset(self, df):
        df = remove_rows_contains_null(df, self.feature)
        df = remove_rows_contains_null(df, self.label)
        pbar = tqdm.tqdm(total=len(df), desc="Cleaning the dataset")
        cleaned_texts = []
        for text in df[self.feature].values:
            text = self.normalizer.split_text_for_tokenizer(text)
            text = self.normalizer.replace_char_rep(text=text)
            text = self.normalizer.replace_words_rep(text=text)
            if self.keep_emot:
                text = self.normalizer.text_demojis(text=text)
                text = self.normalizer.text_demoticons(text=text)
            if self.cleaning_func:
                text = self.cleaning_func(text)
            text = self.normalizer.remove_multiple_spaces(text=text)
            text = text.strip()
            cleaned_texts.append(text)
            pbar.update(1)
        pbar.close()
        df[self.feature] = cleaned_texts
        if self.multi_label:
            df[self.label] = self.clean_labels(df[self.label].values)

        return df

    def to_tensorflow_dataset(self, x, y, is_training=True):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            # dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def split_x_y(self, df):
        return df[self.feature].values, df[self.label].values

    def generate(self, dataset):
        dataset = self.clean_dataset(dataset)
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
        test, val = train_test_split(test, test_size=0.5, random_state=42)

        x_train, y_train = self.split_x_y(train)
        x_test, y_test = self.split_x_y(test)
        x_val, y_val = self.split_x_y(val)

        self.fit_encoder(y_train)
        y_train_encoded = self.transform_encoder(y_train)
        y_test_encoded = self.transform_encoder(y_test)
        y_val_encoded = self.transform_encoder(y_val)

        train_dataset = self.to_tensorflow_dataset(x_train, y_train_encoded)
        test_dataset = self.to_tensorflow_dataset(x_test, y_test_encoded)
        val_dataset = self.to_tensorflow_dataset(x_val, y_val_encoded)

        return train_dataset, test_dataset, val_dataset
