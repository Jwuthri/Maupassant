import ast

import tensorflow as tf

from maupassant.utils import timer
from maupassant.dataset.labels import LabelEncoding
from maupassant.dataset.pandas import remove_rows_contains_null


class TensorflowDataset(LabelEncoding):

    def __init__(self, text, label, multi_label=True, batch_size=512, buffer_size=512):
        super().__init__(multi_label)
        self.text = text
        self.label = label
        self.multi_label = multi_label
        self.buffer_size = buffer_size
        self.batch_size = batch_size

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

    @timer
    def clean_dataset(self, df):
        df = remove_rows_contains_null(df, self.text)
        df = remove_rows_contains_null(df, self.label)
        if self.multi_label:
            df[self.label] = self.clean_labels(df[self.label].values)

        return df

    @timer
    def to_tensorflow_dataset(self, x, y, is_training=True):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def split_x_y(self, df):
        return df[self.text].values, df[self.label].values

    def tf_dataset(self, train_df, test_df, val_df):
        test_df = self.clean_dataset(test_df)
        val_df = self.clean_dataset(val_df)
        train_df = self.clean_dataset(train_df)

        x_test, y_test = self.split_x_y(test_df)
        x_val, y_val = self.split_x_y(val_df)
        x_train, y_train = self.split_x_y(train_df)

        self.fit_lb(y_train)
        y_val_encoded = self.transform_lb(y_val)
        y_test_encoded = self.transform_lb(y_test)
        y_train_encoded = self.transform_lb(y_train)

        val_dataset = self.to_tensorflow_dataset(x_val, y_val_encoded)
        train_dataset = self.to_tensorflow_dataset(x_train, y_train_encoded)
        test_dataset = self.to_tensorflow_dataset(x_test, y_test_encoded)

        return train_dataset, test_dataset, val_dataset
