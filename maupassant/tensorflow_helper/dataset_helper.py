import ast

from tqdm import tqdm

import tensorflow as tf

from maupassant.preprocessing.normalization import TextNormalization


class TensorflowDataset(object):

    def __init__(self, buffer_size=512, batch_size=512):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

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

        return cleaned_labels

    def to_tensorflow_dataset(self, x, y, dataset_name="train", is_training=True):
        print(f"creating the dataset: {dataset_name}")
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def generate(self, data, x_column, y_column, test_size=0.1):
        pass
