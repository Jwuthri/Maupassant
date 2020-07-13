import re
import tqdm

from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

from maupassant.utils import timer
from maupassant.preprocessing.normalization import TextNormalization


class DatasetGenerator(object):

    def __init__(self, input_shape=64, max_labels=5000, batch_size=512, buffer_size=512):
        self.input_shape = input_shape
        self.max_labels = max_labels
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(filters='', num_words=20000, oov_token='[UNK]')
        self.vocab_size = 0
        self.splitter = "|".join([
            "!", "@", "#", "$", "%", "^",  "&",  "\\(", "\\)", "_",  "-",  ",", "<", "\\.", ">", "\\?", "`", "~", ":",
            ";","\\+", "=", "[", "]", "{", "}", "\n{2,}", "\\s", "\n"
        ])

    def split_text(self, text):
        text = text.lower()

        return re.sub(r'(' + self.splitter + ')', r' \1 ', text)

    def set_tokenizer(self, text):
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def clean_dataset(self, data):
        pbar = tqdm.tqdm(total=len(data))
        cleaned_texts = []
        tn = TextNormalization()
        for text in data.values:
            text = self.split_text(text)
            text = tn.replace_char_rep(text=text)
            text = tn.replace_words_rep(text=text)
            text = tn.text_demojis(text=text)
            text = tn.text_demoticons(text=text)
            text = tn.remove_multiple_spaces(text=text)
            text = text.strip()
            cleaned_texts.append(text)
            pbar.update(1)
        pbar.close()

        return cleaned_texts

    @staticmethod
    def concat_text(text):
        return " [END] ".join(text)

    def predictable_words(self):
        value = min(self.tokenizer.num_words, self.max_labels)
        words = list(range(1, value))

        return words

    def texts_to_sequences(self, cleaned_data, labels):
        texts_to_sequences = list()
        pbar = tqdm.tqdm(total=len(cleaned_data))
        for text in cleaned_data:
            encoded_text = self.tokenizer.texts_to_sequences([text])[0]
            for idx in range(1, len(encoded_text) - 1):
                if encoded_text[idx:idx+1][0] in labels:
                    texts_to_sequences.append(encoded_text[:idx+1])
            pbar.update(1)
        pbar.close()

        return texts_to_sequences

    def create_dataset(self, data, labels):
        texts_to_sequences = self.texts_to_sequences(data, labels)
        padded_sequences = np.array(pad_sequences(texts_to_sequences, maxlen=self.input_shape + 1, padding='pre'))
        x, y = padded_sequences[:, :-1], padded_sequences[:, -1:]
        dataset = self.to_tensorflow_dataset(x, y)

        return dataset

    @timer
    def to_tensorflow_dataset(self, x, y, is_training=True):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    @timer
    def generate(self, data):
        cleaned_data = self.clean_dataset(data)
        text = self.concat_text(cleaned_data)
        self.set_tokenizer(text)
        labels = self.predictable_words()
        _, test = train_test_split(cleaned_data, test_size=0.05, random_state=42)
        test, val = train_test_split(test, test_size=0.5, random_state=42)
        train_dataset = self.create_dataset(cleaned_data, labels)
        test_dataset = self.create_dataset(test, labels)
        val_dataset = self.create_dataset(val, labels)

        return train_dataset, test_dataset, val_dataset
