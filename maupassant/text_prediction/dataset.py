import re
import ast
import tqdm
import emot

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

from maupassant.utils import timer


class DatasetGenerator(object):

    def __init__(self, input_shape=128, words_to_predict=1, max_labels=1500, batch_size=128, buffer_size=128):
        self.input_shape = input_shape
        self.words_to_predict = words_to_predict
        self.max_labels = max_labels
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.le = preprocessing.LabelEncoder()
        self.tokenizer = Tokenizer(filters='')
        self.vocab_size = 0
        self.to_tokenize = '!|"|\\#|\\$|%|\\&|\\(|\\)|\\*|\\+|,|\\-|\\.|/|:|;|<|=|>|\\?|@|\\[|\\\\|\\]|\\^|_|`|\\{|\\||\\}|\\~|\n{2,}|\\s|\n'

    def tokenize_text(self, text):
        text = text.lower()

        return re.sub(r'(' + self.to_tokenize + ')', r' \1 ', text)

    def concat_text(self, text):
        return " ".join(text)

    def concat_and_tokenize(self, text):
        concat_text = self.concat_text(text)

        return self.tokenize_text(concat_text)

    def remove_tokens(self, tokens):
        for token in tokens:
            if token in self.tokenizer.word_index:
                del self.tokenizer.word_index[token]

    def clean_tokenizer(self):
        to_remove = [
            '(', ')', '-', '/', '@', '#', '%', '^', '&', '*', '_', '{', '}', '|', '>',
            '<', ':', ';', '"', '$', '\x06', ' ', '\x08']
        self.remove_tokens(to_remove)
        self.remove_tokens([str(i) for i in range(100000)])
        self.remove_tokens([x for x in emot.EMOTICONS.keys()])
        self.remove_tokens([x for x in emot.UNICODE_EMO.keys()])

    def set_tokenizer(self, text):
        self.tokenizer.fit_on_texts([text])
        # self.clean_tokenizer()
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def create_sequences(self, dataset, commons_ngrams=None):
        sequences = list()
        pbar = tqdm.tqdm(total=len(dataset))
        for row in dataset:
            tokenized_text = self.tokenize_text(row)
            encoded_text = self.tokenizer.texts_to_sequences([tokenized_text])[0]
            for idx in range(1, len(encoded_text) - self.words_to_predict):
                sequence = encoded_text[: idx + self.words_to_predict]
                if not commons_ngrams:
                    sequences.append(sequence)
                else:
                    if encoded_text[idx: idx + self.words_to_predict] in commons_ngrams:
                        sequences.append(sequence)
            pbar.update(1)
        pbar.close()

        return sequences

    def sequences_to_padded_sequences(self, sequences):
        padded_sequences = pad_sequences(sequences, maxlen=self.input_shape + self.words_to_predict, padding="pre")

        return np.array(padded_sequences)

    def get_commons_entities(self, padded_sequences):
        x, y = padded_sequences[:, :-self.words_to_predict], padded_sequences[:, -self.words_to_predict:]
        most_commons = [str(list(i)) for i in y]
        counter = Counter(most_commons)
        commons_ngrams = counter.most_common(n=self.max_labels)
        self.max_labels = len(commons_ngrams)

        return [ast.literal_eval(common[0]) for common in commons_ngrams]

    def fit_label_encoder(self, data, commons_ngrams):
        filtered_sequences = self.create_sequences(data, commons_ngrams=commons_ngrams)
        filtered_padded_sequences = self.sequences_to_padded_sequences(filtered_sequences)
        y = filtered_padded_sequences[:, -self.words_to_predict:]
        y_encoded = [str(list(i)) for i in y]
        self.le.fit(y_encoded)

    def get_x_y(self, filtered_padded_sequences, max_size=-1):
        filtered_padded_sequences = filtered_padded_sequences[:max_size]
        x = filtered_padded_sequences[:, :-self.words_to_predict]
        y = filtered_padded_sequences[:, -self.words_to_predict:]
        y_encoded = [str(list(i)) for i in y]
        y_encoded = self.le.transform(y_encoded)

        return x, y_encoded

    @timer
    def to_tensorflow_dataset(self, data, commons_ngrams, is_training=True):
        filtered_sequences = self.create_sequences(data, commons_ngrams=commons_ngrams)
        filtered_padded_sequences = self.sequences_to_padded_sequences(filtered_sequences)
        x, y = self.get_x_y(filtered_padded_sequences)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    @timer
    def to_dataset(self, data, commons_ngrams):
        filtered_sequences = self.create_sequences(data, commons_ngrams=commons_ngrams)
        filtered_padded_sequences = self.sequences_to_padded_sequences(filtered_sequences)
        x, y = self.get_x_y(filtered_padded_sequences)

        return x, y

    @timer
    def generate(self, data):
        text = self.concat_and_tokenize(data)
        self.set_tokenizer(text)
        sequences = self.create_sequences(data)
        padded_sequences = self.sequences_to_padded_sequences(sequences)
        commons_ngrams = self.get_commons_entities(padded_sequences)
        del sequences, padded_sequences
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        test, val = train_test_split(test, test_size=0.5, random_state=42)
        self.fit_label_encoder(data, commons_ngrams)
        train_dataset = self.to_dataset(train, commons_ngrams)
        val_dataset = self.to_dataset(val, commons_ngrams)
        test_dataset = self.to_dataset(test, commons_ngrams)

        return train_dataset, val_dataset, test_dataset
