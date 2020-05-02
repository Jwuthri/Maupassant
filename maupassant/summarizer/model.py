import os
import math
import pickle

import numpy as np
import pandas as pd

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from maupassant.preprocessing.tokenization import SequenceTokenization, SentenceTokenization
from maupassant.preprocessing.normalization import TextNormalization
from maupassant.settings import EXTERNAL_PATH
from maupassant.utils import timer


class GoogleT5(object):

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.device = torch.device('cpu')

    def summarize(self, tokenized_text):
        return self.model.generate(
            tokenized_text, num_beams=4, no_repeat_ngram_size=2,
            min_length=30, max_length=100, early_stopping=True)

    @timer
    def predict(self, text):
        processed_text = "summarize: " + text.strip()
        tokenized_text = self.tokenizer.encode(processed_text, return_tensors="pt").to(self.device)
        relevant_indexes = self.summarize(tokenized_text)

        return self.tokenizer.decode(relevant_indexes[0], skip_special_tokens=True)


class WeightedTfIdf(object):

    def __init__(self, keywords):
        self.stemmer = TextNormalization()
        self.stopwords = pickle.load(open(os.path.join(EXTERNAL_PATH, 'stopwords.p'), "rb"))
        self.keywords = keywords
        self.augmented_keywords = self.keywords_augmentation()

    def keywords_augmentation(self):
        return self.keywords

    def create_dictionary(self, text):
        tokens = SentenceTokenization().tokenize(text)
        unique_words = frozenset(self.stemmer.word_stemming(w) for w in tokens if w not in self.stopwords)

        return dict((w, i) for i, w in enumerate(unique_words))

    def create_matrix(self, dictionary, sentences):
        words_count = len(dictionary)
        sentences_count = len(sentences)
        matrix = np.zeros((words_count, sentences_count))
        sentence_to_token = {sentence: SentenceTokenization().tokenize(sentence) for sentence in sentences}
        for col, sentence in enumerate(sentences):
            for word in map(self.stemmer.word_stemming, sentence_to_token[sentence]):
                if word in dictionary:
                    row = dictionary[word]
                    if word in self.keywords:
                        matrix[row, col] += 3
                    elif word in self.augmented_keywords:
                        matrix[row, col] += 2.5
                    else:
                        matrix[row, col] += 1

        return matrix

    def compute_term_frequency(self, matrix, dictionary):
        df = pd.DataFrame(matrix)
        df["word"] = dictionary
        ddf = df[~df["word"].isin(self.keywords)]
        ddf = ddf.drop(columns=["word"])
        ddf = ddf.as_matrix()

        max_word_frequencies = np.max(ddf)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies
                if max_word_frequency != 0:
                    frequency = matrix[row, col] / max_word_frequency
                    matrix[row, col] = frequency

        return matrix

    @staticmethod
    def compute_ranks(sigma, v_matrix):
        dimensions = max(3, int(len(sigma)))
        powered_sigma = tuple(s ** 2 if i < dimensions else 0.0 for i, s in enumerate(sigma))

        ranks = []
        for column_vector in v_matrix.T:
            rank = sum(s * v ** 2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))

        return ranks

    @staticmethod
    def get_best_sentences(sentences, ranks):
        vals = dict(zip(sentences, ranks))
        res = {str(key): vals[key] for key in sorted(vals, key=vals.get, reverse=True)}

        return res

    @timer
    def predict(self, text):
        sequence_to_sentences = SequenceTokenization().tokenize(text)
        dictionary = self.create_dictionary(text)
        tf = self.create_matrix(dictionary, sequence_to_sentences)
        idf = self.compute_term_frequency(tf, dictionary)
        u, sigma, v = np.linalg.svd(idf, full_matrices=False)
        ranks = self.compute_ranks(sigma, v)
        res = self.get_best_sentences(sequence_to_sentences, ranks)

        return res
