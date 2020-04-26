import os
import pickle

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from maupassant.preprocessing.normalization import TextNormalization
from maupassant.preprocessing.tokenization import SequenceTokenization, SentenceTokenization
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

    def __init__(self):
        self.stemmer = TextNormalization()
        self.stopwords = pickle.load(open(os.path.join(EXTERNAL_PATH, 'stopwords.p'), "rb"))

    def create_dictionary(self, text):
        tokens = SentenceTokenization().tokenize(text)
        unique_words = frozenset(self.stemmer.word_stemming(w) for w in tokens if w not in self.stopwords)

        return dict((w, i) for i, w in enumerate(unique_words))

    @timer
    def predict(self, text):
        sentences = SequenceTokenization().tokenize(text)
