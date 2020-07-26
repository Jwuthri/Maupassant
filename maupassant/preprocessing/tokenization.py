from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from tensorflow.keras.preprocessing.text import Tokenizer


class SentenceTokenization(object):

    def __int__(self):
        pass

    @staticmethod
    def tokenize(sentence):
        return word_tokenize(sentence)

    @staticmethod
    def detokenize(words):
        return TreebankWordDetokenizer().detokenize(words)


class SequenceTokenization(object):

    def __int__(self):
        pass

    @staticmethod
    def tokenize(sequence):
        return sent_tokenize(sequence)

    @staticmethod
    def detokenize(sentences):
        return TreebankWordDetokenizer().detokenize(sentences)


def get_text_tokenizer(text):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts([text])

    return tokenizer
