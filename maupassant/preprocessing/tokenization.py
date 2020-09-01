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


def text_tokenizer(text, filters='', num_words= 10000, lower=True, char_level=False, oov_token="[UNK]"):
    tokenizer = Tokenizer(filters=filters, num_words=num_words, lower=lower, char_level=char_level, oov_token=oov_token)
    tokenizer.fit_on_texts([text])

    return tokenizer
