from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class SentenceTokenization:

    def __int__(self):
        pass

    @staticmethod
    def tokenize(sentence):
        return word_tokenize(sentence)

    @staticmethod
    def detokenize(words):
        return TreebankWordDetokenizer().detokenize(words)


class SequenceTokenization:

    def __int__(self):
        pass

    @staticmethod
    def tokenize(sequence):
        return sent_tokenize(sequence)

    @staticmethod
    def detokenize(sentences):
        return TreebankWordDetokenizer().detokenize(sentences)
