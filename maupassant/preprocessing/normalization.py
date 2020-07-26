import re
import emot
import contractions

from autocorrect import Speller

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from maupassant.preprocessing.tokenization import SentenceTokenization


class TextNormalization(object):

    def __init__(self, stemmer='PorterStemmer', language='en'):
        self.stemmer = stemmer
        self.lemstem = self.get_lemstem
        self.checker = Speller(lang=language)

    @property
    def get_lemstem(self):
        if self.stemmer == "WordNetLemmatizer":
            return WordNetLemmatizer()
        elif self.stemmer == "LancasterStemmer":
            return LancasterStemmer("english")
        elif self.stemmer == "SnowballStemmer":
            return SnowballStemmer("english", ignore_stopwords=True)
        else:
            return PorterStemmer()

    def text_correction(self, text):
        return self.checker(text)

    def word_stemming(self, word):
        if self.stemmer == 'WordNetLemmatizer':
            return self.lemstem.lemmatize(word)
        else:
            return self.lemstem.stem(word)

    def text_stemming(self, text):
        words = SentenceTokenization().tokenize(text)
        stemmed = [self.word_stemming(word) for word in words]

        return SentenceTokenization().detokenize(stemmed)

    @staticmethod
    def text_demojis(text, how_replace=""):
        emojis = emot.emoji(text)
        if isinstance(emojis, list):
            emojis = emojis[0]
        if emojis['flag']:
            for index in range(len(emojis["value"])):
                if how_replace == 'mean':
                    source = emojis['value'][index]
                    target = emojis['mean'][index].split(':')[1]
                    text = text.replace(source, target)
                else:
                    text = text.replace(emojis['value'][index], "")

        return text

    @staticmethod
    def text_demoticons(text, how_replace=""):
        emoticons = emot.emoticons(text)
        if isinstance(emoticons, list):
            emoticons = emoticons[0]
        if emoticons['flag']:
            for index in range(len(emoticons["value"])):
                if how_replace == 'mean':
                    source = emoticons['value'][index]
                    target = emoticons['mean'][index]
                    text = text.replace(source, target)
                else:
                    text = text.replace(emoticons['value'][index], "")

        return text

    @staticmethod
    def text_decontraction(text):
        return contractions.fix(text)

    @staticmethod
    def spec_add_spaces(text):
        return re.sub(r'([/#\n])', r' \1 ', text)

    @staticmethod
    def remove_multiple_spaces(text):
        return re.sub(' {2,}', ' ', text)

    @staticmethod
    def _replace_group(m):
        c, cc = m.groups()
        return c

    def replace_char_rep(self, text):
        char_rep = re.compile(r'(\S)(\1{2,})')

        return char_rep.sub(self._replace_group, text)

    def replace_words_rep(self, text):
        word_rep = re.compile(r'(\b\w+\W+)(\1{2,})')

        return word_rep.sub(self._replace_group, text)
