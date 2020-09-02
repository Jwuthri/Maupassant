import pytest

from maupassant.preprocessing.tokenization import SentenceTokenization, SequenceTokenization
from maupassant.preprocessing.normalization import TextNormalization


class TestTokenization(object):

    def test_sentence_tokenization_tokenize(self):
        sentence = "Let me tell you something you already know."
        tokens = SentenceTokenization().tokenize(sentence)
        assert tokens == ['Let', 'me', 'tell', 'you', 'something', 'you', 'already', 'know', '.']

    def test_sentence_tokenization_detokenize(self):
        tokens = ['Let', 'me', 'tell', 'you', 'something', 'you', 'already', 'know', '.']
        sentence = SentenceTokenization().detokenize(tokens)
        assert sentence == "Let me tell you something you already know."

    def test_sequence_tokenization_tokenize(self):
        sequence = "Let me tell you something you already know. The world ain’t all sunshine and rainbows."
        sentences = SequenceTokenization().tokenize(sequence)
        assert sentences == [
            'Let me tell you something you already know.',
            'The world ain’t all sunshine and rainbows.'
        ]

    def test_sequence_tokenization_detokenize(self):
        sentences = [
            'Let me tell you something you already know.',
            'The world ain’t all sunshine and rainbows.'
        ]
        sequence = SequenceTokenization().detokenize(sentences)
        assert sequence == "Let me tell you something you already know. The world ain’t all sunshine and rainbows."


class TestNormalization(object):

    def test_text_correction(self):
        text = 'Let me tell you somthing you alrady know.'
        cleaned_text = TextNormalization().text_correction(text)
        assert cleaned_text == 'Let me tell you something you already know.'

    def test_word_stemming(self):
        word = "shipping"
        stemmed_word = TextNormalization().word_stemming(word)
        assert stemmed_word == 'ship'

    def test_text_demojize(self):
        text = 'Let me tell you something you already know 👍'
        demojize_text = TextNormalization().text_demojis(text, how_replace='mean')
        assert demojize_text == 'Let me tell you something you already know thumbs_up'
        demojize_text = TextNormalization().text_demojis(text, how_replace="")
        assert demojize_text == 'Let me tell you something you already know '

    def test_text_demoticons(self):
        text = 'Let me tell you something you already know :)'
        demoticons_text = TextNormalization().text_demoticons(text, how_replace='mean')
        assert demoticons_text == 'Let me tell you something you already know Happy face or smiley'
        demoticons_text = TextNormalization().text_demoticons(text, how_replace="")
        assert demoticons_text == 'Let me tell you something you already know '

    # @pytest.mark.skip(reason="not implemented yet")
    def test_text_decontraction(self):
        text = "I'd like to know yall guys"
        decontraction_text = TextNormalization().text_decontraction(text)
        assert decontraction_text == "I would like to know you all guys"
