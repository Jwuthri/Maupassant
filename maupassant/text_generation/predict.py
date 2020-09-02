import re
import string
import operator

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import numpy as np

from maupassant.settings import MODEL_PATH
from maupassant.preprocessing.normalization import TextNormalization
from maupassant.tensorflow_models_compile import BaseTensorflowModel, ModelSaverLoader

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()


class Predicter(BaseTensorflowModel):

    def __init__(self, base_path=MODEL_PATH, name='model_name', max_words=20000, cleaning_func=None):
        self.normalizer = TextNormalization()
        self.cleaning_func = cleaning_func
        msl = ModelSaverLoader(base_path, name, True)
        info = msl.load_info()
        label_type = info.get('label_type')
        architecture = info.get('architecture')
        number_labels = info.get('number_labels')
        pretrained_embedding = info.get('pretrained_embedding')
        input_size = info.get('input_size', 0)
        vocab_size = info.get('vocab_size', 0)
        embedding_size = info.get('embedding_size')
        super().__init__(label_type, architecture, number_labels, pretrained_embedding, base_path, "text_generation", False)
        self.model = self.build_model(input_size, embedding_size, vocab_size)
        self.model = self.load_weights(self.model)
        self.tokenizer = self.load_tokenizer()
        self.input_size = input_size
        self.possible_words = list(self.tokenizer.word_index.keys())[:max_words]
        self.no_completion = (',', '!', '?', ':', ';', '.', '\n', '\n\n', '', ' ')
        self.end_sentence = ('!', '?', '.', '\n', '\n\n')
        self.no_space_needed = (" ", "\n", "\n\n")
        self.splitter = " "
        self.delimiters = "|".join([
            "!", "@", "#", "$", "%", "^", "&", "\\(", "\\)", "_", "-", ",", "<", "\\.",
            ">", "\\?", "`", "~", ":", ";", "\\+", "=", "[", "]", "{", "}", "\n{2,}", "\\s"
        ])
        _ = self.predict(" ")

    def split_text(self, text):
        text = text.lower()

        return re.sub(r'(' + self.delimiters + ')', r' \1 ', text).strip()

    def clean_text(self, text):
        text = self.split_text(text)
        text = self.normalizer.replace_char_rep(text=text)
        text = self.normalizer.replace_words_rep(text=text)
        text = self.normalizer.text_demojis(text=text)
        text = self.normalizer.text_demoticons(text=text)
        if self.cleaning_func:
            text = self.cleaning_func(text)
        text = self.normalizer.remove_multiple_spaces(text=text)
        text = text.strip()

        return text

    def to_sequence(self, text):
        tokenized_text = self.tokenizer.texts_to_sequences([text])[0]
        padded_tokenized_text = pad_sequences([tokenized_text], maxlen=self.input_size, padding='pre')

        return padded_tokenized_text

    def split_text_last_word(self, text):
        return text.split(self.splitter)[-1], text.split(self.splitter)[:-1]

    def get_prediction_startwith(self, predictions, word, threshold=0.3):
        tokens_ids = np.where(predictions[0] >= threshold)[0]
        scores = [predictions[0][idx] for idx in tokens_ids]
        token_score = list(zip(tokens_ids, scores))
        token_score_possible = dict(
            [(self.tokenizer.index_word.get(token), score) for (token, score) in token_score if score >= threshold])
        sorted_token = sorted(token_score_possible.items(), key=operator.itemgetter(1), reverse=True)
        for token, score in sorted_token:
            if token.startswith(word):
                return token, score

        return "", 0.0

    def word_completion(self, text, threshold=0.3):
        score, token = 0.0, ""
        if text[-1] == self.splitter:
            return text, [score], [token]
        splitted_text = self.split_text(text)
        last_word, begin_text = self.split_text_last_word(splitted_text)
        cleaned_text = self.clean_text(begin_text)
        if last_word not in self.no_completion:
            padded_tokenized_text = self.to_sequence(cleaned_text)
            predictions = self.model.predict(padded_tokenized_text)
            token, score = self.get_prediction_startwith(predictions,last_word, threshold)
            text += token[len(last_word):]

        return text, [score], [token[len(last_word):]]

    def next_word(self, text):
        splitted_text = self.split_text(text)
        last_word = splitted_text.split(self.splitter)[-1]
        cleaned_text = self.clean_text(splitted_text)
        tokenized_text = self.tokenizer.texts_to_sequences([cleaned_text])[0]
        padded_tokenized_text = pad_sequences([tokenized_text], maxlen=self.input_size, padding='pre')
        predictions = self.model.predict(padded_tokenized_text)
        prediction = np.argmax(predictions[0])
        token = self.tokenizer.index_word.get(prediction)
        score = predictions[0][prediction]
        if (token not in string.punctuation) and (token not in self.no_space_needed) and (text[-1] != self.splitter):
            if last_word in self.end_sentence:
                token = token.capitalize()
            token = " " + token

        return score, token

    def next_words(self, text, threshold=0.4, max_predictions=5):
        scores, tokens, score, number_predicted_words, continue_prediction = [], [], 1.0, 0, True
        while continue_prediction:
            score_token, token = self.next_word(text)
            score *= score_token
            if score >= threshold:
                scores.append(score)
                tokens.append(token)
                number_predicted_words += 1
                text += token
            else:
                continue_prediction = False
            if number_predicted_words == max_predictions:
                continue_prediction = False

        return text, scores, tokens

    def generate(self, x, max_predictions=100):
        text = x
        text, _, _ = self.word_completion(text)
        text, _, _ = self.next_words(text, threshold=0.0, max_predictions=max_predictions)
        predicted_text = text[len(x) :]

        return predicted_text

    def predict(self, x, completion_threshold=0.2, prediction_threshold=0.3, max_predictions=5):
        text = x
        text, completion_score, completion_token = self.word_completion(text, threshold=completion_threshold)
        text, next_word_scores, next_word_tokens = self.next_words(
            text, threshold=prediction_threshold, max_predictions=max_predictions)
        predicted_text = text[len(x) :]
        scores = completion_score + next_word_scores
        tokens = completion_token + next_word_tokens

        return predicted_text, scores, tokens


if __name__ == '__main__':
    predicter = Predicter(MODEL_PATH, "2020_08_02_19_16_52_text_generation")
    predicter.predict("Bonjour, je voulais", max_predictions=3)
    predicter.generate("Bonjour, je voulais", max_predictions=100)
