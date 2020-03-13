import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from ttk.utils import timer


class BertEmbedding:

    def __init__(self, model='multilingual'):
        self.model = model.lower()
        self.embedding = self.get_embedding

    @property
    def get_embedding(self):
        assert self.model in ["multilingual", "multilingual-qa", "english"]
        module_mapping = {
            "multilingual": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "multilingual-qa": "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
            "english": "https://tfhub.dev/google/universal-sentence-encoder/4"
        }
        bert_module = module_mapping[self.model]

        return hub.KerasLayer(bert_module, input_shape=[], dtype=tf.string, trainable=False)

    @timer
    def fit(self, x):
        raise NotImplemented

    @timer
    def transform(self, txt):
        return self.embedding(txt)

    def save(self, filename):
        raise NotImplemented
