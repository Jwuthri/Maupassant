import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub


class BertEmbedding:

    def __init__(self, model='multilingual'):
        self.model = model.lower()

    def get_embedding(self, multi_output=True):
        assert self.model in ["multilingual", "multilingual-qa", "english"]
        module_mapping = {
            "multilingual": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "multilingual-qa": "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
            "english": "https://tfhub.dev/google/universal-sentence-encoder/4"
        }
        bert_module = hub.load(module_mapping[self.model]) if multi_output else module_mapping[self.model]

        return hub.KerasLayer(bert_module, input_shape=[], dtype=tf.string, trainable=False, name='multilingual_embed')
