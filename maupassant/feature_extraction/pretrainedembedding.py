import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from maupassant.utils import predict_format


class PretrainedEmbedding(object):

    def __init__(self, model_type='multilingual', name="embedding_layer"):
        self.name = name
        self.model_type = model_type.lower()
        self.model = self.init_model()

    def init_model(self):
        assert self.model_type in ["multilingual", "multilingual-qa", "universal-encoder"]
        module_mapping = {
            "multilingual": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "multilingual-qa": "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
            "english": "https://tfhub.dev/google/universal-sentence-encoder/4"
        }
        bert_module = module_mapping[self.model_type]

        return hub.KerasLayer(bert_module, input_shape=[], dtype=tf.string, trainable=False, name=self.name)

    @predict_format
    def predict_one(self, x):
        return self.model(x)

    @predict_format
    def predict_batch(self, x):
        raise NotImplemented
