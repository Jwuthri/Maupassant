import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from maupassant.utils import predict_format


class PretrainedEmbedding(object):

    def __init__(self, name="pretrained_embedding"):
        self.name = name
        self.model = self.get_model()

    def get_model(self):
        bert_module = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"

        return hub.KerasLayer(bert_module, input_shape=[], dtype=tf.string, trainable=False, name=self.name)

    @predict_format
    def predict_one(self, x):
        return self.model(x)

    @predict_format
    def predict_batch(self, x):
        raise self.model(x)
