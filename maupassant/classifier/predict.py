import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from maupassant.feature_extraction.embedding import BertEmbedding
from maupassant.tensorflow_utils import PredictHelper


class Predictor(PredictHelper):

    def __init__(self, model_path):
        self.model_path = model_path
