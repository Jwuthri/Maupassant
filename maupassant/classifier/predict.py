from maupassant.tensorflow_utils import PredictHelper


class Predictor(PredictHelper):

    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(self.model_path)
