import numpy as np

from maupassant.utils import timer
from maupassant.summarizer.model import TensorflowPredictorHelper


class Predictor(TensorflowPredictorHelper):

    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(self.model_path)

    @timer
    def predict(self, text, threshold=0.5):
        return self.predict(x=text, threshold=threshold)


if __name__ == '__main__':
    import os
    from maupassant.settings import MODEL_PATH

    model_path = os.path.join(MODEL_PATH, "binary-label_is_relevant_2020_05_28_10_42_11")
    pred = Predictor(model_path)
    data = [
        np.asarray(['My order number is 62767']),
        np.asarray(['hello. where is my order? My order number is 62767.'])
    ]
    print(pred.predict(x=data))
