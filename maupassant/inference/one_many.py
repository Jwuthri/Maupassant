import os
import glob
import pickle

import tensorflow as tf
import tensorflow_text

from maupassant.utils import timer


class Predictor(object):

    def __init__(self, model_dir):
        self.model = self.load_model(model_dir)
        self.encoders = self.load_encoder(model_dir)

    @timer
    def __call__(self, x):
        preds = self.model(x)
        print(preds)
        predictions = []
        for pred in preds:
            size = pred.shape[1]
            for k, v in self.encoders.items():
                if len(v) == size:
                    p_label = []
                    for p in pred:
                        p_label.append([(v[label], float(th)) for label, th in enumerate(p) if th >= 0.5])
                    predictions.append(p_label)
        return list(zip(*predictions))

    def load_model(self, model_dir):
        return tf.saved_model.load(model_dir)

    def load_encoder(self, model_dir):
        encoders_files = glob.glob(model_dir + "/*encoder.pkl")
        encoders = {}
        for file in encoders_files:
            encoder = pickle.load(open(file, "rb"))
            encoder_name = os.path.split(file)[1].split('.')[0]
            encoders[encoder_name] = dict(enumerate(encoder.classes_))

        return encoders


if __name__ == '__main__':
    path = "/home/jwuthri/Documents/GitHub/Maupassant/maupassant/models/one_to_many_2020_03_17_16_34_05"
    predictor = Predictor(path)
    print(predictor(x=['where is my order?', 'Je veux un remboursement et un return']))
    print(predictor(x=['Customer support so bad. Help me with my issue please']))
    print(predictor(x=['Je suis pas content']))
