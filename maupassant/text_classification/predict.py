import tensorflow as tf

from maupassant.settings import MODEL_PATH
from maupassant.utils import ModelSaverLoader, predict_format, timer
from maupassant.preprocessing.normalization import TextNormalization
from maupassant.tensorflow_models_compile import BaseTensorflowModel

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()


class Predicter(BaseTensorflowModel):

    def __init__(self, base_path=MODEL_PATH, name='model_name', cleaning_func=None):
        self.normalizer = TextNormalization()
        self.cleaning_func = cleaning_func
        msl = ModelSaverLoader(base_path, name, True)
        info = msl.load_info()
        label_type = info.get('label_type')
        architecture = info.get('architecture')
        number_labels = info.get('number_labels')
        pretrained_embedding = info.get('pretrained_embedding')
        embedding_size = info.get('embedding_size')
        super().__init__(label_type, architecture, number_labels, pretrained_embedding, base_path, name, True)
        self.model = self.build_model(0, embedding_size, 0)
        self.model = self.load_weights(self.model)
        self.encoder = self.load_encoder()
        self.classes = self.encoder.classes_
        _ = self.predict(" ")

    def clean_text(self, text):
        text = self.normalizer.replace_char_rep(text=text)
        text = self.normalizer.replace_words_rep(text=text)
        if self.cleaning_func:
            text = self.cleaning_func(text)
        text = self.normalizer.remove_multiple_spaces(text=text)
        text = text.strip()

        return text

    @predict_format
    def predict_probabilities(self, x):
        return self.model.predict(x)

    def predict_classes(self, prediction, threshold):
        results = [(self.classes[label], float(th)) for label, th in enumerate(prediction) if float(th) >= threshold]

        return dict(results)

    @timer
    def predict(self, x, threshold=0.5):
        x = self.clean_text(x)
        probabilities = self.predict_probabilities(x=x)
        classes = self.predict_classes(probabilities[0], threshold)

        return classes


if __name__ == '__main__':
    predicter = Predicter(MODEL_PATH, "2020_08_02_19_16_52_text_classification")
    classes = predicter.predict("You are stupid")
