import tensorflow as tf

from maupassant.settings import MODEL_PATH, USE_GPU
from maupassant.utils import predict_format, timer
from maupassant.preprocessing.normalization import TextNormalization
from maupassant.tensorflow_models_compile import BaseTensorflowModel, ModelSaverLoader

GPUS = tf.config.experimental.list_physical_devices("GPU")
if GPUS:
    try:
        for gpu in GPUS:
            if USE_GPU:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                assert gpu.device_type != "GPU"
    except RuntimeError as e:
        print(e)


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
        self.build_model(0, embedding_size, 0)
        self.model = self.load_weights(self.model)
        self.encoders = self.load_encoder()
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

    @staticmethod
    def predict_classes(prediction, threshold, encoder):
        results = [(encoder.classes_[label], float(th)) for label, th in enumerate(prediction) if float(th) >= threshold]

        return dict(results)

    @timer
    def predict(self, x, threshold=0.5, clean=False):
        if clean:
            x = self.clean_text(x)
        probabilities = self.predict_probabilities(x=x)
        classes = []
        for i in range(len(self.encoders)):
            classes.append(self.predict_classes(probabilities[i], threshold, self.encoders[i]))

        return classes


if __name__ == '__main__':
    predicter = Predicter(MODEL_PATH, "2020_09_01_16_40_23_text_classification")
    pred = predicter.predict("You are fucking retard", threshold=0.5)
    print(pred)
    pred = predicter.predict("Je vous aime", threshold=0.5)
    print(pred)
