from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from maupassant.tensorflow_models_compile import ModelSaverLoader
from maupassant.settings import MODEL_PATH


class LabelEncoding(ModelSaverLoader):

    def __init__(self, multi_label, base_path=MODEL_PATH, name="label_encoder", model_load=False):
        super().__init__(base_path, name, model_load)
        self.multi_label = multi_label
        self.classes_mapping = dict()
        self.number_labels = 0
        self.encoder = self.init_encoder()

    def init_encoder(self):
        if self.multi_label:
            return MultiLabelBinarizer()
        else:
            return LabelEncoder()

    def fit_encoder(self, y):
        self.encoder.fit(y)
        self.classes_mapping = dict(enumerate(self.encoder.classes_))
        self.number_labels = len(self.encoder.classes_)

    def transform_encoder(self, y):
        return self.encoder.transform(y)

    def inverse_encoder(self, y):
        return self.encoder.inverse_transform(y)
