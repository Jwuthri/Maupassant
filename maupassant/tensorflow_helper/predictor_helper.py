import os
import json

from maupassant.settings import MODEL_PATH
from maupassant.tensorflow_helper.model_helper import TensorflowModel


class TensorflowPredicter(TensorflowModel):

    def __init(self, base_path=MODEL_PATH, name='model_name', **kwargs):
        path = os.path.join(base_path, name, "model.json")
        info = self.infos(path)
        architecture = info.get('architecture')
        label_type = info.get('label_type')
        super().__init__(architecture, label_type, name, base_path=base_path, **kwargs)

    @staticmethod
    def infos(path):
        with open(path, "rb") as json_file:
            info = json.load(json_file)

        return info