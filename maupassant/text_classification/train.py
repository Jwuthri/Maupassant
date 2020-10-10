import os

import pandas as pd

from maupassant.settings import DATASET_PATH
from maupassant.tensorflow_helper.trainer_helper import TensorflowTrainer


if __name__ == '__main__':
    dataset_path = os.path.join(DATASET_PATH, "intent.csv")
    architecture = [('LCNN', 512), ("GLOBAL_AVERAGE_POOL", 0), ("DROPOUT", 0.1), ('DENSE', 256)]
    dataset = pd.read_csv(dataset_path)
    x_col, y_col, label_type, epochs, name = "x", "y", "multi-label", 5, "sentiment"

    projects = {
        "intent": "intent-classifier",
        "sentiment": "sentiment-classifier",
        "summarization": "text-summarizer",
        "macro": "macro-suggestion-stacker",
        "phrase": "phrase-prediction",
    }
    api_key = "EJGg4eN8erTkn5o2OHAZyN7RY"
    project_name = projects.get(name, "gorgias-ml")
    workspace = "gorgias"

    trainer = TensorflowTrainer(
        label_type, name, architecture, api_key=api_key, project_name=project_name, workspace=workspace, epochs=epochs
    )
    trainer.train(dataset, x_col, y_col)
