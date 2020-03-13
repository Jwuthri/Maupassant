import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ABS_PATH, "models")
DATA_PATH = os.path.join(ABS_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")
LOGS_PATH = os.path.join(DATA_PATH, "logs")
