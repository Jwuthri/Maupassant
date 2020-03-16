import os
from dotenv import load_dotenv
load_dotenv()

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ABS_PATH, "models")
DATA_PATH = os.path.join(ABS_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")
LOGS_PATH = os.path.join(DATA_PATH, "logs")

API_KEY = os.environ['API_KEY']
PROJECT_NAME = os.environ['PROJECT_NAME']
WORKSPACE = os.environ['WORKSPACE']