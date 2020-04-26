import os
from dotenv import load_dotenv
load_dotenv()

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ABS_PATH, "models")
DATA_PATH = os.path.join(ABS_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")
EXTERNAL_PATH = os.path.join(DATA_PATH, "external")

try:
    API_KEY = os.environ['API_KEY']
    PROJECT_NAME = os.environ['PROJECT_NAME']
    WORKSPACE = os.environ['WORKSPACE']
except KeyError:
    API_KEY = None
    PROJECT_NAME = None
    WORKSPACE = None
