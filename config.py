import pathlib

# Path to the root directory of the project
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

HF_USERNAME = "therem"
BASE_MODEL_PATH = "lvwerra/gpt2-imdb"
TRAIN_DATASET_PATH = "therem/dpo_dataset"
EVAL_DATASET_PATH = "therem/dpo_dataset_eval"