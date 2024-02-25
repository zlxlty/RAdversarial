import os

# Get absolute path of root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

EVAL_PATH = f"{ROOT_DIR}/evals"
IMAGE_PATH = f"{ROOT_DIR}/images"

CONFIG_PATH = f"{ROOT_DIR}/attacks/config"
CACHE_PATH = f"{ROOT_DIR}/.cache"
DATASET_PATH = f'/dataset/benchmark10000rand'
# DATASET_PATH = f"{ROOT_DIR}/benchmark10000rand"

if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

os.environ["HF_HOME"] = CACHE_PATH
