from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(r"D:\Huy_Feat\dataset")

MODEL_NAME = "resnext"
NUM_CLASSES = 10

EPOCHS = 1
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.01
MOMENTUM = 0.9
TRAIN_RATIO = 0.7
SEED = 42
NUM_WORKERS = 2
PRETRAINED = False

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
RUNS_DIR = OUTPUT_DIR / "runs"
DOCS_DIR = PROJECT_ROOT / "docs"

############ SMOKE TEST ###############
USE_SUBSET = True # TODO: CHANGE IT TO FALSE WHEN REAL TRAIN
TRAIN_SUBSET_SIZE = 5000
TEST_SUBSET_SIZE = 2000
TRAIN_SUBSET_PER_CLASS = 500
TEST_SUBSET_PER_CLASS = 200

CUSTOM_EXPERIMENT_NAME = None
#######################################

def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_experiment_name() -> str:
    if CUSTOM_EXPERIMENT_NAME:
        return CUSTOM_EXPERIMENT_NAME
    return "smoke" if USE_SUBSET else "full"