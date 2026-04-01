from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(r"E:\dataset")

MODEL_NAME = "resnext"
NUM_CLASSES = 10

EPOCHS = 1
BATCH_SIZE = 4
IMG_SIZE = 224
LR = 0.01
MOMENTUM = 0.9
TRAIN_RATIO = 0.7
SEED = 42
NUM_WORKERS = 0
PRETRAINED = False

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"