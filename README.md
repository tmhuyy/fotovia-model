# fotovia-model

## 1. Current Confirmed Status

### Local machine
- **MacBook Air M1 (2020)**
- RAM: **16 GB**
- macOS: **Sequoia 15.6.1**

### Python / Torch environment
The local environment has already been fixed so it can use the Apple M1 GPU:

- `Machine: arm64`
- `Torch version: 2.11.0`
- `MPS built: True`
- `MPS available: True`

This means the local notebook now **can use `mps`**, and no longer falls back to `cpu` like before.

---

## 2. Current Dataset Structure

The 10 classes have already been extracted and currently have this format:

```text
dataset/
  aerial/
  architecture/
  event/
  fashion/
  food/
  nature/
  sports/
  street/
  wedding/
  wildlife/
```

Each class folder contains its corresponding images.

Notes:
- The dataset is **not pre-split into train/test**
- The **70/30 split** will be done in code, not manually in Finder

---

## 3. Local Environment Setup

### 3.1 Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install packages

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio pillow tqdm scikit-learn matplotlib ipykernel
```

### 3.3 Create notebook kernel

```bash
python -m ipykernel install --user --name fotovia-m1 --display-name "Python (.venv-fotovia-m1)"
```

### 3.4 Select the correct kernel in the VS Code notebook

In the `.ipynb` file:
- click the kernel selector in the top-right corner
- choose the local `.venv` kernel

If the notebook accidentally uses a Colab-style kernel or `/usr/bin/python3`, it may run in the wrong environment again.


## 4. Current Notebook Config

Basic config currently used:

```python
DATASET_ROOT = "/PATH/TO/YOUR/dataset"
MODEL_NAME = "resnext"   # "resnext", "wideresnet", "efficientnet"
EPOCHS = 1
BATCH_SIZE = 8
IMG_SIZE = 224
LR = 0.01
MOMENTUM = 0.9
TRAIN_RATIO = 0.7
SEED = 42
NUM_WORKERS = 0
PRETRAINED = False
```

### Short explanation
- `DATASET_ROOT`: path to the `dataset` folder
- `MODEL_NAME`: which model to train
- `EPOCHS`: number of full training passes through the train set
- `BATCH_SIZE`: number of images processed at one time
- `IMG_SIZE`: image resize size
- `LR`: learning rate
- `MOMENTUM`: SGD momentum
- `TRAIN_RATIO`: percentage used for training, the rest is testing
- `SEED`: fixes randomness so the split is stable
- `NUM_WORKERS`: number of worker processes for loading data
- `PRETRAINED`: whether to use pretrained weights or not

---

## 5. Recommended Notebook Workflow

The notebook should be split into cells with this flow:

1. **Imports**
2. **Config**
3. **Seed + device**
4. **Transforms**
5. **Load dataset**
6. **Split 70/30 per class**
7. **DataLoader**
8. **Build model**
9. **Train / evaluate functions**
10. **Run training**
11. **Save history**
12. **Save train/test split**

---

## 6. How to Split Train/Test

Because the dataset is not pre-split, the code should:
- read all images per class
- shuffle with `random.seed(42)`
- split **70% train / 30% test** for each class

Notes:
- do **not** duplicate images into physical `train/` and `test/` folders
- split by **indices/path lists** in code to save disk space
