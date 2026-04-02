# Experiment Log

This file records important experiment runs for the Photozilla replication project.

---

## Run 001

- Date:
- Run ID:
- Experiment Name:
- Purpose:
  - Example: smoke test / baseline classification / full dataset run / checkpoint test

### Model Setup
- Model:
- Pretrained:
- Device:
- Epochs:
- Batch Size:
- Image Size:
- Optimizer:
- Learning Rate:
- Momentum:
- Loss Function:

### Dataset Setup
- Dataset:
- Number of Classes:
- Split Strategy:
  - Example: stratified 70% train / 30% test
- Train Samples:
- Test Samples:
- Subset Enabled:
- Train Subset Size:
- Test Subset Size:

### Results
- Train Loss:
- Train Accuracy:
- Test Loss:
- Test Accuracy:

### Notes
- What was the goal of this run?
- Did the pipeline run successfully?
- Any issues found?
- Any changes needed for the next run?

---

## Run 002

- Date:
- Run ID:
- Experiment Name:
- Purpose:

### Model Setup
- Model:
- Pretrained:
- Device:
- Epochs:
- Batch Size:
- Image Size:
- Optimizer:
- Learning Rate:
- Momentum:
- Loss Function:

### Dataset Setup
- Dataset:
- Number of Classes:
- Split Strategy:
- Train Samples:
- Test Samples:
- Subset Enabled:
- Train Subset Size:
- Test Subset Size:

### Results
- Train Loss:
- Train Accuracy:
- Test Loss:
- Test Accuracy:

### Notes
- 

---

## Suggested Naming Convention

Use a clear experiment name so that each run is easy to identify.

Examples:
- `smoke`
- `baseline-resnet18`
- `baseline-resnext`
- `full-train-resnet18`
- `full-train-resnext`
- `paper-replication`
- `debug-split-check`

---

## Suggested Workflow

For each important run:
1. Run the training script
2. Save outputs to `outputs/runs/...`
3. Copy key results into this file
4. Add notes about what changed from the previous run

This file is useful for:
- tracking progress
- comparing experiments
- writing the thesis report later

## 2026-04-02_060747_resnext_smoke

- Date: 2026-04-02
- Run ID: 2026-04-02_060747_resnext_smoke
- Experiment Name: smoke
- Purpose: smoke test

### Model Setup
- Model: resnext
- Pretrained: False
- Device: cpu
- Epochs: 1
- Batch Size: 4
- Image Size: 224
- Optimizer: SGD
- Learning Rate: 0.01
- Momentum: 0.9
- Loss Function: CrossEntropyLoss

### Dataset Setup
- Dataset: E:\dataset
- Number of Classes: 10
- Split Strategy: stratified 70% train / 30% test
- Train Samples: 70
- Test Samples: 30
- Subset Enabled: True
- Train Subset Size: 70
- Test Subset Size: 30

### Results
- Train Loss: 14.4999
- Train Accuracy: 0.1286
- Test Loss: 8555.0426
- Test Accuracy: 0.0333

### Notes
- Output folder: D:\Huy_Feat\GitHub\fotovia-model\outputs\runs\2026-04-02_060747_resnext_smoke
- Checkpoint file: model.pth
- Auto-generated after training run.

---

## 2026-04-02_115313_resnext_full

- Date: 2026-04-02
- Run ID: 2026-04-02_115313_resnext_full
- Experiment Name: full
- Purpose: full training

### Model Setup
- Model: resnext
- Pretrained: True
- Device: cuda
- Epochs: 1
- Batch Size: 128
- Image Size: 224
- Optimizer: SGD
- Learning Rate: 0.01
- Momentum: 0.9
- Loss Function: CrossEntropyLoss

### Dataset Setup
- Dataset: /workspace/fotovia-model/dataset/dataset
- Number of Classes: 10
- Split Strategy: stratified 70% train / 30% test
- Train Samples: 693030
- Test Samples: 297020
- Subset Enabled: False
- Train Subset Size: 5000
- Test Subset Size: 2000

### Results
- Train Loss: 0.1120
- Train Accuracy: 0.9578
- Test Loss: 0.0905
- Test Accuracy: 0.9629

### Notes
- Output folder: /workspace/fotovia-model/outputs/runs/2026-04-02_115313_resnext_full
- Checkpoint file: model.pth
- Auto-generated after training run.

---

## 2026-04-02_120024_resnet18_full

- Date: 2026-04-02
- Run ID: 2026-04-02_120024_resnet18_full
- Experiment Name: full
- Purpose: full training

### Model Setup
- Model: resnet18
- Pretrained: True
- Device: cuda
- Epochs: 1
- Batch Size: 128
- Image Size: 224
- Optimizer: SGD
- Learning Rate: 0.01
- Momentum: 0.9
- Loss Function: CrossEntropyLoss

### Dataset Setup
- Dataset: /workspace/fotovia-model/dataset/dataset
- Number of Classes: 10
- Split Strategy: stratified 70% train / 30% test
- Train Samples: 693030
- Test Samples: 297020
- Subset Enabled: False
- Train Subset Size: 5000
- Test Subset Size: 2000

### Results
- Train Loss: 0.1222
- Train Accuracy: 0.9556
- Test Loss: 0.1004
- Test Accuracy: 0.9602

### Notes
- Output folder: /workspace/fotovia-model/outputs/runs/2026-04-02_120024_resnet18_full
- Checkpoint file: model.pth
- Auto-generated after training run.

---

## 2026-04-02_123559_wide_resnet_full

- Date: 2026-04-02
- Run ID: 2026-04-02_123559_wide_resnet_full
- Experiment Name: full
- Purpose: full training

### Model Setup
- Model: wide_resnet
- Pretrained: True
- Device: cuda
- Epochs: 1
- Batch Size: 128
- Image Size: 224
- Optimizer: SGD
- Learning Rate: 0.01
- Momentum: 0.9
- Loss Function: CrossEntropyLoss

### Dataset Setup
- Dataset: /workspace/fotovia-model/dataset/dataset
- Number of Classes: 10
- Split Strategy: stratified 70% train / 30% test
- Train Samples: 693030
- Test Samples: 297020
- Subset Enabled: False
- Train Subset Size: 5000
- Test Subset Size: 2000

### Results
- Train Loss: 0.1119
- Train Accuracy: 0.9580
- Test Loss: 0.0917
- Test Accuracy: 0.9631

### Notes
- Output folder: /workspace/fotovia-model/outputs/runs/2026-04-02_123559_wide_resnet_full
- Checkpoint file: model.pth
- Auto-generated after training run.

---

## 2026-04-02_124524_efficientnet_full

- Date: 2026-04-02
- Run ID: 2026-04-02_124524_efficientnet_full
- Experiment Name: full
- Purpose: full training

### Model Setup
- Model: efficientnet
- Pretrained: True
- Device: cuda
- Epochs: 1
- Batch Size: 128
- Image Size: 224
- Optimizer: SGD
- Learning Rate: 0.01
- Momentum: 0.9
- Loss Function: CrossEntropyLoss

### Dataset Setup
- Dataset: /workspace/fotovia-model/dataset/dataset
- Number of Classes: 10
- Split Strategy: stratified 70% train / 30% test
- Train Samples: 693030
- Test Samples: 297020
- Subset Enabled: False
- Train Subset Size: 5000
- Test Subset Size: 2000

### Results
- Train Loss: 0.1323
- Train Accuracy: 0.9536
- Test Loss: 0.0993
- Test Accuracy: 0.9616

### Notes
- Output folder: /workspace/fotovia-model/outputs/runs/2026-04-02_124524_efficientnet_full
- Checkpoint file: model.pth
- Auto-generated after training run.

---
