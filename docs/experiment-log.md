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
