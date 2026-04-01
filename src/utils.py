from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_run_dir(base_dir: Path, model_name: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_{experiment_name}"
    run_dir = base_dir / run_name
    ensure_dir(run_dir)
    return run_dir


def save_json(filepath: Path, data: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_checkpoint(filepath: Path, model: torch.nn.Module) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def append_experiment_log(
    log_path: Path,
    run_id: str,
    experiment_name: str,
    config_data: dict,
    metrics_data: dict,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""
## {run_id}

- Date: {run_id.split("_")[0]}
- Run ID: {run_id}
- Experiment Name: {experiment_name}
- Purpose: {"smoke test" if config_data["use_subset"] else "full training"}

### Model Setup
- Model: {config_data["model_name"]}
- Pretrained: {config_data["pretrained"]}
- Device: {config_data["device"]}
- Epochs: {config_data["epochs"]}
- Batch Size: {config_data["batch_size"]}
- Image Size: {config_data["img_size"]}
- Optimizer: SGD
- Learning Rate: {config_data["lr"]}
- Momentum: {config_data["momentum"]}
- Loss Function: CrossEntropyLoss

### Dataset Setup
- Dataset: {config_data["dataset_root"]}
- Number of Classes: {config_data["num_classes"]}
- Split Strategy: stratified 70% train / 30% test
- Train Samples: {config_data["train_samples"]}
- Test Samples: {config_data["test_samples"]}
- Subset Enabled: {config_data["use_subset"]}
- Train Subset Size: {config_data["train_subset_size"]}
- Test Subset Size: {config_data["test_subset_size"]}

### Results
- Train Loss: {metrics_data["train_loss"]:.4f}
- Train Accuracy: {metrics_data["train_acc"]:.4f}
- Test Loss: {metrics_data["test_loss"]:.4f}
- Test Accuracy: {metrics_data["test_acc"]:.4f}

### Notes
- Output folder: {config_data["run_dir"]}
- Checkpoint file: model.pth
- Auto-generated after training run.

---
"""

    if not log_path.exists():
        log_path.write_text("# Experiment Log\n\n", encoding="utf-8")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content)