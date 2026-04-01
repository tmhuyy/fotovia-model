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