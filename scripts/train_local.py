import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch.nn as nn
import torch.optim as optim

from src.config import (
    EPOCHS,
    LR,
    MOMENTUM,
    SEED,
    MODEL_NAME,
    TRAIN_RATIO,
    BATCH_SIZE,
    IMG_SIZE,
    NUM_WORKERS,
    PRETRAINED,
    USE_SUBSET,
    TRAIN_SUBSET_SIZE,
    TEST_SUBSET_SIZE,
    RUNS_DIR,
    get_device,
    get_experiment_name,
)
from src.dataset import build_dataloaders
from src.model import build_model
from src.train import train_one_epoch, evaluate_on_test
from src.utils import set_seed, create_run_dir, save_json, save_checkpoint


def main():
    set_seed(SEED)
    device = get_device()

    print("Building dataloaders...")
    full_dataset, train_loader, test_loader = build_dataloaders()

    print("Classes:", full_dataset.classes)
    print("Number of classes:", len(full_dataset.classes))
    print("Train samples:", len(train_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Device:", device)

    print("Building model...")
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    final_train_loss = None
    final_train_acc = None

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        final_train_loss = train_loss
        final_train_acc = train_acc

        print(f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f}")

    print("Evaluating on test set...")
    test_loss, test_acc = evaluate_on_test(
        model, test_loader, criterion, device
    )

    print(f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

    experiment_name = get_experiment_name()
    # create run folder
    run_dir = create_run_dir(RUNS_DIR, MODEL_NAME, experiment_name)

    # save config snapshot
    config_data = {
        "model_name": MODEL_NAME,
        "train_ratio": TRAIN_RATIO,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "lr": LR,
        "momentum": MOMENTUM,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "pretrained": PRETRAINED,
        "use_subset": USE_SUBSET,
        "train_subset_size": TRAIN_SUBSET_SIZE,
        "test_subset_size": TEST_SUBSET_SIZE,
        "device": device,
        "num_classes": len(full_dataset.classes),
        "classes": full_dataset.classes,
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }

    # save metrics
    metrics_data = {
        "train_loss": final_train_loss,
        "train_acc": final_train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    save_json(run_dir / "config.json", config_data)
    save_json(run_dir / "metrics.json", metrics_data)
    save_checkpoint(run_dir / "model.pth", model)

    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()