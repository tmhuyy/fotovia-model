import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch.nn as nn
import torch.optim as optim

from src.config import EPOCHS, LR, MOMENTUM, SEED, get_device
from src.dataset import build_dataloaders
from src.model import build_model
from src.train import train_one_epoch, validate_one_epoch
from src.utils import set_seed


def main():
    set_seed(SEED)
    device = get_device()

    print("Building dataloaders...")
    full_dataset, train_loader, val_loader = build_dataloaders()

    print("Classes:", full_dataset.classes)
    print("Number of classes:", len(full_dataset.classes))
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Device:", device)

    print("Building model...")
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()