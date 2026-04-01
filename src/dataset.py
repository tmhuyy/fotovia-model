from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

from src.config import (
    DATASET_ROOT,
    IMG_SIZE,
    TRAIN_RATIO,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def build_datasets():
    transform = build_transforms()
    full_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=transform)

    train_size = int(len(full_dataset) * TRAIN_RATIO)
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    return full_dataset, train_dataset, val_dataset


def build_dataloaders():
    full_dataset, train_dataset, val_dataset = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return full_dataset, train_loader, val_loader