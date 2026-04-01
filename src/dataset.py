import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from src.config import (
    DATASET_ROOT,
    IMG_SIZE,
    TRAIN_RATIO,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    USE_SUBSET,
    TRAIN_SUBSET_SIZE,
    VAL_SUBSET_SIZE,
)


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def apply_subset_if_needed(train_dataset, val_dataset):
    if not USE_SUBSET:
        return train_dataset, val_dataset

    train_limit = min(TRAIN_SUBSET_SIZE, len(train_dataset))
    val_limit = min(VAL_SUBSET_SIZE, len(val_dataset))

    train_indices = list(range(train_limit))
    val_indices = list(range(val_limit))

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    return train_dataset, val_dataset


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

    train_dataset, val_dataset = apply_subset_if_needed(train_dataset, val_dataset)

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