import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
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
    TEST_SUBSET_SIZE,
)


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def apply_subset_if_needed(train_dataset, test_dataset):
    if not USE_SUBSET:
        return train_dataset, test_dataset

    train_limit = min(TRAIN_SUBSET_SIZE, len(train_dataset))
    test_limit = min(TEST_SUBSET_SIZE, len(test_dataset))

    train_indices = list(range(train_limit))
    test_indices = list(range(test_limit))

    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)

    return train_dataset, test_dataset


def build_datasets():
    transform = build_transforms()
    full_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=transform)

    indices = list(range(len(full_dataset)))
    targets = full_dataset.targets

    train_indices, test_indices = train_test_split(
        indices,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
        stratify=targets,
    )

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_dataset, test_dataset = apply_subset_if_needed(train_dataset, test_dataset)

    return full_dataset, train_dataset, test_dataset


def build_dataloaders():
    full_dataset, train_dataset, test_dataset = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return full_dataset, train_loader, test_loader