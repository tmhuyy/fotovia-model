from collections import defaultdict

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
    TRAIN_SUBSET_PER_CLASS,
    TEST_SUBSET_PER_CLASS,
)


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def subset_indices_per_class(indices, targets, per_class_limit):
    grouped = defaultdict(list)

    for idx in indices:
        label = targets[idx]
        grouped[label].append(idx)

    selected_indices = []

    for label in sorted(grouped.keys()):
        class_indices = grouped[label]
        limit = min(per_class_limit, len(class_indices))
        selected_indices.extend(class_indices[:limit])

    return selected_indices


def apply_subset_if_needed(full_dataset, train_indices, test_indices):
    if not USE_SUBSET:
        return train_indices, test_indices

    targets = full_dataset.targets

    train_indices = subset_indices_per_class(
        train_indices,
        targets,
        TRAIN_SUBSET_PER_CLASS,
    )

    test_indices = subset_indices_per_class(
        test_indices,
        targets,
        TEST_SUBSET_PER_CLASS,
    )

    return train_indices, test_indices


def build_datasets():
    transform = build_transforms()
    full_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=transform)

    targets = full_dataset.targets
    class_to_idx = full_dataset.class_to_idx

    all_train_indices = []
    all_test_indices = []

    for class_name, class_idx in class_to_idx.items():
        class_indices = [i for i, label in enumerate(targets) if label == class_idx]

        train_indices, test_indices = train_test_split(
            class_indices,
            train_size=TRAIN_RATIO,
            random_state=SEED,
            shuffle=True,
        )

        all_train_indices.extend(train_indices)
        all_test_indices.extend(test_indices)

        print(
            f"{class_name}: total={len(class_indices)}, "
            f"train={len(train_indices)}, test={len(test_indices)}"
        )

    all_train_indices, all_test_indices = apply_subset_if_needed(
        full_dataset,
        all_train_indices,
        all_test_indices,
    )

    train_dataset = Subset(full_dataset, all_train_indices)
    test_dataset = Subset(full_dataset, all_test_indices)

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