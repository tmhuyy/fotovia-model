import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.dataset import build_datasets

full_dataset, train_dataset, test_dataset = build_datasets()

train_indices = train_dataset.indices
test_indices = test_dataset.indices
targets = full_dataset.targets

train_counts = Counter(targets[i] for i in train_indices)
test_counts = Counter(targets[i] for i in test_indices)

idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

print("TRAIN COUNTS")
for label in sorted(train_counts.keys()):
    print(idx_to_class[label], train_counts[label])

print("\nTEST COUNTS")
for label in sorted(test_counts.keys()):
    print(idx_to_class[label], test_counts[label])