"""
src/data_loader.py

Loads the processed (X, y, subject) arrays produced by preprocessing.py and
exposes two split modes, both required per the evaluation protocol:

  1. Official train/test split   -> get_official_split()
  2. Subject-independent k-fold  -> get_loso_folds()   [headline result]

Normalization is deliberately NOT done in preprocessing.py -- it happens here,
per split/fold, fit on the TRAIN portion only, to avoid leakage. Every
training script should go through build_dataloaders() rather than reading
the .npy files directly.

train.py uses iter_folds(), which yields both protocols in one format and
additionally carves a SUBJECT-WISE validation set out of each fold's
training subjects (for early stopping), with labels encoded through one
dataset-wide mapping. get_official_split()/get_loso_folds() are kept with
their original return values for the notebooks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

from src.preprocessing import normalize_channels, PROCESSED_DIR

# Number of subject-groups to fold into for LOSO CV. True leave-ONE-out is used
# where the subject count is small (PAMAP2); larger subject pools are grouped
# into folds for tractable runtime. Document/justify in the report if changed.
# 10/12 folds (protocol v2, was 6/6): with 6 folds a two-sided Wilcoxon can't go
# below p=0.031 (0.19 after Holm over 6 comparisons), so no per-dataset test could pass.
LOSO_N_FOLDS = {
    "ucihar": 10,  # 30 subjects -> 3 subjects held out per fold
    "wisdm": 12,   # 36 subjects -> 3 subjects held out per fold
    "pamap2": 8,   # 8 subjects (109 excluded) -> true leave-one-subject-out
}

# Fixed random seed for any split that isn't dictated by an official protocol
# (subject-level shuffling only -- never shuffles within a subject's windows).
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Raw array loading
# ---------------------------------------------------------------------------
def _load_processed(dataset: str, harmonized: bool = False):
    suffix = "_harmonized" if harmonized else ""
    d = PROCESSED_DIR / dataset
    X = np.load(d / f"X{suffix}.npy")
    y = np.load(d / f"y{suffix}.npy")
    subject = np.load(d / f"subject{suffix}.npy")
    official_split = None
    p = d / f"official_split{suffix}.npy"
    if p.exists():
        official_split = np.load(p, allow_pickle=True)
    return X, y, subject, official_split


class HARDataset(Dataset):
    """Simple (X, y) wrapper. X: (N, T, C) float32, y: (N,) int64."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Split 1: official train/test
# ---------------------------------------------------------------------------
def get_official_split(dataset: str, seed: int = SPLIT_SEED):
    """
    Returns (X_train, y_train, X_test, y_test), normalized using train-fold
    statistics.

    - UCI-HAR: uses the dataset's real official split (saved by preprocessing.py).
    - WISDM / PAMAP2: no official split is published, so we do a fixed,
      seeded, SUBJECT-LEVEL 80/20 split (never split within a subject).
      This is a documented substitute, not a claimed "official" protocol --
      say so explicitly in the report's evaluation section.
    """
    X, y, subject, official_split = _load_processed(dataset)

    if official_split is not None:
        train_mask = official_split == "train"
        test_mask = official_split == "test"
    else:
        rng = np.random.RandomState(seed)
        unique_subjects = np.unique(subject)
        rng.shuffle(unique_subjects)
        n_test = max(1, int(round(0.2 * len(unique_subjects))))
        test_subjects = set(unique_subjects[:n_test])
        train_mask = np.array([s not in test_subjects for s in subject])
        test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    X_train, X_test, mean, std = normalize_channels(X_train, X_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Split 2: subject-independent (LOSO) k-fold -- the headline evaluation
# ---------------------------------------------------------------------------
def get_loso_folds(dataset: str, n_folds: Optional[int] = None, seed: int = SPLIT_SEED):
    """
    Generator yielding one dict per fold:
        {fold_id, held_out_subjects, X_train, y_train, X_test, y_test}

    Subjects are randomly (seed-fixed) partitioned into n_folds disjoint
    groups; each fold holds one group out entirely (leave-N-subjects-out).
    Normalization is fit on that fold's train subjects only.
    """
    X, y, subject, _ = _load_processed(dataset)
    n_folds = n_folds or LOSO_N_FOLDS[dataset]

    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(subject)
    rng.shuffle(unique_subjects)
    subject_groups = np.array_split(unique_subjects, n_folds)

    for fold_id, held_out in enumerate(subject_groups):
        held_out_set = set(held_out.tolist())
        test_mask = np.array([s in held_out_set for s in subject])
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        X_train, X_test, mean, std = normalize_channels(X_train, X_test)

        yield dict(
            fold_id=fold_id,
            held_out_subjects=held_out.tolist(),
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )


# ---------------------------------------------------------------------------
# Unified fold iterator used by train.py (both protocols, subject-wise val)
# ---------------------------------------------------------------------------
def get_classes(dataset: str) -> np.ndarray:
    """Sorted raw label values of the whole dataset -- the one label mapping
    used for every split/fold, so class index k means the same activity in
    every fold (label space is dataset metadata, not learned -> no leakage)."""
    _, y, _, _ = _load_processed(dataset)
    return np.unique(y)


def encode_with_classes(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(classes, y)
    assert np.all(classes[idx] == y), "label not in dataset class list"
    return idx.astype(np.int64)


def pick_val_subjects(train_subjects, val_fraction: float, seed: int, min_subjects: int = 2):
    """Deterministic subject-level validation pick (>= min_subjects subjects).
    Depends only on the fold, never on the model or training seed, so every
    model sees identical train/val/test partitions."""
    subs = np.unique(train_subjects)
    n_val = max(min_subjects, int(round(val_fraction * len(subs))))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(subs, size=n_val, replace=False).tolist())


def iter_folds(dataset: str, split_mode: str, n_folds: Optional[int] = None,
               val_fraction: float = 0.2, min_val_subjects: int = 2, seed: int = SPLIT_SEED):
    """
    Yields one dict per fold (a single fold for split_mode="official"):
        fold_id (None for official), held_out_subjects, val_subjects, classes,
        X_train, y_train, X_val, y_val, X_test, y_test, subject_test
    y_* are encoded through the dataset-wide class list (get_classes).
    Validation subjects are removed from the fold's training subjects; the
    normalization statistics are fit on the remaining training windows only.
    """
    X, y, subject, official_split = _load_processed(dataset)
    classes = np.unique(y)
    y_enc = encode_with_classes(y, classes)

    if split_mode == "official":
        if official_split is not None:
            test_masks = [(None, official_split == "test")]
        else:
            rng = np.random.RandomState(seed)
            subs = np.unique(subject)
            rng.shuffle(subs)
            n_test = max(1, int(round(0.2 * len(subs))))
            test_masks = [(None, np.isin(subject, subs[:n_test]))]
    elif split_mode == "loso":
        n_folds = n_folds or LOSO_N_FOLDS[dataset]
        rng = np.random.RandomState(seed)
        subs = np.unique(subject)
        rng.shuffle(subs)
        test_masks = [(i, np.isin(subject, g)) for i, g in enumerate(np.array_split(subs, n_folds))]
    else:
        raise ValueError(split_mode)

    for fold_id, test_mask in test_masks:
        val_seed = seed + 1000 + (fold_id if fold_id is not None else 0)
        val_subjects = pick_val_subjects(subject[~test_mask], val_fraction, val_seed, min_val_subjects)
        val_mask = np.isin(subject, val_subjects) & ~test_mask
        train_mask = ~test_mask & ~val_mask

        X_train, X_val, X_test, _, _ = normalize_channels(X[train_mask], X[val_mask], X[test_mask])
        yield dict(
            fold_id=fold_id,
            held_out_subjects=sorted(np.unique(subject[test_mask]).tolist()),
            val_subjects=val_subjects,
            classes=classes,
            X_train=X_train.astype(np.float32), y_train=y_enc[train_mask],
            X_val=X_val.astype(np.float32), y_val=y_enc[val_mask],
            X_test=X_test.astype(np.float32), y_test=y_enc[test_mask],
            subject_test=subject[test_mask],
        )


# ---------------------------------------------------------------------------
# Convenience: DataLoader builder
# ---------------------------------------------------------------------------
def build_dataloaders(X_train, y_train, X_test, y_test,
                       batch_size: int = 64, num_workers: int = 0,
                       seed: Optional[int] = None):
    # seeded generator -> the shuffle order is reproducible per training seed
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(
        HARDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=False,
        generator=generator,
    )
    test_loader = DataLoader(
        HARDataset(X_test, y_test), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Smoke test -- run after preprocessing.py has populated data/processed/
    for ds in ("ucihar", "wisdm", "pamap2"):
        try:
            X_tr, y_tr, X_te, y_te = get_official_split(ds)
            print(f"[{ds}] official split -> train {X_tr.shape}, test {X_te.shape}")

            for mode in ("official", "loso"):
                for f in iter_folds(ds, mode):
                    print(f"[{ds}] {mode} fold={f['fold_id']} train={len(f['y_train'])} "
                          f"val={len(f['y_val'])} (subjects {f['val_subjects']}) "
                          f"test={len(f['y_test'])} (subjects {f['held_out_subjects']}) "
                          f"classes={len(f['classes'])}")
        except FileNotFoundError as e:
            print(f"[{ds}] skipped, processed data not found yet: {e}")