# Train/val split utilities for kfold CV.

from sklearn.model_selection import train_test_split
import numpy as np


def split_by_systematics(n_cosmologies, n_systematics, split_fraction=0.2, seed=42):
    """Split along systematics axis so all cosmologies appear in both train and val."""
    sys_idx = np.arange(n_systematics)
    train_sys, val_sys = train_test_split(sys_idx, test_size=split_fraction, random_state=seed)

    train_idx = []
    val_idx = []
    for c in range(n_cosmologies):
        for s in train_sys:
            train_idx.append(c * n_systematics + s)
        for s in val_sys:
            val_idx.append(c * n_systematics + s)

    return np.array(train_idx), np.array(val_idx)


def verify_split(train_idx, val_idx, n_cosmologies, n_systematics):
    """Sanity check that split covers all data without overlap."""
    assert len(set(train_idx) & set(val_idx)) == 0, "overlap detected"
    train_cosmo = set(train_idx // n_systematics)
    val_cosmo = set(val_idx // n_systematics)
    assert len(train_cosmo) == n_cosmologies and len(val_cosmo) == n_cosmologies
    assert len(train_idx) + len(val_idx) == n_cosmologies * n_systematics
    print(f"  train={len(train_idx)}, val={len(val_idx)}")


def create_kfold_splits_by_systematics(n_cosmologies, n_systematics, n_folds=5, seed=42):
    """Create k-fold CV splits along systematics dimension."""
    np.random.seed(seed)
    sys_idx = np.arange(n_systematics)
    np.random.shuffle(sys_idx)

    # divide systematics into folds
    fold_size = n_systematics // n_folds
    remainder = n_systematics % n_folds
    folds = []
    start = 0
    for f in range(n_folds):
        size = fold_size + (1 if f < remainder else 0)
        folds.append(sys_idx[start:start+size])
        start += size

    # build train/val splits
    splits = []
    for val_f in range(n_folds):
        val_sys = folds[val_f]
        train_sys = np.concatenate([folds[i] for i in range(n_folds) if i != val_f])

        train_idx, val_idx = [], []
        for c in range(n_cosmologies):
            for s in train_sys:
                train_idx.append(c * n_systematics + s)
            for s in val_sys:
                val_idx.append(c * n_systematics + s)
        splits.append((np.array(train_idx), np.array(val_idx)))

    print(f"\nK-Fold setup (k={n_folds}):")
    for i, (tr, va) in enumerate(splits):
        print(f"  Fold {i}: train={len(tr)}, val={len(va)}")
        verify_split(tr, va, n_cosmologies, n_systematics)

    return splits
