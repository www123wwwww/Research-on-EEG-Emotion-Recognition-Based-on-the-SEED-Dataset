"""
Member A: Data loading and splitting for SEED EEG dataset.

Dataset structure:
  EEG/  {subject}_{session}.npz  (subject: 1-14 minus 6,7; session: 1-3)
  Each file: train_data/test_data (dict with 5 bands, each (N, 62))
             train_label/test_label  (N,)  values: 0=negative, 1=neutral, 2=positive

Two split strategies:
  1. subject_dependent  — use the pre-existing 9-trial/6-trial split per file
  2. loso               — Leave-One-Subject-Out across all 12 subjects
"""

import os
import pickle
import numpy as np
from pathlib import Path

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BANDS)       # 5
N_CHANNELS = 62
FEATURE_DIM = N_BANDS * N_CHANNELS  # 310

EEG_DIR = Path(__file__).parent.parent / "dataset" / "dataset" / "EEG"

# subjects with available data
AVAILABLE_SUBJECTS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
SESSIONS = [1, 2, 3]


def _load_npz(subject: int, session: int):
    """Return (X_train, y_train, X_test, y_test) for one file.

    X shape: (N, 5, 62)  — bands × channels, not yet flattened.
    y shape: (N,)  int64
    """
    path = EEG_DIR / f"{subject}_{session}.npz"
    npz = np.load(path, allow_pickle=True)
    train_dict = pickle.loads(npz["train_data"])
    test_dict = pickle.loads(npz["test_data"])

    X_train = np.stack([train_dict[b] for b in BANDS], axis=1).astype(np.float32)
    X_test = np.stack([test_dict[b] for b in BANDS], axis=1).astype(np.float32)
    y_train = npz["train_label"].astype(np.int64)
    y_test = npz["test_label"].astype(np.int64)

    return X_train, y_train, X_test, y_test


def _normalize(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score normalise using training set statistics. Operates on flat or (N,5,62)."""
    shape = X_train.shape
    flat_train = X_train.reshape(shape[0], -1)
    flat_test = X_test.reshape(X_test.shape[0], -1)

    mu = flat_train.mean(axis=0)
    sigma = flat_train.std(axis=0) + 1e-8

    flat_train = (flat_train - mu) / sigma
    flat_test = (flat_test - mu) / sigma

    return flat_train.reshape(shape), flat_test.reshape(X_test.shape)


# ──────────────────────────────────────────────
# Strategy 1: subject-dependent (intra-subject)
# ──────────────────────────────────────────────

def get_subject_dependent_splits(normalize: bool = True):
    """
    Return a list of dicts, one per (subject, session) file.
    Each dict: {subject, session, X_train, y_train, X_test, y_test}
    X shape: (N, 5, 62)
    """
    splits = []
    for subj in AVAILABLE_SUBJECTS:
        for sess in SESSIONS:
            X_tr, y_tr, X_te, y_te = _load_npz(subj, sess)
            if normalize:
                X_tr, X_te = _normalize(X_tr, X_te)
            splits.append({
                "subject": subj,
                "session": sess,
                "X_train": X_tr,
                "y_train": y_tr,
                "X_test": X_te,
                "y_test": y_te,
            })
    return splits


# ──────────────────────────────────────────────
# Strategy 2: cross-subject LOSO
# ──────────────────────────────────────────────

def _load_subject_all_sessions(subject: int):
    """Merge train+test across all 3 sessions for one subject."""
    X_parts, y_parts = [], []
    for sess in SESSIONS:
        X_tr, y_tr, X_te, y_te = _load_npz(subject, sess)
        X_parts.extend([X_tr, X_te])
        y_parts.extend([y_tr, y_te])
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def get_loso_splits(normalize: bool = True):
    """
    Leave-One-Subject-Out: 12 folds.
    Each dict: {test_subject, X_train, y_train, X_test, y_test}
    X shape: (N, 5, 62)
    """
    all_X = {}
    all_y = {}
    for subj in AVAILABLE_SUBJECTS:
        all_X[subj], all_y[subj] = _load_subject_all_sessions(subj)

    splits = []
    for test_subj in AVAILABLE_SUBJECTS:
        X_te = all_X[test_subj]
        y_te = all_y[test_subj]

        train_parts_X = [all_X[s] for s in AVAILABLE_SUBJECTS if s != test_subj]
        train_parts_y = [all_y[s] for s in AVAILABLE_SUBJECTS if s != test_subj]
        X_tr = np.concatenate(train_parts_X, axis=0)
        y_tr = np.concatenate(train_parts_y, axis=0)

        if normalize:
            X_tr, X_te = _normalize(X_tr, X_te)

        splits.append({
            "test_subject": test_subj,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
        })
    return splits


def get_flat_features(X: np.ndarray) -> np.ndarray:
    """Flatten (N, 5, 62) → (N, 310) for SVM/flat-MLP."""
    return X.reshape(X.shape[0], -1)


def print_dataset_summary():
    splits = get_subject_dependent_splits(normalize=False)
    print(f"Subject-dependent: {len(splits)} splits")
    s = splits[0]
    print(f"  Example subject={s['subject']} session={s['session']}")
    print(f"  X_train: {s['X_train'].shape}  y_train: {s['y_train'].shape}")
    print(f"  X_test:  {s['X_test'].shape}   y_test:  {s['y_test'].shape}")
    print(f"  Label distribution train: {np.bincount(s['y_train'])}")

    loso = get_loso_splits(normalize=False)
    print(f"\nLOSO: {len(loso)} folds")
    s = loso[0]
    print(f"  Example test_subject={s['test_subject']}")
    print(f"  X_train: {s['X_train'].shape}  X_test: {s['X_test'].shape}")


if __name__ == "__main__":
    print_dataset_summary()
