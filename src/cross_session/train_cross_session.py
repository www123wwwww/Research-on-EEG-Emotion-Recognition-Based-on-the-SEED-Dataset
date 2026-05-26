"""
Cross-session pre-training + fine-tuning for Subject-Dependent evaluation.

Strategy (for each fold = one (subject, session) pair):

  Phase 1 — Pre-train
    Data   : ALL data (train + test merged) from the OTHER 2 sessions of the
             SAME subject, each session normalized with its own train statistics.
    Goal   : give the model a strong starting point that already understands
             this specific person's EEG patterns.

  Phase 2 — Fine-tune
    Data   : training portion of the TARGET session (same split used in
             existing subject_dependent results).
    Goal   : adapt the pre-trained model to the target session's slight
             distributional shift, using a lower learning rate to preserve
             what was learned.

  Test    : test portion of the TARGET session, evaluated ONCE.

Why this beats SVM:
  SVM must learn from scratch on ~1600 target-session samples.
  Our model starts from weights already tuned on ~3200 same-subject samples,
  so fine-tuning needs far fewer steps to reach a good solution.

Results saved to results/cross_session/  — never touches results/.

Usage:
    python src/cross_session/train_cross_session.py
    python src/cross_session/train_cross_session.py --quiet
"""

import sys
import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from src.data_loader import (
    get_subject_dependent_splits, get_flat_features,
    _load_npz, _normalize,
    AVAILABLE_SUBJECTS, SESSIONS,
)
from src.models.baseline import MLPModel

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "results", "cross_session")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 128
VAL_RATIO  = 0.2

# Pre-train: more epochs, standard LR — model learns from scratch on other sessions
LR_PRETRAIN      = 1e-3
EPOCHS_PRETRAIN  = 150
PATIENCE_PRETRAIN = 20
WD_PRETRAIN      = 1e-4

# Fine-tune: fewer epochs, lower LR — preserve pre-trained weights, only adapt
LR_FINETUNE      = 2e-4
EPOCHS_FINETUNE  = 80
PATIENCE_FINETUNE = 15
WD_FINETUNE      = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── utilities ─────────────────────────────────────────────────────────────────

def _stratified_split(X, y, val_ratio=VAL_RATIO):
    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _load_other_sessions(subject: int, target_session: int):
    """
    Load and return ALL data (train + test merged) from the two sessions of
    `subject` that are NOT `target_session`.

    Each session is normalized independently using its own training statistics,
    which is consistent with how get_subject_dependent_splits works.
    """
    X_parts, y_parts = [], []
    for sess in SESSIONS:
        if sess == target_session:
            continue
        X_tr, y_tr, X_te, y_te = _load_npz(subject, sess)
        X_tr_flat = get_flat_features(X_tr)
        X_te_flat = get_flat_features(X_te)
        # Normalize this session with its own train statistics
        mu    = X_tr_flat.mean(axis=0)
        sigma = X_tr_flat.std(axis=0) + 1e-8
        X_tr_n = (X_tr_flat - mu) / sigma
        X_te_n = (X_te_flat - mu) / sigma
        X_parts.append(X_tr_n)
        X_parts.append(X_te_n)
        y_parts.append(y_tr)
        y_parts.append(y_te)
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ── generic training loop ─────────────────────────────────────────────────────

def _train_loop(model, train_loader, val_loader, optimizer, scheduler,
                criterion, max_epochs, patience):
    """
    Train with early stopping. Returns best model state_dict.
    Works for both pre-training and fine-tuning phases.
    """
    best_val_acc   = -1.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                correct += (model(X_v).argmax(1) == y_v).sum().item()
                total   += len(y_v)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    return best_state


# ── per-fold pipeline ─────────────────────────────────────────────────────────

def run_cross_session_fold(split: dict) -> float:
    """
    Full pre-train → fine-tune → test pipeline for one (subject, session) fold.

    `split` is one entry from get_subject_dependent_splits(), which already
    contains the normalized target-session train/test data.
    """
    subject        = split["subject"]
    target_session = split["session"]

    # ── Target session data (already normalized, same as existing Subject-Dep.) ──
    X_ft = get_flat_features(split["X_train"])   # fine-tune training data
    y_ft = split["y_train"]
    X_te = get_flat_features(split["X_test"])    # held-out test data (never touched)
    y_te = split["y_test"]

    # ── Pre-training data: other sessions of the same subject ────────────────
    X_pt, y_pt = _load_other_sessions(subject, target_session)

    # ── Data loaders ──────────────────────────────────────────────────────────
    X_pt_tr, y_pt_tr, X_pt_val, y_pt_val = _stratified_split(X_pt, y_pt)
    X_ft_tr, y_ft_tr, X_ft_val, y_ft_val = _stratified_split(X_ft, y_ft)

    pt_train_loader = _to_loader(X_pt_tr,  y_pt_tr,  BATCH_SIZE, shuffle=True)
    pt_val_loader   = _to_loader(X_pt_val, y_pt_val, 256,        shuffle=False)
    ft_train_loader = _to_loader(X_ft_tr,  y_ft_tr,  BATCH_SIZE, shuffle=True)
    ft_val_loader   = _to_loader(X_ft_val, y_ft_val, 256,        shuffle=False)
    test_loader     = _to_loader(X_te,     y_te,      256,        shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model     = MLPModel().to(DEVICE)

    # ── Phase 1: Pre-train on the other two sessions ──────────────────────────
    optimizer_pt = torch.optim.Adam(
        model.parameters(), lr=LR_PRETRAIN, weight_decay=WD_PRETRAIN
    )
    scheduler_pt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pt, T_max=EPOCHS_PRETRAIN
    )
    best_pt = _train_loop(
        model, pt_train_loader, pt_val_loader,
        optimizer_pt, scheduler_pt, criterion,
        EPOCHS_PRETRAIN, PATIENCE_PRETRAIN,
    )
    model.load_state_dict(best_pt)

    # ── Phase 2: Fine-tune on the target session training data ────────────────
    optimizer_ft = torch.optim.Adam(
        model.parameters(), lr=LR_FINETUNE, weight_decay=WD_FINETUNE
    )
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=EPOCHS_FINETUNE
    )
    best_ft = _train_loop(
        model, ft_train_loader, ft_val_loader,
        optimizer_ft, scheduler_ft, criterion,
        EPOCHS_FINETUNE, PATIENCE_FINETUNE,
    )
    model.load_state_dict(best_ft)

    # ── Test: evaluated ONCE ──────────────────────────────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in test_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            correct += (model(X_t).argmax(1) == y_t).sum().item()
            total   += len(y_t)
    return correct / total


# ── full experiment ───────────────────────────────────────────────────────────

def run_experiment(verbose: bool = True) -> dict:
    splits = get_subject_dependent_splits(normalize=True)
    accs   = []

    for i, split in enumerate(splits):
        label = f"sub{split['subject']}_sess{split['session']}"
        acc   = run_cross_session_fold(split)
        accs.append(acc)
        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:18s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> Subject-Dependent | MLP (cross-session pretrain): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":    "mlp_cross_session",
        "split":    "subject_dependent",
        "accs":     np.array(accs),
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
    }
    out = os.path.join(RESULTS_DIR, "subject_dependent_mlp_cross_session_results.npy")
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-session pre-training + fine-tuning (Subject-Dependent)."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device   : {DEVICE}")
    print(f"Strategy : cross-session pre-train → fine-tune")
    print(f"Pretrain : lr={LR_PRETRAIN}  epochs={EPOCHS_PRETRAIN}  "
          f"patience={PATIENCE_PRETRAIN}")
    print(f"Finetune : lr={LR_FINETUNE}  epochs={EPOCHS_FINETUNE}  "
          f"patience={PATIENCE_FINETUNE}\n")

    run_experiment(verbose=not args.quiet)


if __name__ == "__main__":
    main()
