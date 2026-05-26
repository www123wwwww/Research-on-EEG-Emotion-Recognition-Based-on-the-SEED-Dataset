"""
Cross-Session Domain Adaptation: pretrain with CORAL alignment + fine-tune.

Extends the pretrain→finetune pipeline (train_cross_session.py) by adding
a CORAL loss during Phase 1 to explicitly align the feature distributions
of the source sessions with the target session.

Setup (per fold = one (subject, session) pair):

  Phase 1 — Domain-Adaptive Pre-train
    Source data  : ALL data from the OTHER 2 sessions of the same subject
                   (labeled, used for CrossEntropy)
    Target data  : test portion of the TARGET session (unlabeled, used only
                   for CORAL alignment — labels never seen by the model)
    Loss = CrossEntropy(source) + λ_coral × CORAL(F(X_src), F(X_tgt))

  Phase 2 — Fine-tune  (identical to train_cross_session.py)
    Data   : training portion of the target session (labeled)
    Loss   : CrossEntropy only, lower lr to preserve pretrained weights

  Test    : test portion of target session, evaluated ONCE.

Why this improves over plain pretrain+finetune:
  Even within the same subject, EEG distributions shift across recording
  sessions (different days, electrode impedance, fatigue levels).  The
  plain pretrain ignores this shift.  CORAL aligns covariance matrices
  of source and target features so the pretrained representation is
  already session-invariant before fine-tuning begins.

Reference:
  Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain
  Adaptation", ECCV Workshop 2016.

  Li et al., "EEG-Based Emotion Recognition Using Deep Learning with
  Local and Global Average Pooling", 2020 — shows cross-session DA
  within-subject outperforms naive cross-session transfer.

Usage:
    python src/cross_session/train_cross_session_coral.py
    python src/cross_session/train_cross_session_coral.py --lambda_coral 0.5
    python src/cross_session/train_cross_session_coral.py --quiet
"""

import sys
import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.domain_adaptation.coral_model import CoralMLP
from src.domain_adaptation.coral_loss  import coral_loss

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

# Pre-train (domain-adaptive)
LR_PRETRAIN       = 1e-3
EPOCHS_PRETRAIN   = 150
PATIENCE_PRETRAIN = 20
WD_PRETRAIN       = 1e-4

# Fine-tune
LR_FINETUNE       = 2e-4
EPOCHS_FINETUNE   = 80
PATIENCE_FINETUNE = 15
WD_FINETUNE       = 1e-4

# Default CORAL regularisation weight
LAMBDA_CORAL_DEFAULT = 1.0

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


def _to_unlabeled_loader(X, batch_size, shuffle):
    """Loader without labels — used for the unlabeled target session data."""
    ds = TensorDataset(torch.tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _load_other_sessions(subject: int, target_session: int):
    """
    Load ALL data (train + test) from the two non-target sessions of `subject`.
    Each session is normalized with its own training statistics.
    """
    X_parts, y_parts = [], []
    for sess in SESSIONS:
        if sess == target_session:
            continue
        X_tr, y_tr, X_te, y_te = _load_npz(subject, sess)
        X_tr_flat = get_flat_features(X_tr)
        X_te_flat = get_flat_features(X_te)
        mu    = X_tr_flat.mean(axis=0)
        sigma = X_tr_flat.std(axis=0) + 1e-8
        X_parts.extend([(X_tr_flat - mu) / sigma, (X_te_flat - mu) / sigma])
        y_parts.extend([y_tr, y_te])
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ── training loops ────────────────────────────────────────────────────────────

def _pretrain_loop_coral(model, src_train_loader, src_val_loader,
                         tgt_unlabeled_loader, optimizer, scheduler,
                         max_epochs, patience, lambda_coral):
    """
    Phase 1: pretrain with CrossEntropy on source data + CORAL on target data.

    The target loader is cycled infinitely so its length never limits the
    number of source batches.
    """
    criterion  = nn.CrossEntropyLoss()
    best_val   = -1.0
    best_state = None
    no_improve = 0

    # Build an infinite iterator over target unlabeled batches
    def _cycle(loader):
        while True:
            for batch in loader:
                yield batch

    tgt_iter = _cycle(tgt_unlabeled_loader)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for (X_src, y_src) in src_train_loader:
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            # Fetch a target batch (no labels)
            (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)

            optimizer.zero_grad()

            # Forward — CoralMLP returns (logits, features)
            logits_src, feats_src = model(X_src)
            _,           feats_tgt = model(X_tgt)

            loss_cls   = criterion(logits_src, y_src)
            loss_coral = coral_loss(feats_src, feats_tgt)
            loss       = loss_cls + lambda_coral * loss_coral

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation on source val set (classification accuracy)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in src_val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                correct += (logits.argmax(1) == y_v).sum().item()
                total   += len(y_v)
        val_acc = correct / total

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_state


def _finetune_loop(model, ft_train_loader, ft_val_loader,
                   optimizer, scheduler, max_epochs, patience):
    """
    Phase 2: standard fine-tuning with CrossEntropy only.
    CoralMLP forward returns (logits, features); only logits are used here.
    """
    criterion  = nn.CrossEntropyLoss()
    best_val   = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in ft_train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in ft_val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                correct += (logits.argmax(1) == y_v).sum().item()
                total   += len(y_v)
        val_acc = correct / total

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_state


# ── per-fold pipeline ─────────────────────────────────────────────────────────

def run_fold(split: dict, lambda_coral: float) -> float:
    subject        = split["subject"]
    target_session = split["session"]

    # Target session: labeled train portion + unlabeled test portion
    X_ft = get_flat_features(split["X_train"])
    y_ft = split["y_train"]
    X_te = get_flat_features(split["X_test"])
    y_te = split["y_test"]

    # Source: all data from the other 2 sessions (labeled)
    X_src, y_src = _load_other_sessions(subject, target_session)

    # Splits
    X_src_tr, y_src_tr, X_src_val, y_src_val = _stratified_split(X_src, y_src)
    X_ft_tr,  y_ft_tr,  X_ft_val,  y_ft_val  = _stratified_split(X_ft,  y_ft)

    # Loaders
    src_train_loader = _to_loader(X_src_tr,  y_src_tr,  BATCH_SIZE, shuffle=True)
    src_val_loader   = _to_loader(X_src_val, y_src_val, 256,        shuffle=False)
    ft_train_loader  = _to_loader(X_ft_tr,   y_ft_tr,   BATCH_SIZE, shuffle=True)
    ft_val_loader    = _to_loader(X_ft_val,  y_ft_val,  256,        shuffle=False)
    test_loader      = _to_loader(X_te,      y_te,      256,        shuffle=False)

    # X_te is the unlabeled target data used for CORAL alignment in Phase 1.
    # Labels are never passed to the model during pretraining.
    tgt_unlabeled_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)

    model = CoralMLP().to(DEVICE)

    # ── Phase 1: domain-adaptive pretraining ──────────────────────────────────
    opt_pt  = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=WD_PRETRAIN)
    sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=EPOCHS_PRETRAIN)
    best_pt  = _pretrain_loop_coral(
        model, src_train_loader, src_val_loader,
        tgt_unlabeled_loader, opt_pt, sched_pt,
        EPOCHS_PRETRAIN, PATIENCE_PRETRAIN, lambda_coral,
    )
    model.load_state_dict(best_pt)

    # ── Phase 2: fine-tune on target session ─────────────────────────────────
    opt_ft   = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WD_FINETUNE)
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=EPOCHS_FINETUNE)
    best_ft  = _finetune_loop(
        model, ft_train_loader, ft_val_loader,
        opt_ft, sched_ft,
        EPOCHS_FINETUNE, PATIENCE_FINETUNE,
    )
    model.load_state_dict(best_ft)

    # ── Test ──────────────────────────────────────────────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in test_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total   += len(y_t)
    return correct / total


# ── full experiment ───────────────────────────────────────────────────────────

def run_experiment(lambda_coral: float, verbose: bool = True) -> dict:
    splits = get_subject_dependent_splits(normalize=True)
    accs   = []

    for i, split in enumerate(splits):
        label = f"sub{split['subject']}_sess{split['session']}"
        acc   = run_fold(split, lambda_coral)
        accs.append(acc)
        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:18s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> Subject-Dependent | MLP + Cross-Session CORAL (λ={lambda_coral}): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":        "mlp_cross_session_coral",
        "split":        "subject_dependent",
        "lambda_coral": lambda_coral,
        "accs":         np.array(accs),
        "mean_acc":     mean_acc,
        "std_acc":      std_acc,
    }
    out = os.path.join(
        RESULTS_DIR,
        f"subject_dependent_mlp_cross_session_coral_results.npy"
    )
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-session pretrain with CORAL alignment + fine-tune."
    )
    parser.add_argument(
        "--lambda_coral", type=float, default=LAMBDA_CORAL_DEFAULT,
        help=f"CORAL regularisation weight (default: {LAMBDA_CORAL_DEFAULT})",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device       : {DEVICE}")
    print(f"Strategy     : cross-session pretrain (+ CORAL) → fine-tune")
    print(f"lambda_coral : {args.lambda_coral}")
    print(f"Pretrain     : lr={LR_PRETRAIN}  epochs={EPOCHS_PRETRAIN}  "
          f"patience={PATIENCE_PRETRAIN}")
    print(f"Finetune     : lr={LR_FINETUNE}  epochs={EPOCHS_FINETUNE}  "
          f"patience={PATIENCE_FINETUNE}\n")

    run_experiment(lambda_coral=args.lambda_coral, verbose=not args.quiet)


if __name__ == "__main__":
    main()
