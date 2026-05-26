"""
Cross-Session Adaptive Batch Normalization (AdaBN) + Fine-tune.

Extends the pretrain→finetune pipeline with an unsupervised domain
adaptation step between the two phases:

  Phase 1 — Pre-train (identical to train_cross_session.py)
    Data : all data from the other 2 sessions of the same subject
    Loss : CrossEntropy

  Phase 2 — AdaBN Adaptation  ← new, zero cost, zero labels
    Data : ALL target-session data (train + test, labels ignored)
    Action: reset BN running statistics and re-estimate them from target
            domain samples via cumulative moving average.
    Why  : BN running_mean / running_var encode source-session distribution
           statistics.  Replacing them with target-session statistics
           compensates for cross-session covariate shift before any
           gradient update is needed.

  Phase 3 — Fine-tune (identical to train_cross_session.py)
    Data : training portion of the target session (labeled)
    Loss : CrossEntropy, lower lr

  Test   : test portion of target session, evaluated ONCE.

Ablation flag --no_finetune skips Phase 3 so the pure AdaBN effect
(no labeled adaptation at all) can be measured separately.

Reference:
  Li et al., "Revisiting Batch Normalization For Practical Domain
  Adaptation," ICLR Workshop 2017.

Usage:
    python src/cross_session/train_cross_session_adabn.py
    python src/cross_session/train_cross_session_adabn.py --no_finetune
    python src/cross_session/train_cross_session_adabn.py --quiet
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
    _load_npz,
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

# ── hyper-parameters (kept identical to train_cross_session.py) ───────────────
BATCH_SIZE = 128
VAL_RATIO  = 0.2

LR_PRETRAIN       = 1e-3
EPOCHS_PRETRAIN   = 150
PATIENCE_PRETRAIN = 20
WD_PRETRAIN       = 1e-4

LR_FINETUNE       = 2e-4
EPOCHS_FINETUNE   = 80
PATIENCE_FINETUNE = 15
WD_FINETUNE       = 1e-4

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


def _to_unlabeled_loader(X, batch_size=256):
    ds = TensorDataset(torch.tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def _load_other_sessions(subject: int, target_session: int):
    """ALL data (train + test) from the two non-target sessions, normalized."""
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


# ── generic supervised training loop (early stopping) ────────────────────────

def _train_loop(model, train_loader, val_loader, optimizer, scheduler,
                max_epochs, patience):
    criterion  = nn.CrossEntropyLoss()
    best_val   = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(X_b), y_b).backward()
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

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_state


# ── AdaBN adaptation step ─────────────────────────────────────────────────────

def adabn_adapt(model: nn.Module, X_target: np.ndarray) -> None:
    """
    Replace all BN running statistics with target-domain statistics.

    Uses PyTorch's cumulative moving average mode (momentum=None) so a
    single pass over all target samples gives the exact batch mean and
    variance — equivalent to computing them offline and assigning directly.

    Args:
        model    : the pretrained model (modified in-place)
        X_target : all target-session samples, shape (N, 310), numpy float32
                   Labels are NOT used or required.
    """
    # 1. Reset running stats and enable cumulative moving average
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None       # CMA: running_mean ← exact sample mean

    # 2. Forward pass — train() mode so BN updates running stats
    model.train()
    loader = _to_unlabeled_loader(X_target)
    with torch.no_grad():
        for (X_b,) in loader:
            model(X_b.to(DEVICE))

    # 3. Restore standard EMA momentum so fine-tuning behaves normally
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = 0.1
    model.eval()


# ── per-fold pipeline ─────────────────────────────────────────────────────────

def run_fold(split: dict, finetune: bool = True) -> float:
    subject        = split["subject"]
    target_session = split["session"]

    X_ft = get_flat_features(split["X_train"])
    y_ft = split["y_train"]
    X_te = get_flat_features(split["X_test"])
    y_te = split["y_test"]

    X_src, y_src = _load_other_sessions(subject, target_session)

    X_src_tr, y_src_tr, X_src_val, y_src_val = _stratified_split(X_src, y_src)

    src_train_loader = _to_loader(X_src_tr,  y_src_tr,  BATCH_SIZE, shuffle=True)
    src_val_loader   = _to_loader(X_src_val, y_src_val, 256,        shuffle=False)
    test_loader      = _to_loader(X_te,      y_te,      256,        shuffle=False)

    model = MLPModel().to(DEVICE)

    # ── Phase 1: pretrain on other sessions ──────────────────────────────────
    opt_pt   = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=WD_PRETRAIN)
    sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=EPOCHS_PRETRAIN)
    best_pt  = _train_loop(model, src_train_loader, src_val_loader,
                            opt_pt, sched_pt, EPOCHS_PRETRAIN, PATIENCE_PRETRAIN)
    model.load_state_dict(best_pt)

    # ── Phase 2: AdaBN — update BN stats to target session (no labels) ───────
    # Use BOTH train and test portions so the estimate covers the full session.
    X_target_all = np.concatenate([X_ft, X_te], axis=0)
    adabn_adapt(model, X_target_all)

    # ── Phase 3 (optional): fine-tune on labeled target-session train data ────
    if finetune:
        X_ft_tr, y_ft_tr, X_ft_val, y_ft_val = _stratified_split(X_ft, y_ft)
        ft_train_loader = _to_loader(X_ft_tr,  y_ft_tr,  BATCH_SIZE, shuffle=True)
        ft_val_loader   = _to_loader(X_ft_val, y_ft_val, 256,        shuffle=False)

        opt_ft   = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WD_FINETUNE)
        sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=EPOCHS_FINETUNE)
        best_ft  = _train_loop(model, ft_train_loader, ft_val_loader,
                                opt_ft, sched_ft, EPOCHS_FINETUNE, PATIENCE_FINETUNE)
        model.load_state_dict(best_ft)

    # ── Test ──────────────────────────────────────────────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in test_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            correct += (model(X_t).argmax(1) == y_t).sum().item()
            total   += len(y_t)
    return correct / total


# ── full experiment ───────────────────────────────────────────────────────────

def run_experiment(finetune: bool = True, verbose: bool = True) -> dict:
    splits = get_subject_dependent_splits(normalize=True)
    accs   = []

    for i, split in enumerate(splits):
        label = f"sub{split['subject']}_sess{split['session']}"
        acc   = run_fold(split, finetune=finetune)
        accs.append(acc)
        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:18s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    tag = "mlp_cross_session_adabn" if finetune else "mlp_cross_session_adabn_only"
    label_str = ("MLP + Cross-Session + AdaBN + Finetune"
                 if finetune else "MLP + Cross-Session + AdaBN (no finetune)")
    print(f"\n>>> Subject-Dependent | {label_str}: "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":    tag,
        "split":    "subject_dependent",
        "finetune": finetune,
        "accs":     np.array(accs),
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
    }
    out = os.path.join(RESULTS_DIR, f"subject_dependent_{tag}_results.npy")
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-session AdaBN adaptation + fine-tune."
    )
    parser.add_argument("--no_finetune", action="store_true",
                        help="Skip fine-tuning: evaluate AdaBN effect alone.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    finetune = not args.no_finetune
    print(f"Device   : {DEVICE}")
    print(f"Strategy : cross-session pretrain → AdaBN → "
          f"{'finetune' if finetune else 'direct eval (no finetune)'}")
    print(f"Pretrain : lr={LR_PRETRAIN}  epochs={EPOCHS_PRETRAIN}  "
          f"patience={PATIENCE_PRETRAIN}")
    if finetune:
        print(f"Finetune : lr={LR_FINETUNE}  epochs={EPOCHS_FINETUNE}  "
              f"patience={PATIENCE_FINETUNE}")
    print()

    run_experiment(finetune=finetune, verbose=not args.quiet)


if __name__ == "__main__":
    main()
