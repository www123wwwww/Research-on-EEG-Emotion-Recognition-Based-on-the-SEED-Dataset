"""
Training pipeline: MLP + MMD domain adaptation (LOSO split only).

MMD aligns the feature distributions of source and target domains by
minimising the Maximum Mean Discrepancy in a kernel space, using multiple
RBF kernels at different scales.

Loss:
    total = CrossEntropy(source logits, source labels)
          + lambda_mmd * MMD²(source features, target features)

Results saved to results/domain_adaptation/  (never touches results/).

Usage:
    python src/domain_adaptation/train_mmd.py
    python src/domain_adaptation/train_mmd.py --lambda_mmd 0.5
    python src/domain_adaptation/train_mmd.py --quiet
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

from src.data_loader import get_loso_splits, get_flat_features
from src.domain_adaptation.mmd_model import MmdMLP
from src.domain_adaptation.mmd_loss import mmd_loss

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "results", "domain_adaptation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters (same as baseline and CORAL for fair comparison) ─────────
VAL_RATIO    = 0.2
PATIENCE     = 20
MAX_EPOCHS   = 200
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── utilities ─────────────────────────────────────────────────────────────────

def _train_val_split(X, y, val_ratio=VAL_RATIO):
    """Stratified split: proportional class representation in both sets."""
    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _to_unlabeled_loader(X, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── per-fold training ─────────────────────────────────────────────────────────

def run_mmd_fold(split: dict, lambda_mmd: float,
                 max_epochs: int = MAX_EPOCHS,
                 patience: int = PATIENCE) -> float:
    """
    Train MmdMLP on one LOSO fold and return test accuracy.

    The test subject's features (without labels) serve as the target
    domain for MMD alignment during training.
    """
    X_tr_all = get_flat_features(split["X_train"])
    X_te     = get_flat_features(split["X_test"])
    y_tr_all = split["y_train"]
    y_te     = split["y_test"]

    X_tr, y_tr, X_val, y_val = _train_val_split(X_tr_all, y_tr_all)

    src_loader = _to_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    tgt_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te,  y_te,  256, shuffle=False)

    model     = MmdMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = -1.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tgt_iter = iter(tgt_loader)

        for (X_src, y_src) in src_loader:
            X_src = X_src.to(DEVICE)
            y_src = y_src.to(DEVICE)

            try:
                (X_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)

            optimizer.zero_grad()

            logits_src, feat_src = model(X_src)
            _,          feat_tgt = model(X_tgt)

            loss_ce  = criterion(logits_src, y_src)
            loss_mmd = mmd_loss(feat_src, feat_tgt)
            loss     = loss_ce + lambda_mmd * loss_mmd

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation (source side only)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                correct += (logits.argmax(1) == y_v).sum().item()
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

    # Evaluate on the test subject once
    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total   += len(y_t)
    return correct / total


# ── full experiment ───────────────────────────────────────────────────────────

def run_mmd_experiment(lambda_mmd: float = 1.0, verbose: bool = True) -> dict:
    splits = get_loso_splits(normalize=True)
    accs   = []

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"
        acc   = run_mmd_fold(split, lambda_mmd=lambda_mmd)
        accs.append(acc)
        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:20s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> LOSO | MLP+MMD (λ={lambda_mmd}): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":      "mlp_mmd",
        "split":      "loso",
        "lambda_mmd": lambda_mmd,
        "accs":       np.array(accs),
        "mean_acc":   mean_acc,
        "std_acc":    std_acc,
    }

    out = os.path.join(RESULTS_DIR, "loso_mlp_mmd_results.npy")
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP + MMD domain adaptation on SEED (LOSO split)."
    )
    parser.add_argument(
        "--lambda_mmd", type=float, default=1.0,
        help="Weight of the MMD loss term (default: 1.0)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device     : {DEVICE}")
    print(f"Model      : MLP + MMD  |  Split: LOSO")
    print(f"lambda_mmd : {args.lambda_mmd}")
    print(f"Config     : max_epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"val_ratio={VAL_RATIO}  lr={LR}\n")

    run_mmd_experiment(
        lambda_mmd=args.lambda_mmd,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
