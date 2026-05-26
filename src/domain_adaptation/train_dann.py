"""
Training pipeline: MLP + DANN domain adaptation (LOSO split only).

DANN loss breakdown
───────────────────
  L_total = L_label(source) + λ · [L_domain(source) + L_domain(target)]

  L_label  : CrossEntropy on emotion labels  (source domain only)
  L_domain : CrossEntropy on domain labels   (source=0, target=1, both domains)

  The GRL inside the model automatically turns L_domain into an adversarial
  signal for the feature extractor — no separate optimizer tricks needed.

α scheduling
────────────
  α grows from 0 to 1 following the sigmoid schedule from the original paper:
      α(p) = 2 / (1 + exp(−10 · p)) − 1,   p = epoch / max_epochs

  Starting at α≈0 prevents the domain adversary from destabilising the
  label classifier before it has had a chance to train.

Results saved to results/domain_adaptation/  — never touches results/.

Usage:
    python src/domain_adaptation/train_dann.py
    python src/domain_adaptation/train_dann.py --lambda_dann 0.5
    python src/domain_adaptation/train_dann.py --quiet
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
from src.domain_adaptation.dann_model import DannMLP

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "results", "domain_adaptation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters (same as CORAL/MMD for fair comparison) ─────────────────
VAL_RATIO    = 0.2
PATIENCE     = 20
MAX_EPOCHS   = 200
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── utilities ─────────────────────────────────────────────────────────────────

def _train_val_split(X, y, val_ratio=VAL_RATIO):
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
    ds = TensorDataset(torch.tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _get_alpha(epoch: int, max_epochs: int) -> float:
    """
    Sigmoid schedule for GRL strength α.
    α(0)=0.0, α(max_epochs/2)≈0.98, α(max_epochs)≈1.0
    """
    p = min(epoch / max_epochs, 1.0)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ── per-fold training ─────────────────────────────────────────────────────────

def run_dann_fold(split: dict, lambda_dann: float,
                  max_epochs: int = MAX_EPOCHS,
                  patience: int = PATIENCE) -> float:
    """
    Train DannMLP on one LOSO fold and return test accuracy.

    Domain labels:
      source (training subjects) = 0
      target (test subject)      = 1
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

    model          = DannMLP().to(DEVICE)
    optimizer      = torch.optim.Adam(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=max_epochs)
    label_crit     = nn.CrossEntropyLoss()
    domain_crit    = nn.CrossEntropyLoss()

    best_val_acc   = -1.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        alpha = _get_alpha(epoch, max_epochs)
        model.train()
        tgt_iter = iter(tgt_loader)

        for (X_src, y_src) in src_loader:
            X_src = X_src.to(DEVICE)
            y_src = y_src.to(DEVICE)

            # Fetch one target batch (cycle if exhausted)
            try:
                (X_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)

            n_src = X_src.size(0)
            n_tgt = X_tgt.size(0)
            dom_src = torch.zeros(n_src, dtype=torch.long, device=DEVICE)  # source = 0
            dom_tgt = torch.ones (n_tgt, dtype=torch.long, device=DEVICE)  # target = 1

            optimizer.zero_grad()

            # Source: label loss + domain loss
            label_logits, dom_logits_src, _ = model(X_src, alpha)
            # Target: domain loss only (no labels available)
            _,            dom_logits_tgt, _ = model(X_tgt, alpha)

            loss_label  = label_crit(label_logits, y_src)
            loss_domain = domain_crit(dom_logits_src, dom_src) + \
                          domain_crit(dom_logits_tgt, dom_tgt)
            loss        = loss_label + lambda_dann * loss_domain

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation: use α=0 so GRL does not distort label predictions
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                label_logits, _, _ = model(X_v, alpha=0.0)
                correct += (label_logits.argmax(1) == y_v).sum().item()
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

    # Evaluate best checkpoint on the test subject once
    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            label_logits, _, _ = model(X_t, alpha=0.0)
            correct += (label_logits.argmax(1) == y_t).sum().item()
            total   += len(y_t)
    return correct / total


# ── full experiment ───────────────────────────────────────────────────────────

def run_dann_experiment(lambda_dann: float = 1.0, verbose: bool = True) -> dict:
    splits = get_loso_splits(normalize=True)
    accs   = []

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"
        acc   = run_dann_fold(split, lambda_dann=lambda_dann)
        accs.append(acc)
        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:20s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> LOSO | MLP+DANN (λ={lambda_dann}): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":       "mlp_dann",
        "split":       "loso",
        "lambda_dann": lambda_dann,
        "accs":        np.array(accs),
        "mean_acc":    mean_acc,
        "std_acc":     std_acc,
    }
    out = os.path.join(RESULTS_DIR, "loso_mlp_dann_results.npy")
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP + DANN domain adaptation on SEED (LOSO split)."
    )
    parser.add_argument("--lambda_dann", type=float, default=1.0,
                        help="Weight of the domain adversarial loss (default: 1.0)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device      : {DEVICE}")
    print(f"Model       : MLP + DANN  |  Split: LOSO")
    print(f"lambda_dann : {args.lambda_dann}")
    print(f"Config      : max_epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"val_ratio={VAL_RATIO}  lr={LR}")
    print(f"α schedule  : sigmoid  0 → 1  over {MAX_EPOCHS} epochs\n")

    run_dann_experiment(lambda_dann=args.lambda_dann, verbose=not args.quiet)


if __name__ == "__main__":
    main()
