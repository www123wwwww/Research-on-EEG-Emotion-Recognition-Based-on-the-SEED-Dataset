"""
Training pipeline: MLP + CORAL domain adaptation (LOSO split only).

Based on src/domain_adaptation/train_coral.py. Only changes:
  - Results saved to testresults/ instead of results/domain_adaptation/
  - Results saved as JSON instead of npy
  - Per-epoch train_loss / val_acc printed and logged

Usage:
    python testsrc/domain_adaptation/train_coral.py
    python testsrc/domain_adaptation/train_coral.py --lambda_coral 0.5
    python testsrc/domain_adaptation/train_coral.py --quiet
"""

import sys
import os
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # project root
sys.path.insert(0, _ROOT)

from src.data_loader import get_loso_splits, get_flat_features
from src.domain_adaptation.coral_model import CoralMLP
from src.domain_adaptation.coral_loss import coral_loss

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── changed: results dir ─────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "testresults")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters (same as baseline train.py for fair comparison) ──────────
VAL_RATIO    = 0.2
PATIENCE     = 999999       # disabled: always run fixed epochs
MAX_EPOCHS   = 50
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
    """Loader for the target domain — labels are not needed."""
    ds = TensorDataset(torch.tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── changed: save as JSON ─────────────────────────────────────────────────────
def _save_json(result, filename):
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"    Saved → {path}")


# ── per-fold training ─────────────────────────────────────────────────────────

def run_coral_fold(split: dict, lambda_coral: float,
                   max_epochs: int = MAX_EPOCHS,
                   patience: int = PATIENCE):
    """
    Train CoralMLP on one LOSO fold.

    Returns (test_acc, epoch_logs).
    """
    X_tr_all = get_flat_features(split["X_train"])   # (N_src, 310)
    X_te     = get_flat_features(split["X_test"])    # (N_tgt, 310)
    y_tr_all = split["y_train"]
    y_te     = split["y_test"]

    # Held-out validation from the source (for early stopping only)
    X_tr, y_tr, X_val, y_val = _train_val_split(X_tr_all, y_tr_all)

    src_loader = _to_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    tgt_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te,  y_te,  256, shuffle=False)

    model     = CoralMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = -1.0
    best_state     = None
    patience_count = 0

    # ── changed: collect per-epoch log ─────────────────────────────────────────
    epoch_logs = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        tgt_iter = iter(tgt_loader)
        epoch_loss_sum = 0.0
        epoch_loss_n   = 0
        epoch_correct  = 0

        for (X_src, y_src) in src_loader:
            X_src = X_src.to(DEVICE)
            y_src = y_src.to(DEVICE)

            # Fetch one target batch (cycle if the target set is exhausted)
            try:
                (X_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)

            optimizer.zero_grad()

            logits_src, feat_src = model(X_src)
            _,          feat_tgt = model(X_tgt)

            loss_ce    = criterion(logits_src, y_src)
            loss_coral = coral_loss(feat_src, feat_tgt)
            loss       = loss_ce + lambda_coral * loss_coral

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * X_src.size(0)
            epoch_loss_n   += X_src.size(0)
            epoch_correct  += (logits_src.argmax(1) == y_src).sum().item()

        scheduler.step()

        train_loss = epoch_loss_sum / epoch_loss_n
        train_acc  = epoch_correct / epoch_loss_n

        # Validation (source side only — target labels not used)
        model.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                val_loss_sum += criterion(logits, y_v).item() * len(y_v)
                correct += (logits.argmax(1) == y_v).sum().item()
                total   += len(y_v)
        val_loss = val_loss_sum / total
        val_acc  = correct / total

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc":  float(train_acc),
            "val_loss":   float(val_loss),
            "val_acc":    float(val_acc),
        })

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    # Evaluate best model on the test subject (once)
    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total   += len(y_t)
    test_acc = correct / total
    return test_acc, epoch_logs


# ── full experiment ───────────────────────────────────────────────────────────

def run_coral_experiment(lambda_coral: float = 1.0, verbose: bool = True) -> dict:
    """
    Run LOSO cross-validation with MLP + CORAL and save results.

    Args:
        lambda_coral : weight of the CORAL loss term
        verbose      : print per-fold progress
    """
    splits = get_loso_splits(normalize=True)
    accs   = []
    # ── changed: collect all fold logs ─────────────────────────────────────────
    all_fold_logs = []

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"
        acc, fold_logs = run_coral_fold(split, lambda_coral=lambda_coral)
        accs.append(acc)

        # ── changed: print per-epoch log ────────────────────────────────────────
        if verbose:
            print(f"\n  [{i+1:2d}/{len(splits)}] {label:20s}")
            print(f"  {'Epoch':>5s}  {'Train_Loss':>10s}  {'Train_Acc':>9s}  {'Val_Loss':>8s}  {'Val_Acc':>8s}")
            print(f"  {'-----':>5s}  {'----------':>10s}  {'---------':>9s}  {'--------':>8s}  {'--------':>8s}")
            for entry in fold_logs:
                print(f"  {entry['epoch']:5d}  {entry['train_loss']:10.4f}  {entry['train_acc']*100:8.2f}%  {entry['val_loss']:8.4f}  {entry['val_acc']*100:7.2f}%")
            print(f"  Test acc = {acc*100:.2f}%")

        all_fold_logs.append({"fold_label": label, "epochs": fold_logs, "test_acc": float(acc)})

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> LOSO | MLP+CORAL (λ={lambda_coral}): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":        "mlp_coral",
        "split":        "loso",
        "lambda_coral": lambda_coral,
        "accs":         np.array(accs),
        "mean_acc":     mean_acc,
        "std_acc":      std_acc,
    }

    # ── changed: save as JSON ─────────────────────────────────────────────────
    _save_json(result, "loso_mlp_coral_results.json")

    # ── changed: save epoch log as txt ─────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "loso_mlp_coral_epoch_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for fold in all_fold_logs:
            f.write(f"=== {fold['fold_label']}  (test_acc={fold['test_acc']*100:.2f}%) ===\n")
            f.write(f"{'Epoch':>5s}  {'Train_Loss':>10s}  {'Train_Acc':>9s}  {'Val_Loss':>8s}  {'Val_Acc':>8s}\n")
            for entry in fold["epochs"]:
                f.write(f"{entry['epoch']:5d}  {entry['train_loss']:10.4f}  {entry['train_acc']*100:8.2f}%  {entry['val_loss']:8.4f}  {entry['val_acc']*100:7.2f}%\n")
            f.write("\n")
    print(f"    Epoch log → {log_path}")

    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP + CORAL domain adaptation on SEED (LOSO split)."
    )
    parser.add_argument(
        "--lambda_coral", type=float, default=1.0,
        help="Weight of the CORAL loss term (default: 1.0)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device       : {DEVICE}")
    print(f"Model        : MLP + CORAL  |  Split: LOSO")
    print(f"lambda_coral : {args.lambda_coral}")
    print(f"Config       : max_epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"val_ratio={VAL_RATIO}  lr={LR}\n")

    run_coral_experiment(
        lambda_coral=args.lambda_coral,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
