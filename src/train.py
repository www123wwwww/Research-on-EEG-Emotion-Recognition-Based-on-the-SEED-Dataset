"""
Training pipeline for all models.

Key design decisions for experimental rigor:
  - A held-out validation set (20% of training data) is used for early stopping.
    The test set is evaluated ONLY once, using the model that achieved the best
    validation accuracy. This prevents test-set leakage.
  - SVM uses the same normalised features as the neural models.
  - All random seeds are fixed for reproducibility.

Usage:
    python src/train.py --model svm      --split subject_dependent
    python src/train.py --model mlp      --split subject_dependent
    python src/train.py --model attn_mlp --split subject_dependent
    python src/train.py --model svm      --split loso
    python src/train.py --model mlp      --split loso
    python src/train.py --model attn_mlp --split loso

Results are saved to results/<split>_<model>_results.npy
"""

import sys
import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_subject_dependent_splits, get_loso_splits, get_flat_features
from src.models.baseline import SVMModel, MLPModel
from src.models.attention_mlp import AttentionMLP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_RATIO    = 0.2    # fraction of training data held out for validation
PATIENCE     = 20     # early-stopping patience (epochs without val improvement)
MAX_EPOCHS   = 200    # upper bound; early stopping will usually trigger first
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def _train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = VAL_RATIO):
    """Stratified train/val split: sample val_ratio from each class independently.

    SEED labels are temporally contiguous blocks, so a plain random permutation
    can concentrate an entire class in the val set. Stratification ensures each
    class is proportionally represented in both subsets.
    """
    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        n_val   = max(1, int(len(cls_idx) * val_ratio))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ──────────────────────────────────────────────
# SVM
# ──────────────────────────────────────────────

def run_svm_fold(split: dict) -> float:
    X_tr = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    model = SVMModel(C=1.0)
    model.fit(X_tr, split["y_train"])
    return model.score(X_te, split["y_test"])


# ──────────────────────────────────────────────
# Neural model training
# ──────────────────────────────────────────────

def _forward(model, X_batch, model_type):
    if model_type == "attn_mlp":
        logits, band_w, chan_w = model(X_batch)
        return logits, band_w, chan_w
    return model(X_batch), None, None


def _epoch_loss(model, loader, criterion, model_type, train: bool,
                optimizer=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits, _, _ = _forward(model, X_b, model_type)
            loss = criterion(logits, y_b)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_b)
            correct += (logits.argmax(1) == y_b).sum().item()
            total   += len(y_b)
    return total_loss / total, correct / total


def _collect_weights(model, loader, model_type):
    """Collect attention weights over the entire loader (for visualisation)."""
    model.eval()
    band_ws, chan_ws = [], []
    with torch.no_grad():
        for X_b, _ in loader:
            X_b = X_b.to(DEVICE)
            _, bw, cw = _forward(model, X_b, model_type)
            if bw is not None:
                band_ws.append(bw.cpu().numpy())
                chan_ws.append(cw.cpu().numpy())
    if band_ws:
        return np.concatenate(band_ws), np.concatenate(chan_ws)
    return None, None


def run_torch_fold(split: dict, model_type: str,
                   max_epochs: int = MAX_EPOCHS,
                   patience: int = PATIENCE):
    # Prepare tensors
    if model_type == "mlp":
        X_tr_all = get_flat_features(split["X_train"])
        X_te     = get_flat_features(split["X_test"])
    else:
        X_tr_all = split["X_train"]           # (N, 5, 62)
        X_te     = split["X_test"]

    y_tr_all = split["y_train"]
    y_te     = split["y_test"]

    # Train / validation split — test set is NOT touched during training
    X_tr, y_tr, X_val, y_val = _train_val_split(X_tr_all, y_tr_all)

    train_loader = _to_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    val_loader   = _to_loader(X_val, y_val, 256,        shuffle=False)
    test_loader  = _to_loader(X_te,  y_te,  256,        shuffle=False)

    # Model
    if model_type == "mlp":
        model = MLPModel().to(DEVICE)
    else:
        model = AttentionMLP().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = -1.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        _epoch_loss(model, train_loader, criterion, model_type, train=True, optimizer=optimizer)
        scheduler.step()

        _, val_acc = _epoch_loss(model, val_loader, criterion, model_type, train=False)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break  # early stopping

    # Restore best model, evaluate on test set ONCE
    model.load_state_dict(best_state)
    _, test_acc = _epoch_loss(model, test_loader, criterion, model_type, train=False)

    band_w, chan_w = _collect_weights(model, test_loader, model_type)
    return test_acc, band_w, chan_w


# ──────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────

def run_experiment(model_type: str, split_strategy: str, verbose: bool = True):
    if split_strategy == "subject_dependent":
        splits   = get_subject_dependent_splits(normalize=True)
        fold_key = lambda s: f"sub{s['subject']}_sess{s['session']}"
    else:
        splits   = get_loso_splits(normalize=True)
        fold_key = lambda s: f"test_sub{s['test_subject']}"

    accs, all_band_w, all_chan_w = [], [], []

    for i, split in enumerate(splits):
        label = fold_key(split)

        if model_type == "svm":
            acc      = run_svm_fold(split)
            bw = cw  = None
        else:
            acc, bw, cw = run_torch_fold(split, model_type)

        accs.append(acc)
        if bw is not None:
            all_band_w.append(bw.mean(axis=0))
            all_chan_w.append(cw.mean(axis=0))

        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:25s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> {split_strategy} | {model_type}: "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model": model_type, "split": split_strategy,
        "accs": np.array(accs), "mean_acc": mean_acc, "std_acc": std_acc,
    }
    if all_band_w:
        result["band_weights"] = np.stack(all_band_w).mean(axis=0)
        result["chan_weights"]  = np.stack(all_chan_w).mean(axis=0)

    out = os.path.join(RESULTS_DIR, f"{split_strategy}_{model_type}_results.npy")
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["svm", "mlp", "attn_mlp"], default="attn_mlp")
    parser.add_argument("--split",  choices=["subject_dependent", "loso"],
                        default="subject_dependent")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Model  : {args.model}  |  Split: {args.split}")
    print(f"Config : max_epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"val_ratio={VAL_RATIO}  lr={LR}\n")
    run_experiment(args.model, args.split, verbose=not args.quiet)


if __name__ == "__main__":
    main()
