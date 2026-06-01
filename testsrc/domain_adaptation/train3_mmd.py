"""
Training pipeline: MLP + MMD weighted multi-source domain adaptation (LOSO).

Based on src/domain_adaptation/train_mmd.py. Key changes:
  1. Per-subject z-score normalization before any merging.
  2. Each source subject is treated as a SEPARATE domain — not merged into
     one heterogeneous source.
  3. Per-source MMD distance to target is pre-computed on raw features.
  4. MMD alignment loss is weighted: closer sources contribute more.
       weight_s ∝ softmax(-d_s / τ)
     where d_s = MMD(source_s, target) and τ is a temperature parameter.
     - Small τ → weights concentrate on the nearest sources (sharp)
     - Large τ → weights become uniform (flat)
     After normalisation the mean weight is 1.0, keeping the effective λ
     comparable to the original single-source MMD baseline.

This addresses the problem that merging 11 heterogeneous subjects into
one source domain creates a multi-modal distribution that is hard to
align with a single target subject.

Results saved to testresults/ (as JSON + epoch log txt).

Usage:
    python testsrc/domain_adaptation/train3_mmd.py
    python testsrc/domain_adaptation/train3_mmd.py --lambda_mmd 0.5
    python testsrc/domain_adaptation/train3_mmd.py --temperature 0.05
    python testsrc/domain_adaptation/train3_mmd.py --quiet
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
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from src.data_loader import AVAILABLE_SUBJECTS, _load_subject_all_sessions, get_flat_features
from src.domain_adaptation.mmd_model import MmdMLP
from src.domain_adaptation.mmd_loss import mmd_loss

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "testresults")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters ──────────────────────────────────────────────────────────
VAL_RATIO    = 0.2
PATIENCE     = 999999       # disabled: always run fixed epochs
MAX_EPOCHS   = 50
BATCH_SIZE   = 64
LR           = 5e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── per-subject normalization ─────────────────────────────────────────────────

def _per_subject_normalize(all_X: dict) -> dict:
    """Z-score normalize each subject independently using its own statistics."""
    normalized = {}
    for subj, X in all_X.items():
        shape = X.shape
        flat = X.reshape(shape[0], -1)
        mu = flat.mean(axis=0)
        sigma = flat.std(axis=0) + 1e-8
        flat = (flat - mu) / sigma
        normalized[subj] = flat.reshape(shape)
    return normalized


# ── new: LOSO splits with per-subject data kept separate ──────────────────────

def get_loso_splits_multi_source():
    """LOSO splits where each source subject is kept as a separate domain.

    Returns a list of dicts, one per fold.  Each dict contains:
        test_subject : int
        source_data  : {subject_id: (X_s, y_s)} — NOT flattened
        X_test, y_test : target subject data
    """
    all_X = {}
    all_y = {}
    for subj in AVAILABLE_SUBJECTS:
        all_X[subj], all_y[subj] = _load_subject_all_sessions(subj)

    all_X = _per_subject_normalize(all_X)

    splits = []
    for test_subj in AVAILABLE_SUBJECTS:
        source_data = {}
        for s in AVAILABLE_SUBJECTS:
            if s != test_subj:
                source_data[s] = (all_X[s], all_y[s])

        splits.append({
            "test_subject": test_subj,
            "source_data": source_data,
            "X_test": all_X[test_subj],
            "y_test": all_y[test_subj],
        })
    return splits


# ── new: compute per-source weights ───────────────────────────────────────────

def _compute_source_weights(src_train: dict, X_target: np.ndarray,
                            temperature: float = 0.05,
                            n_sample: int = 500):
    """Compute per-source weights based on MMD distance to target.

    Uses softmax over negative distances divided by temperature:
        weight_s ∝ exp(-d_s / τ)
    where d_s = MMD²(source_s, target) and τ = temperature.

    - Small τ → weights concentrate on the nearest sources (sharp).
    - Large τ → weights become uniform (flat).

    After normalisation the mean weight is 1.0, keeping the effective λ
    comparable to the original single-source MMD baseline.

    Args:
        src_train   : {subject_id: (X_flat, y)}  — already flattened
        X_target    : (N_tgt, 310) flattened target features
        temperature : softmax temperature τ (default 0.05)
        n_sample    : number of samples to draw per domain

    Returns:
        weights   : {subject_id: float}  mean = 1.0
        distances : {subject_id: float}  raw MMD² distances
    """
    rng = np.random.RandomState(SEED)

    # Sub-sample target
    n_tgt = min(n_sample, len(X_target))
    tgt_idx = rng.choice(len(X_target), n_tgt, replace=False)
    feat_tgt = torch.tensor(X_target[tgt_idx], dtype=torch.float32, device=DEVICE)

    distances = {}
    for s, (X_s, _) in src_train.items():
        n_s = min(n_sample, len(X_s))
        src_idx = rng.choice(len(X_s), n_s, replace=False)
        feat_src = torch.tensor(X_s[src_idx], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            dist = mmd_loss(feat_src, feat_tgt).item()
        distances[s] = max(dist, 1e-8)

    # weight_s ∝ exp(-d_s / τ)   (softmax with temperature)
    subjects = sorted(distances.keys())
    logits = np.array([-distances[s] / temperature for s in subjects])
    # Numerically stable softmax
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    softmax_weights = exp_logits / exp_logits.sum()
    weight_map = dict(zip(subjects, softmax_weights))

    # Rescale so mean weight = 1.0
    n_sources = len(weight_map)
    weights = {s: w / sum(weight_map.values()) * n_sources
               for s, w in weight_map.items()}

    return weights, distances


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

def run_mmd_fold(split: dict, lambda_mmd: float,
                 temperature: float = 0.05,
                 max_epochs: int = MAX_EPOCHS,
                 patience: int = PATIENCE):
    """
    Train MmdMLP with weighted multi-source MMD on one LOSO fold.

    Each source subject gets its own DataLoader; MMD is computed per-source
    and weighted by softmax(-d_s / τ) where d_s = MMD(source_s, target).

    Returns (test_acc, epoch_logs, weights, distances).
    """
    source_data = split["source_data"]          # {subj: (X, y)} raw shape
    X_te = get_flat_features(split["X_test"])   # (N_tgt, 310)
    y_te = split["y_test"]

    # ── per-source train/val split + flatten ────────────────────────────────────
    src_train = {}       # {subj: (X_flat_train, y_train)}
    val_parts = []       # list of (X_flat_val, y_val)

    for s, (X_s, y_s) in source_data.items():
        X_flat = get_flat_features(X_s)
        X_tr_s, y_tr_s, X_val_s, y_val_s = _train_val_split(X_flat, y_s)
        src_train[s] = (X_tr_s, y_tr_s)
        val_parts.append((X_val_s, y_val_s))

    X_val = np.concatenate([p[0] for p in val_parts], axis=0)
    y_val = np.concatenate([p[1] for p in val_parts], axis=0)

    # ── compute source weights ──────────────────────────────────────────────────
    weights, distances = _compute_source_weights(src_train, X_te,
                                                  temperature=temperature)

    # ── build loaders ───────────────────────────────────────────────────────────
    src_loaders = {s: _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
                   for s, (X_tr, y_tr) in src_train.items()}
    src_subjects = sorted(src_loaders.keys())

    tgt_loader = _to_loader(X_te, y_te, BATCH_SIZE, shuffle=True)  # labeled for iteration
    tgt_unlabeled_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te, y_te, 256, shuffle=False)

    model     = MmdMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = -1.0
    best_state     = None
    patience_count = 0
    epoch_logs     = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        # Shared target iterator — cycles through all target data each epoch
        tgt_iter = iter(tgt_unlabeled_loader)

        epoch_loss_sum = 0.0
        epoch_loss_n   = 0
        epoch_correct  = 0
        epoch_mmd_sum  = 0.0

        # ── iterate through each source subject separately ──────────────────────
        for s in src_subjects:
            w_s = weights[s]

            for (X_src, y_src) in src_loaders[s]:
                X_src = X_src.to(DEVICE)
                y_src = y_src.to(DEVICE)

                # Fetch target batch (cycle if exhausted)
                try:
                    (X_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_unlabeled_loader)
                    (X_tgt,) = next(tgt_iter)
                X_tgt = X_tgt.to(DEVICE)

                optimizer.zero_grad()

                logits_src, feat_src = model(X_src)
                _,          feat_tgt = model(X_tgt)

                loss_ce  = criterion(logits_src, y_src)
                loss_mmd = mmd_loss(feat_src, feat_tgt)
                loss     = loss_ce + w_s * lambda_mmd * loss_mmd

                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item() * len(y_src)
                epoch_loss_n   += len(y_src)
                epoch_correct  += (logits_src.argmax(1) == y_src).sum().item()
                epoch_mmd_sum  += loss_mmd.item()

        scheduler.step()

        train_loss = epoch_loss_sum / epoch_loss_n
        train_acc  = epoch_correct / epoch_loss_n

        # Validation (combined source val set)
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
            "avg_mmd":    float(epoch_mmd_sum / len(src_subjects)),
        })

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
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total   += len(y_t)
    test_acc = correct / total
    return test_acc, epoch_logs, weights, distances


# ── full experiment ───────────────────────────────────────────────────────────

def run_mmd_experiment(lambda_mmd: float = 1.0, temperature: float = 0.05,
                       test_subjects: list = None,
                       verbose: bool = True) -> dict:
    splits = get_loso_splits_multi_source()
    accs   = []
    all_fold_logs = []

    # ── filter folds if only a subset of test subjects is requested ────────────
    if test_subjects is not None:
        splits = [s for s in splits if s["test_subject"] in test_subjects]

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"
        acc, fold_logs, weights, distances = run_mmd_fold(
            split, lambda_mmd=lambda_mmd, temperature=temperature)

        accs.append(acc)

        if verbose:
            # Print source weights and distances
            print(f"\n  [{i+1:2d}/{len(splits)}] {label:20s}")
            print(f"  Source weights (MMD distance → weight):")
            for s in sorted(weights.keys()):
                print(f"    sub{s:2d}  MMD²={distances[s]:.4f}  w={weights[s]:.3f}")

            print(f"  {'Epoch':>5s}  {'Train_Loss':>10s}  {'Train_Acc':>9s}  "
                  f"{'Val_Loss':>8s}  {'Val_Acc':>8s}  {'Avg_MMD':>8s}")
            print(f"  {'-----':>5s}  {'----------':>10s}  {'---------':>9s}  "
                  f"{'--------':>8s}  {'--------':>8s}  {'--------':>8s}")
            for entry in fold_logs:
                print(f"  {entry['epoch']:5d}  {entry['train_loss']:10.4f}  "
                      f"{entry['train_acc']*100:8.2f}%  {entry['val_loss']:8.4f}  "
                      f"{entry['val_acc']*100:7.2f}%  {entry['avg_mmd']:8.4f}")
            print(f"  Test acc = {acc*100:.2f}%")

        all_fold_logs.append({
            "fold_label": label,
            "epochs": fold_logs,
            "test_acc": float(acc),
            "weights": {str(s): float(w) for s, w in weights.items()},
            "distances": {str(s): float(d) for s, d in distances.items()},
        })

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> LOSO | MLP+MMD weighted multi-source "
          f"(λ={lambda_mmd}, τ={temperature}): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":      "mlp_mmd_weighted_multisource",
        "split":      "loso",
        "lambda_mmd": lambda_mmd,
        "temperature": temperature,
        "accs":       np.array(accs),
        "mean_acc":   mean_acc,
        "std_acc":    std_acc,
    }

    _save_json(result, "loso_mlp_mmd_weighted_multisource_results.json")

    # Save epoch log as txt
    log_path = os.path.join(RESULTS_DIR, "loso_mlp_mmd_weighted_multisource_epoch_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for fold in all_fold_logs:
            f.write(f"=== {fold['fold_label']}  (test_acc={fold['test_acc']*100:.2f}%) ===\n")
            f.write(f"  Source weights:\n")
            for s in sorted(fold["weights"].keys(), key=lambda x: int(x)):
                f.write(f"    sub{s:>2s}  MMD²={fold['distances'][s]:.4f}  "
                        f"w={fold['weights'][s]:.3f}\n")
            f.write(f"{'Epoch':>5s}  {'Train_Loss':>10s}  {'Train_Acc':>9s}  "
                    f"{'Val_Loss':>8s}  {'Val_Acc':>8s}  {'Avg_MMD':>8s}\n")
            for entry in fold["epochs"]:
                f.write(f"{entry['epoch']:5d}  {entry['train_loss']:10.4f}  "
                        f"{entry['train_acc']*100:8.2f}%  {entry['val_loss']:8.4f}  "
                        f"{entry['val_acc']*100:7.2f}%  {entry['avg_mmd']:8.4f}\n")
            f.write("\n")
    print(f"    Epoch log → {log_path}")

    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP + MMD weighted multi-source on SEED (LOSO split)."
    )
    parser.add_argument(
        "--lambda_mmd", type=float, default=1.0,
        help="Weight of the MMD loss term (default: 1.0)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.05,
        help="Softmax temperature τ for source weighting. "
             "Smaller → sharper weights, larger → uniform (default: 0.05)"
    )
    parser.add_argument(
        "--test_subjects", type=int, nargs="+", default=None,
        help="Only run folds for these test subjects, e.g. --test_subjects 1 3 5. "
             "Default: run all 12 subjects."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device      : {DEVICE}")
    print(f"Model       : MLP + MMD weighted multi-source  |  Split: LOSO")
    print(f"lambda_mmd  : {args.lambda_mmd}")
    print(f"temperature : {args.temperature}")
    subj_str = (str(args.test_subjects) if args.test_subjects
                else "all 12 subjects")
    print(f"test_subjects : {subj_str}")
    print(f"Config      : max_epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"val_ratio={VAL_RATIO}  lr={LR}\n")

    run_mmd_experiment(
        lambda_mmd=args.lambda_mmd,
        temperature=args.temperature,
        test_subjects=args.test_subjects,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
