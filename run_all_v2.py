"""
Unified experiment runner for all groups (A/B/C/D/F) with temporal validation split.

Step 1: A/B groups - subject-dependent and LOSO baseline (unified hyperparams)
Step 2: C group - mixed normalization + DA methods (lambda=1)
Step 3: F group - per-subject norm + lambda sensitivity (7 lambda values)
Step 4: D group - per-subject norm + DA methods (optimal lambda from F)

Temporal validation: first 80% train, last 20% validation (no shuffling).
Results saved separately to aaaresult_v2/ for comparison with aaaresult/.

Usage:
    python run_all_v2.py --group A
    python run_all_v2.py --group B
    python run_all_v2.py --group C
    python run_all_v2.py --group F
    python run_all_v2.py --group D
    python run_all_v2.py --group all
    python run_all_v2.py --group temporal  # key comparisons with temporal split
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
sys.path.insert(0, _ROOT)

from src.data_loader import (
    get_loso_splits, get_loso_splits_subject_norm, get_flat_features,
    get_subject_dependent_splits, AVAILABLE_SUBJECTS, SESSIONS,
)

RESULTS_DIR = "/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Unified hyperparameters (same as C/D/F original)
MAX_EPOCHS = 20
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42
VAL_RATIO = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _temporal_split(X, y, val_ratio=VAL_RATIO):
    """Temporal split: first 80% train, last 20% validation. No shuffling."""
    n = len(y)
    split = int(n * (1 - val_ratio))
    return X[:split], y[:split], X[split:], y[split:]


def _stratified_split(X, y, val_ratio=VAL_RATIO):
    """Stratified random split (original method)."""
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


def _get_alpha(epoch, max_epochs):
    p = min(epoch / max_epochs, 1.0)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def _save_json(result, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


# ── Model imports ──

from src.models.baseline import MLPModel
from src.domain_adaptation.coral_model import CoralMLP
from src.domain_adaptation.coral_loss import coral_loss
from src.domain_adaptation.mmd_model import MmdMLP
from src.domain_adaptation.mmd_loss import mmd_loss
from src.domain_adaptation.dann_model import DannMLP


# ── Group A: Subject-Dependent ──

def run_group_a():
    """Subject-dependent experiment: train/test within each subject-session file.
    Uses original 9/6 trial split. Unified hyperparams."""
    print("\n" + "=" * 70)
    print("GROUP A: Subject-Dependent (unified hyperparams, temporal val)")
    print("=" * 70)

    splits = get_subject_dependent_splits(normalize=True)
    accs_svm = []
    accs_mlp = []

    for i, split in enumerate(splits):
        X_tr = get_flat_features(split["X_train"])
        X_te = get_flat_features(split["X_test"])
        y_tr, y_te = split["y_train"], split["y_test"]

        # Temporal validation from training set
        X_train, y_train, X_val, y_val = _temporal_split(X_tr, y_tr)

        train_loader = _to_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
        val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
        test_loader = _to_loader(X_te, y_te, 256, shuffle=False)

        model = MLPModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_val, best_state = -1.0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(X_b), y_b).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                    c += (model(X_v).argmax(1) == y_v).sum().item()
                    t += len(y_v)
            if c / t > best_val:
                best_val = c / t
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                c += (model(X_t).argmax(1) == y_t).sum().item()
                t += len(y_t)
        acc = c / t
        accs_mlp.append(acc)
        print(f"  [{i+1}/{len(splits)}] sub{split['subject']}_s{split['session']}  MLP acc={acc*100:.2f}%")

    result = {
        "group": "A",
        "model": "MLP",
        "split": "subject_dependent",
        "val_method": "temporal",
        "mean_acc": float(np.mean(accs_mlp)),
        "std_acc": float(np.std(accs_mlp)),
        "accs": [float(a) for a in accs_mlp],
        "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                         "weight_decay": WEIGHT_DECAY, "seed": SEED},
    }
    _save_json(result, "group_A_mlp_temporal_results.json")
    print(f"\n>>> Group A MLP: acc = {np.mean(accs_mlp)*100:.2f}% ± {np.std(accs_mlp)*100:.2f}%")


# ── Group B: LOSO Baseline ──

def run_group_b():
    """LOSO baseline experiment (SVM + MLP). Unified hyperparams, temporal validation."""
    print("\n" + "=" * 70)
    print("GROUP B: LOSO Baseline (unified hyperparams, temporal val)")
    print("=" * 70)

    for norm_name, norm_fn in [("mixed", get_loso_splits), ("per_subject", get_loso_splits_subject_norm)]:
        splits = norm_fn() if norm_name == "mixed" else norm_fn()
        accs = []

        for i, split in enumerate(splits):
            X_tr_all = get_flat_features(split["X_train"])
            X_te = get_flat_features(split["X_test"])
            y_tr_all, y_te = split["y_train"], split["y_test"]

            X_train, y_train, X_val, y_val = _temporal_split(X_tr_all, y_tr_all)

            train_loader = _to_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
            val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
            test_loader = _to_loader(X_te, y_te, 256, shuffle=False)

            model = MLPModel().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
            criterion = nn.CrossEntropyLoss()

            best_val, best_state = -1.0, None
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for X_b, y_b in train_loader:
                    X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    criterion(model(X_b), y_b).backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                c, t = 0, 0
                with torch.no_grad():
                    for X_v, y_v in val_loader:
                        X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                        c += (model(X_v).argmax(1) == y_v).sum().item()
                        t += len(y_v)
                if c / t > best_val:
                    best_val = c / t
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for X_t, y_t in test_loader:
                    X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                    c += (model(X_t).argmax(1) == y_t).sum().item()
                    t += len(y_t)
            acc = c / t
            accs.append(acc)
            print(f"  [{i+1}/{len(splits)}] test_sub{split['test_subject']}  acc={acc*100:.2f}%")

        result = {
            "group": "B",
            "model": "MLP",
            "split": "loso",
            "norm": norm_name,
            "val_method": "temporal",
            "lambda": 1.0,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "accs": [float(a) for a in accs],
            "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                             "weight_decay": WEIGHT_DECAY, "seed": SEED},
        }
        _save_json(result, f"group_B_mlp_{norm_name}_temporal_results.json")
        print(f"\n>>> Group B MLP ({norm_name}): acc = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


# ── Generic LOSO DA training ──

def run_loso_fold(method, norm_mode, split, lambda_da=1.0, val_fn=_temporal_split):
    """Train one LOSO fold. Returns accuracy."""
    X_tr_all = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr_all, y_te = split["y_train"], split["y_test"]

    X_train, y_train, X_val, y_val = val_fn(X_tr_all, y_tr_all)

    src_loader = _to_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    test_loader = _to_loader(X_te, y_te, 256, shuffle=False)
    tgt_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)

    if method == "baseline":
        model = MLPModel().to(DEVICE)
    elif method == "coral":
        model = CoralMLP().to(DEVICE)
    elif method == "mmd":
        model = MmdMLP().to(DEVICE)
    elif method == "dann":
        model = DannMLP().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    label_crit = nn.CrossEntropyLoss()
    domain_crit = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tgt_iter = iter(tgt_loader) if method != "baseline" else None
        for X_src, y_src in src_loader:
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            optimizer.zero_grad()

            if method == "baseline":
                loss = label_crit(model(X_src), y_src)
            elif method in ("coral", "mmd"):
                try:
                    (X_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    (X_tgt,) = next(tgt_iter)
                X_tgt = X_tgt.to(DEVICE)
                logits_s, feat_s = model(X_src)
                _, feat_t = model(X_tgt)
                da_fn = coral_loss if method == "coral" else mmd_loss
                loss = label_crit(logits_s, y_src) + lambda_da * da_fn(feat_s, feat_t)
            else:  # dann
                alpha = _get_alpha(epoch, MAX_EPOCHS)
                try:
                    (X_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    (X_tgt,) = next(tgt_iter)
                X_tgt = X_tgt.to(DEVICE)
                n_s, n_t = X_src.size(0), X_tgt.size(0)
                d_s = torch.zeros(n_s, dtype=torch.long, device=DEVICE)
                d_t = torch.ones(n_t, dtype=torch.long, device=DEVICE)
                lab_s, dom_s, _ = model(X_src, alpha)
                _, dom_t, _ = model(X_tgt, alpha)
                loss = label_crit(lab_s, y_src) + lambda_da * (
                    domain_crit(dom_s, d_s) + domain_crit(dom_t, d_t))

            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                if method == "baseline":
                    logits = model(X_v)
                elif method == "dann":
                    logits, _, _ = model(X_v, 0.0)
                else:
                    logits, _ = model(X_v)
                c += (logits.argmax(1) == y_v).sum().item()
                t += len(y_v)
        if c / t > best_val:
            best_val = c / t
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for X_t, y_t in test_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            if method == "baseline":
                logits = model(X_t)
            elif method == "dann":
                logits, _, _ = model(X_t, 0.0)
            else:
                logits, _ = model(X_t)
            c += (logits.argmax(1) == y_t).sum().item()
            t += len(y_t)
    return c / t


def run_experiment(method, norm, lambda_da=1.0, group="C", val_fn=_temporal_split):
    """Run full LOSO experiment and save results."""
    splits_fn = get_loso_splits if norm == "mixed" else get_loso_splits_subject_norm
    splits = splits_fn()
    val_name = "temporal" if val_fn == _temporal_split else "stratified"
    accs = []

    for i, split in enumerate(splits):
        acc = run_loso_fold(method, norm, split, lambda_da=lambda_da, val_fn=val_fn)
        accs.append(acc)
        print(f"  [{i+1}/{len(splits)}] test_sub{split['test_subject']}  acc={acc*100:.2f}%")

    result = {
        "group": group,
        "model": method,
        "split": "loso",
        "norm": norm,
        "lambda": float(lambda_da),
        "val_method": val_name,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "accs": [float(a) for a in accs],
        "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                         "weight_decay": WEIGHT_DECAY, "seed": SEED, "val_ratio": VAL_RATIO},
    }
    fname = f"{method}_{norm}_lambda{lambda_da}_{val_name}_results.json"
    _save_json(result, fname)
    print(f"\n>>> {method} ({norm}, λ={lambda_da}, {val_name}): "
          f"acc = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    return result


# ── Group C: Mixed Norm DA (lambda=1) ──

def run_group_c():
    print("\n" + "=" * 70)
    print("GROUP C: Mixed Norm + DA (lambda=1.0, temporal validation)")
    print("=" * 70)
    for method in ["baseline", "coral", "mmd", "dann"]:
        run_experiment(method, "mixed", lambda_da=1.0, group="C", val_fn=_temporal_split)


# ── Group F: Lambda Sensitivity (per-subject, 7 lambda values) ──

def run_group_f():
    print("\n" + "=" * 70)
    print("GROUP F: Lambda Sensitivity (per-subject, 7 lambda values, temporal validation)")
    print("=" * 70)
    lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for method in ["coral", "mmd", "dann"]:
        for lam in lambdas:
            run_experiment(method, "per_subject", lambda_da=lam, group="F", val_fn=_temporal_split)


# ── Group D: Per-Subject Norm + DA (optimal lambda) ──

def run_group_d():
    print("\n" + "=" * 70)
    print("GROUP D: Per-Subject Norm + DA (optimal lambda, temporal validation)")
    print("=" * 70)
    # Baseline
    run_experiment("baseline", "per_subject", lambda_da=1.0, group="D", val_fn=_temporal_split)
    # DA methods with their optimal lambda (from F group)
    optimal_lambdas = {"coral": 5.0, "mmd": 10.0, "dann": 5.0}
    for method, lam in optimal_lambdas.items():
        run_experiment(method, "per_subject", lambda_da=lam, group="D", val_fn=_temporal_split)


# ── Temporal vs Stratified Comparison ──

def run_temporal_comparison():
    """Rerun key experiments with stratified split for direct comparison."""
    print("\n" + "=" * 70)
    print("COMPARISON: Temporal vs Stratified validation (key configs)")
    print("=" * 70)
    configs = [
        ("baseline", "mixed", 1.0),
        ("baseline", "per_subject", 1.0),
        ("mmd", "per_subject", 10.0),
        ("coral", "per_subject", 5.0),
        ("dann", "per_subject", 5.0),
    ]
    for method, norm, lam in configs:
        print(f"\n--- Stratified validation ---")
        run_experiment(method, norm, lambda_da=lam, group="stratified", val_fn=_stratified_split)
        print(f"\n--- Temporal validation ---")
        run_experiment(method, norm, lambda_da=lam, group="temporal", val_fn=_temporal_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiment groups with temporal validation")
    parser.add_argument("--group", choices=["A", "B", "C", "F", "D", "temporal", "all"],
                        required=True, help="Which group to run")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Hyperparams: epochs={MAX_EPOCHS}, lr={LR}, bs={BATCH_SIZE}, wd={WEIGHT_DECAY}, seed={SEED}")
    print(f"Results dir: {RESULTS_DIR}")

    t0 = time.time()
    if args.group == "A":
        run_group_a()
    elif args.group == "B":
        run_group_b()
    elif args.group == "C":
        run_group_c()
    elif args.group == "F":
        run_group_f()
    elif args.group == "D":
        run_group_d()
    elif args.group == "temporal":
        run_temporal_comparison()
    elif args.group == "all":
        run_group_a()
        run_group_b()
        run_group_c()
        run_group_f()
        run_group_d()
        run_temporal_comparison()

    print(f"\nTotal time: {time.time()-t0:.0f}s")