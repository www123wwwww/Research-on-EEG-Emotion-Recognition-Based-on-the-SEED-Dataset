"""
No-validation-set experiments: train on all source data, use final epoch model.

Groups:
  A: subject-dependent baseline (per-subject norm)
  B: LOSO baseline (mixed + per-subject norm)
  C: mixed norm + DA (lambda=1.0)
  D: per-subject norm + baseline + DA with optimal lambda from v1
     (CORAL λ=5.0, MMD λ=10.0, DANN λ=5.0)

No validation set. Train 20 epochs on all source data, evaluate on target domain.
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
    get_subject_dependent_splits, AVAILABLE_SUBJECTS,
)

RESULTS_DIR = "/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult_v3"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_EPOCHS = 20
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


from src.models.baseline import MLPModel
from src.domain_adaptation.coral_model import CoralMLP
from src.domain_adaptation.coral_loss import coral_loss
from src.domain_adaptation.mmd_model import MmdMLP
from src.domain_adaptation.mmd_loss import mmd_loss
from src.domain_adaptation.dann_model import DannMLP


def run_group_a():
    """Subject-dependent baseline, no validation."""
    print("\n" + "=" * 70)
    print("GROUP A: Subject-Dependent (no validation, final epoch)")
    print("=" * 70)
    splits = get_subject_dependent_splits(normalize=True)
    accs = []

    for i, split in enumerate(splits):
        X_tr = get_flat_features(split["X_train"])
        X_te = get_flat_features(split["X_test"])
        y_tr, y_te = split["y_train"], split["y_test"]

        train_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
        test_loader = _to_loader(X_te, y_te, 256, shuffle=False)

        model = MLPModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss()

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
            for X_t, y_t in test_loader:
                X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                c += (model(X_t).argmax(1) == y_t).sum().item()
                t += len(y_t)
        acc = c / t
        accs.append(acc)
        print(f"  [{i+1}/{len(splits)}] sub{split['subject']}_s{split['session']}  acc={acc*100:.2f}%")

    result = {
        "group": "A", "model": "MLP", "split": "subject_dependent",
        "val_method": "none", "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "accs": [float(a) for a in accs],
        "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                         "weight_decay": WEIGHT_DECAY, "seed": SEED},
    }
    _save_json(result, "group_A_mlp_noval_results.json")
    print(f"\n>>> Group A: acc = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


def run_group_b():
    """LOSO baseline, no validation."""
    print("\n" + "=" * 70)
    print("GROUP B: LOSO Baseline (no validation, final epoch)")
    print("=" * 70)
    for norm_name, splits_fn in [("mixed", get_loso_splits), ("per_subject", get_loso_splits_subject_norm)]:
        splits = splits_fn()
        accs = []
        for i, split in enumerate(splits):
            X_tr = get_flat_features(split["X_train"])
            X_te = get_flat_features(split["X_test"])
            y_tr, y_te = split["y_train"], split["y_test"]

            train_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
            test_loader = _to_loader(X_te, y_te, 256, shuffle=False)

            model = MLPModel().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
            criterion = nn.CrossEntropyLoss()

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
                for X_t, y_t in test_loader:
                    X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                    c += (model(X_t).argmax(1) == y_t).sum().item()
                    t += len(y_t)
            acc = c / t
            accs.append(acc)
            print(f"  [{i+1}/{len(splits)}] test_sub{split['test_subject']}  acc={acc*100:.2f}%")

        result = {
            "group": "B", "model": "MLP", "split": "loso", "norm": norm_name,
            "val_method": "none", "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "accs": [float(a) for a in accs],
            "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                             "weight_decay": WEIGHT_DECAY, "seed": SEED},
        }
        _save_json(result, f"group_B_mlp_{norm_name}_noval_results.json")
        print(f"\n>>> Group B ({norm_name}): acc = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


def run_loso_fold(method, norm, split, lambda_da=1.0):
    """Train one LOSO fold without validation. Returns accuracy."""
    X_tr = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr, y_te = split["y_train"], split["y_test"]

    src_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
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
            else:
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


def run_experiment(method, norm, lambda_da=1.0, group="C"):
    splits_fn = get_loso_splits if norm == "mixed" else get_loso_splits_subject_norm
    splits = splits_fn()
    accs = []
    for i, split in enumerate(splits):
        acc = run_loso_fold(method, norm, split, lambda_da=lambda_da)
        accs.append(acc)
        print(f"  [{i+1}/{len(splits)}] test_sub{split['test_subject']}  acc={acc*100:.2f}%")

    result = {
        "group": group, "model": method, "split": "loso", "norm": norm,
        "lambda": float(lambda_da), "val_method": "none",
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "accs": [float(a) for a in accs],
        "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                         "weight_decay": WEIGHT_DECAY, "seed": SEED},
    }
    fname = f"{method}_{norm}_lambda{lambda_da}_noval_results.json"
    _save_json(result, fname)
    print(f"\n>>> {method} ({norm}, λ={lambda_da}, noval): "
          f"acc = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


def run_group_c():
    print("\n" + "=" * 70)
    print("GROUP C: Mixed Norm + DA (lambda=1.0, no validation)")
    print("=" * 70)
    for method in ["baseline", "coral", "mmd", "dann"]:
        run_experiment(method, "mixed", lambda_da=1.0, group="C")


def run_group_d():
    print("\n" + "=" * 70)
    print("GROUP D: Per-Subject Norm + DA (optimal lambda from v1, no validation)")
    print("=" * 70)
    run_experiment("baseline", "per_subject", lambda_da=1.0, group="D")
    run_experiment("coral", "per_subject", lambda_da=5.0, group="D")
    run_experiment("mmd", "per_subject", lambda_da=10.0, group="D")
    run_experiment("dann", "per_subject", lambda_da=5.0, group="D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments without validation set")
    parser.add_argument("--group", choices=["A", "B", "C", "D", "all"], required=True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Hyperparams: epochs={MAX_EPOCHS}, lr={LR}, bs={BATCH_SIZE}, wd={WEIGHT_DECAY}, seed={SEED}")
    print(f"No validation set - using final epoch model")
    print(f"Results dir: {RESULTS_DIR}")

    t0 = time.time()
    if args.group == "A":
        run_group_a()
    elif args.group == "B":
        run_group_b()
    elif args.group == "C":
        run_group_c()
    elif args.group == "D":
        run_group_d()
    elif args.group == "all":
        run_group_a()
        run_group_b()
        run_group_c()
        run_group_d()

    print(f"\nTotal time: {time.time()-t0:.0f}s")