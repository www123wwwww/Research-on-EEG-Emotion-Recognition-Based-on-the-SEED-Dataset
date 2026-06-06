"""
Unified LOSO training script for C/D/F experiments.

Unified hyperparameters (no early stopping):
  MAX_EPOCHS   = 20
  LR           = 5e-4
  BATCH_SIZE   = 128
  WEIGHT_DECAY = 1e-4
  VAL_RATIO    = 0.2
  SEED         = 42

Normalization modes:
  mixed        — original: z-score with 11-source-subject combined statistics
  per_subject  — improved: z-score each subject independently before merging

Methods:
  baseline     — plain MLP (no domain adaptation)
  coral        — MLP + CORAL loss
  mmd          — MLP + MMD loss
  dann         — MLP + DANN (gradient reversal layer)

Results saved to: /hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult/

Usage examples:
  python run_loso.py --method baseline --norm mixed
  python run_loso.py --method baseline --norm per_subject
  python run_loso.py --method coral --norm mixed
  python run_loso.py --method coral --norm per_subject --lambda_da 0.5
  python run_loso.py --method mmd --norm per_subject --lambda_da 1.0
  python run_loso.py --method dann --norm per_subject --lambda_da 2.0
  python run_loso.py --method mmd --norm per_subject --lambda_da 1.0 --weighted_multisource --temperature 0.05
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
sys.path.insert(0, _ROOT)

from src.data_loader import (
    get_loso_splits,
    get_loso_splits_subject_norm,
    get_flat_features,
    AVAILABLE_SUBJECTS,
    _load_subject_all_sessions,
)
from src.domain_adaptation.coral_model import CoralMLP
from src.domain_adaptation.coral_loss import coral_loss
from src.domain_adaptation.mmd_model import MmdMLP
from src.domain_adaptation.mmd_loss import mmd_loss
from src.domain_adaptation.dann_model import DannMLP
from src.models.baseline import MLPModel

# ── unified config ──────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MAX_EPOCHS = 20
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2

RESULTS_DIR = "/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── utilities ────────────────────────────────────────────────────────────────────

def _stratified_split(X, y, val_ratio=VAL_RATIO):
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


def _per_subject_normalize(all_X):
    normalized = {}
    for subj, X in all_X.items():
        shape = X.shape
        flat = X.reshape(shape[0], -1)
        mu = flat.mean(axis=0)
        sigma = flat.std(axis=0) + 1e-8
        flat = (flat - mu) / sigma
        normalized[subj] = flat.reshape(shape)
    return normalized


def get_loso_splits_weighted():
    """LOSO splits with per-subject norm, keeping source data separate per subject."""
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
    print(f"  Saved -> {path}")


# ── fold runners ───────────────────────────────────────────────────────────────────

def run_baseline_fold(split, norm_mode):
    X_tr_all = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr_all = split["y_train"]
    y_te = split["y_test"]

    X_tr, y_tr, X_val, y_val = _stratified_split(X_tr_all, y_tr_all)
    train_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    test_loader = _to_loader(X_te, y_te, 256, shuffle=False)

    model = MLPModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_logs = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                correct += (model(X_v).argmax(1) == y_v).sum().item()
                total += len(y_v)
        val_acc = correct / total

        epoch_logs.append({"epoch": epoch, "val_acc": float(val_acc)})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in test_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            correct += (model(X_t).argmax(1) == y_t).sum().item()
            total += len(y_t)
    return correct / total, epoch_logs


def run_coral_fold(split, norm_mode, lambda_coral):
    X_tr_all = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr_all = split["y_train"]
    y_te = split["y_test"]

    X_tr, y_tr, X_val, y_val = _stratified_split(X_tr_all, y_tr_all)
    src_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    tgt_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te, y_te, 256, shuffle=False)

    model = CoralMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_logs = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tgt_iter = iter(tgt_loader)
        for X_src, y_src in src_loader:
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            try:
                (X_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)
            optimizer.zero_grad()
            logits_src, feat_src = model(X_src)
            _, feat_tgt = model(X_tgt)
            loss = criterion(logits_src, y_src) + lambda_coral * coral_loss(feat_src, feat_tgt)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                correct += (logits.argmax(1) == y_v).sum().item()
                total += len(y_v)
        val_acc = correct / total
        epoch_logs.append({"epoch": epoch, "val_acc": float(val_acc)})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total += len(y_t)
    return correct / total, epoch_logs


def run_mmd_fold(split, norm_mode, lambda_mmd):
    X_tr_all = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr_all = split["y_train"]
    y_te = split["y_test"]

    X_tr, y_tr, X_val, y_val = _stratified_split(X_tr_all, y_tr_all)
    src_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    tgt_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te, y_te, 256, shuffle=False)

    model = MmdMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_logs = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tgt_iter = iter(tgt_loader)
        for X_src, y_src in src_loader:
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            try:
                (X_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)
            optimizer.zero_grad()
            logits_src, feat_src = model(X_src)
            _, feat_tgt = model(X_tgt)
            loss = criterion(logits_src, y_src) + lambda_mmd * mmd_loss(feat_src, feat_tgt)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                correct += (logits.argmax(1) == y_v).sum().item()
                total += len(y_v)
        val_acc = correct / total
        epoch_logs.append({"epoch": epoch, "val_acc": float(val_acc)})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total += len(y_t)
    return correct / total, epoch_logs


def run_dann_fold(split, norm_mode, lambda_dann):
    X_tr_all = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr_all = split["y_train"]
    y_te = split["y_test"]

    X_tr, y_tr, X_val, y_val = _stratified_split(X_tr_all, y_tr_all)
    src_loader = _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    tgt_loader = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te, y_te, 256, shuffle=False)

    model = DannMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    label_crit = nn.CrossEntropyLoss()
    domain_crit = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_logs = []

    for epoch in range(1, MAX_EPOCHS + 1):
        alpha = _get_alpha(epoch, MAX_EPOCHS)
        model.train()
        tgt_iter = iter(tgt_loader)
        for X_src, y_src in src_loader:
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            try:
                (X_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (X_tgt,) = next(tgt_iter)
            X_tgt = X_tgt.to(DEVICE)
            n_src, n_tgt = X_src.size(0), X_tgt.size(0)
            dom_src = torch.zeros(n_src, dtype=torch.long, device=DEVICE)
            dom_tgt = torch.ones(n_tgt, dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            label_logits, dom_logits_src, _ = model(X_src, alpha)
            _, dom_logits_tgt, _ = model(X_tgt, alpha)
            loss = label_crit(label_logits, y_src) + lambda_dann * (
                domain_crit(dom_logits_src, dom_src) + domain_crit(dom_logits_tgt, dom_tgt)
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                label_logits, _, _ = model(X_v, alpha=0.0)
                correct += (label_logits.argmax(1) == y_v).sum().item()
                total += len(y_v)
        val_acc = correct / total
        epoch_logs.append({"epoch": epoch, "val_acc": float(val_acc)})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            label_logits, _, _ = model(X_t, alpha=0.0)
            correct += (label_logits.argmax(1) == y_t).sum().item()
            total += len(y_t)
    return correct / total, epoch_logs


def run_weighted_mmd_fold(split_w, lambda_mmd, temperature):
    source_data = split_w["source_data"]
    X_te = get_flat_features(split_w["X_test"])
    y_te = split_w["y_test"]

    src_train = {}
    val_parts = []
    for s, (X_s_raw, y_s) in source_data.items():
        X_s = get_flat_features(X_s_raw)
        X_tr_s, y_tr_s, X_val_s, y_val_s = _stratified_split(X_s, y_s)
        src_train[s] = (X_tr_s, y_tr_s)
        val_parts.append((X_val_s, y_val_s))

    X_val = np.concatenate([p[0] for p in val_parts], axis=0)
    y_val = np.concatenate([p[1] for p in val_parts], axis=0)

    # compute source weights
    rng = np.random.RandomState(SEED)
    n_sample = 500
    n_tgt = min(n_sample, len(X_te))
    tgt_idx = rng.choice(len(X_te), n_tgt, replace=False)
    feat_tgt = torch.tensor(X_te[tgt_idx], dtype=torch.float32, device=DEVICE)
    distances = {}
    for s, (X_s, _) in src_train.items():
        n_s = min(n_sample, len(X_s))
        src_idx = rng.choice(len(X_s), n_s, replace=False)
        feat_src = torch.tensor(X_s[src_idx], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            distances[s] = max(mmd_loss(feat_src, feat_tgt).item(), 1e-8)

    subjects = sorted(distances.keys())
    logits_arr = np.array([-distances[s] / temperature for s in subjects])
    logits_arr = logits_arr - logits_arr.max()
    softmax_w = np.exp(logits_arr) / np.exp(logits_arr).sum()
    weights = {s: w / sum(softmax_w) * len(subjects) for s, w in zip(subjects, softmax_w)}

    src_loaders = {s: _to_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
                   for s, (X_tr, y_tr) in src_train.items()}
    src_subjects = sorted(src_loaders.keys())
    val_loader = _to_loader(X_val, y_val, 256, shuffle=False)
    tst_loader = _to_loader(X_te, y_te, 256, shuffle=False)
    tgt_unlabeled = _to_unlabeled_loader(X_te, BATCH_SIZE, shuffle=True)

    model = MmdMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_logs = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tgt_iter = iter(tgt_unlabeled)
        for s in src_subjects:
            w_s = weights[s]
            for X_src, y_src in src_loaders[s]:
                X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
                try:
                    (X_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_unlabeled)
                    (X_tgt,) = next(tgt_iter)
                X_tgt = X_tgt.to(DEVICE)
                optimizer.zero_grad()
                logits_src, feat_src = model(X_src)
                _, feat_tgt = model(X_tgt)
                loss = criterion(logits_src, y_src) + w_s * lambda_mmd * mmd_loss(feat_src, feat_tgt)
                loss.backward()
                optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits, _ = model(X_v)
                correct += (logits.argmax(1) == y_v).sum().item()
                total += len(y_v)
        val_acc = correct / total
        epoch_logs.append({"epoch": epoch, "val_acc": float(val_acc)})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, y_t in tst_loader:
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            logits, _ = model(X_t)
            correct += (logits.argmax(1) == y_t).sum().item()
            total += len(y_t)
    return correct / total, epoch_logs


# ── experiment runner ─────────────────────────────────────────────────────────────

METHOD_MAP = {
    "baseline": run_baseline_fold,
    "coral": run_coral_fold,
    "mmd": run_mmd_fold,
    "dann": run_dann_fold,
}


def run_experiment(method, norm, lambda_da=1.0, weighted_multisource=False, temperature=0.05):
    if norm == "mixed":
        splits = get_loso_splits(normalize=True)
    elif norm == "per_subject":
        splits = get_loso_splits_subject_norm()
    else:
        raise ValueError(f"Unknown norm: {norm}")

    accs = []
    all_fold_logs = []

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"

        if method == "baseline":
            acc, logs = run_baseline_fold(split, norm)
        elif method == "coral":
            acc, logs = run_coral_fold(split, norm, lambda_da)
        elif method == "mmd":
            acc, logs = run_mmd_fold(split, norm, lambda_da)
        elif method == "dann":
            acc, logs = run_dann_fold(split, norm, lambda_da)
        else:
            raise ValueError(f"Unknown method: {method}")

        accs.append(acc)
        all_fold_logs.append({"fold": label, "test_acc": float(acc), "epochs": logs})
        print(f"  [{i+1:2d}/{len(splits)}] {label:18s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    # build filename
    parts = [method, norm]
    if weighted_multisource:
        parts.append(f"weighted_t{temperature}")
    fname = "_".join(parts) + f"_lambda{lambda_da}" + "_results.json"

    result = {
        "method": method,
        "norm": norm,
        "lambda_da": lambda_da,
        "max_epochs": MAX_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "weighted_multisource": weighted_multisource,
        **({"temperature": temperature} if weighted_multisource else {}),
        "accs": np.array(accs),
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "fold_logs": all_fold_logs,
    }
    _save_json(result, fname)
    return result


def run_weighted_mmd_experiment(lambda_mmd=1.0, temperature=0.05):
    splits = get_loso_splits_weighted()
    accs = []
    all_fold_logs = []

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"
        acc, logs = run_weighted_mmd_fold(split, lambda_mmd, temperature)
        accs.append(acc)
        all_fold_logs.append({"fold": label, "test_acc": float(acc), "epochs": logs})
        print(f"  [{i+1:2d}/{len(splits)}] {label:18s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    fname = f"mmd_per_subject_weighted_t{temperature}_lambda{lambda_mmd}_results.json"
    result = {
        "method": "mmd",
        "norm": "per_subject",
        "lambda_da": lambda_mmd,
        "max_epochs": MAX_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "weighted_multisource": True,
        "temperature": temperature,
        "accs": np.array(accs),
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "fold_logs": all_fold_logs,
    }
    _save_json(result, fname)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LOSO training script")
    parser.add_argument("--method", choices=["baseline", "coral", "mmd", "dann"], required=True)
    parser.add_argument("--norm", choices=["mixed", "per_subject"], required=True)
    parser.add_argument("--lambda_da", type=float, default=1.0)
    parser.add_argument("--weighted_multisource", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    norm_str = "per-subject norm" if args.norm == "per_subject" else "mixed norm"
    print(f"Method : {args.method}")
    print(f"Norm   : {norm_str}")
    print(f"Lambda : {args.lambda_da}")
    print(f"Epochs : {MAX_EPOCHS}  LR: {LR}  BS: {BATCH_SIZE}")
    print(f"Device : {DEVICE}")
    print()

    if args.weighted_multisource:
        run_weighted_mmd_experiment(
            lambda_mmd=args.lambda_da, temperature=args.temperature
        )
    else:
        run_experiment(
            method=args.method, norm=args.norm, lambda_da=args.lambda_da,
            weighted_multisource=False
        )