"""
Training pipeline for all models (test reproduction).

Based on src/train.py. Only changes:
  - Results saved to testresults/ instead of results/
  - Results saved as JSON instead of npy
  - Per-epoch train_loss / val_acc printed and logged

Usage:
    python testsrc/train.py --model svm      --split loso
    python testsrc/train.py --model mlp      --split loso
    python testsrc/train.py --model attn_mlp --split loso
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_subject_dependent_splits, get_loso_splits, get_flat_features
from src.models.baseline import SVMModel, MLPModel
from src.models.attention_mlp import AttentionMLP

# ── changed: results dir ─────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testresults")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_RATIO    = 0.2
PATIENCE     = 999999       # disabled: always run fixed epochs
MAX_EPOCHS   = 5
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def _train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = VAL_RATIO):
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
    if model_type == "mlp":
        X_tr_all = get_flat_features(split["X_train"])
        X_te     = get_flat_features(split["X_test"])
    else:
        X_tr_all = split["X_train"]
        X_te     = split["X_test"]

    y_tr_all = split["y_train"]
    y_te     = split["y_test"]

    X_tr, y_tr, X_val, y_val = _train_val_split(X_tr_all, y_tr_all)

    train_loader = _to_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    val_loader   = _to_loader(X_val, y_val, 256,        shuffle=False)
    test_loader  = _to_loader(X_te,  y_te,  256,        shuffle=False)

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

    # ── changed: collect per-epoch log ─────────────────────────────────────────
    epoch_logs = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _epoch_loss(model, train_loader, criterion, model_type, train=True, optimizer=optimizer)
        scheduler.step()

        val_loss, val_acc = _epoch_loss(model, val_loader, criterion, model_type, train=False)

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

    model.load_state_dict(best_state)
    _, test_acc = _epoch_loss(model, test_loader, criterion, model_type, train=False)

    band_w, chan_w = _collect_weights(model, test_loader, model_type)
    return test_acc, band_w, chan_w, epoch_logs


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
    # ── changed: collect all fold logs ─────────────────────────────────────────
    all_fold_logs = []

    for i, split in enumerate(splits):
        label = fold_key(split)

        if model_type == "svm":
            acc      = run_svm_fold(split)
            bw = cw  = None
            fold_logs = None
        else:
            acc, bw, cw, fold_logs = run_torch_fold(split, model_type)

        accs.append(acc)
        if bw is not None:
            all_band_w.append(bw.mean(axis=0))
            all_chan_w.append(cw.mean(axis=0))

        # ── changed: print per-epoch log and collect ────────────────────────────
        if verbose:
            print(f"\n  [{i+1:2d}/{len(splits)}] {label:25s}")
            if fold_logs is not None:
                print(f"  {'Epoch':>5s}  {'Train_Loss':>10s}  {'Train_Acc':>9s}  {'Val_Loss':>8s}  {'Val_Acc':>8s}")
                print(f"  {'-----':>5s}  {'----------':>10s}  {'---------':>9s}  {'--------':>8s}  {'--------':>8s}")
                for entry in fold_logs:
                    print(f"  {entry['epoch']:5d}  {entry['train_loss']:10.4f}  {entry['train_acc']*100:8.2f}%  {entry['val_loss']:8.4f}  {entry['val_acc']*100:7.2f}%")
            print(f"  Test acc = {acc*100:.2f}%")

        if fold_logs is not None:
            all_fold_logs.append({"fold_label": label, "epochs": fold_logs, "test_acc": float(acc)})

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

    # ── changed: save as JSON ─────────────────────────────────────────────────
    _save_json(result, f"{split_strategy}_{model_type}_results.json")

    # ── changed: save epoch log as txt ─────────────────────────────────────────
    if all_fold_logs:
        log_path = os.path.join(RESULTS_DIR, f"{split_strategy}_{model_type}_epoch_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for fold in all_fold_logs:
                f.write(f"=== {fold['fold_label']}  (test_acc={fold['test_acc']*100:.2f}%) ===\n")
                f.write(f"{'Epoch':>5s}  {'Train_Loss':>10s}  {'Train_Acc':>9s}  {'Val_Loss':>8s}  {'Val_Acc':>8s}\n")
                for entry in fold["epochs"]:
                    f.write(f"{entry['epoch']:5d}  {entry['train_loss']:10.4f}  {entry['train_acc']*100:8.2f}%  {entry['val_loss']:8.4f}  {entry['val_acc']*100:7.2f}%\n")
                f.write("\n")
        print(f"    Epoch log → {log_path}")

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
