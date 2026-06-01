"""
Training pipeline: MLP + DANN domain adaptation (LOSO split only).

Based on testsrc/domain_adaptation/train_dann.py. Only change:
  - Per-subject z-score normalization BEFORE merging into source domain.
    Each subject is normalized using its own mean/std, so the 11 source
    subjects are brought to a common scale before being concatenated.
    This eliminates inter-subject distribution mismatch within the source
    domain, making subsequent DANN alignment between source and target
    more meaningful.

Results saved to testresults/ (as JSON + epoch log txt).

Usage:
    python testsrc/domain_adaptation/train2_dann.py
    python testsrc/domain_adaptation/train2_dann.py --lambda_dann 0.5
    python testsrc/domain_adaptation/train2_dann.py --quiet
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
from src.domain_adaptation.dann_model import DannMLP

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── changed: results dir ─────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "testresults")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters (same as CORAL/MMD for fair comparison) ─────────────────
VAL_RATIO    = 0.2
PATIENCE     = 999999       # disabled: always run fixed epochs
MAX_EPOCHS   = 50
BATCH_SIZE   = 128
LR           = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── new: per-subject normalization ───────────────────────────────────────────

def _per_subject_normalize(all_X: dict) -> dict:
    """Z-score normalize each subject independently using its own statistics.

    Args:
        all_X: {subject_id: np.ndarray of shape (N_s, 5, 62)}

    Returns:
        {subject_id: normalized np.ndarray of same shape}
    """
    normalized = {}
    for subj, X in all_X.items():
        shape = X.shape
        flat = X.reshape(shape[0], -1)
        mu = flat.mean(axis=0)
        sigma = flat.std(axis=0) + 1e-8
        flat = (flat - mu) / sigma
        normalized[subj] = flat.reshape(shape)
    return normalized


def get_loso_splits_subject_norm():
    """LOSO splits with per-subject z-score normalization BEFORE merging.

    Each subject is first normalized with its own mean/std, eliminating
    inter-subject scale differences. Then the 11 source subjects are
    concatenated into a single source domain that is much more coherent
    than the raw mixed distribution.
    """
    # Load raw data (no normalization yet)
    all_X = {}
    all_y = {}
    for subj in AVAILABLE_SUBJECTS:
        all_X[subj], all_y[subj] = _load_subject_all_sessions(subj)

    # Per-subject normalization
    all_X = _per_subject_normalize(all_X)

    # Build LOSO folds from already-normalized data
    splits = []
    for test_subj in AVAILABLE_SUBJECTS:
        X_te = all_X[test_subj]
        y_te = all_y[test_subj]

        train_parts_X = [all_X[s] for s in AVAILABLE_SUBJECTS if s != test_subj]
        train_parts_y = [all_y[s] for s in AVAILABLE_SUBJECTS if s != test_subj]
        X_tr = np.concatenate(train_parts_X, axis=0)
        y_tr = np.concatenate(train_parts_y, axis=0)

        splits.append({
            "test_subject": test_subj,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
        })
    return splits


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

def run_dann_fold(split: dict, lambda_dann: float,
                  max_epochs: int = MAX_EPOCHS,
                  patience: int = PATIENCE):
    """
    Train DannMLP on one LOSO fold.

    Returns (test_acc, epoch_logs).

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

    # ── changed: collect per-epoch log ─────────────────────────────────────────
    epoch_logs = []

    for epoch in range(1, max_epochs + 1):
        alpha = _get_alpha(epoch, max_epochs)
        model.train()
        tgt_iter = iter(tgt_loader)
        epoch_loss_sum = 0.0
        epoch_loss_n   = 0
        epoch_correct  = 0

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

            epoch_loss_sum += loss.item() * n_src
            epoch_loss_n   += n_src
            epoch_correct  += (label_logits.argmax(1) == y_src).sum().item()

        scheduler.step()

        train_loss = epoch_loss_sum / epoch_loss_n
        train_acc  = epoch_correct / epoch_loss_n

        # Validation: use α=0 so GRL does not distort label predictions
        model.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                label_logits, _, _ = model(X_v, alpha=0.0)
                val_loss_sum += label_crit(label_logits, y_v).item() * len(y_v)
                correct += (label_logits.argmax(1) == y_v).sum().item()
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
    test_acc = correct / total
    return test_acc, epoch_logs


# ── full experiment ───────────────────────────────────────────────────────────

def run_dann_experiment(lambda_dann: float = 1.0, verbose: bool = True) -> dict:
    splits = get_loso_splits_subject_norm()
    accs   = []
    # ── changed: collect all fold logs ─────────────────────────────────────────
    all_fold_logs = []

    for i, split in enumerate(splits):
        label = f"test_sub{split['test_subject']}"
        acc, fold_logs = run_dann_fold(split, lambda_dann=lambda_dann)
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
    print(f"\n>>> LOSO | MLP+DANN + per-subject norm (λ={lambda_dann}): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":       "mlp_dann_subject_norm",
        "split":       "loso",
        "lambda_dann": lambda_dann,
        "accs":        np.array(accs),
        "mean_acc":    mean_acc,
        "std_acc":     std_acc,
    }

    # ── changed: save as JSON ─────────────────────────────────────────────────
    _save_json(result, "loso_mlp_dann_subject_norm_results.json")

    # ── changed: save epoch log as txt ─────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "loso_mlp_dann_subject_norm_epoch_log.txt")
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
        description="Train MLP + DANN with per-subject normalization on SEED (LOSO split)."
    )
    parser.add_argument("--lambda_dann", type=float, default=1.0,
                        help="Weight of the domain adversarial loss (default: 1.0)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device      : {DEVICE}")
    print(f"Model       : MLP + DANN + per-subject norm  |  Split: LOSO")
    print(f"lambda_dann : {args.lambda_dann}")
    print(f"Config      : max_epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"val_ratio={VAL_RATIO}  lr={LR}")
    print(f"α schedule  : sigmoid  0 → 1  over {MAX_EPOCHS} epochs\n")

    run_dann_experiment(lambda_dann=args.lambda_dann, verbose=not args.quiet)


if __name__ == "__main__":
    main()
