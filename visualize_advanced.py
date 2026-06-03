"""
Generate confusion matrices and t-SNE. Optimized to only train necessary folds.
- Confusion matrix: 5 evenly-spaced folds (1,4,7,10,13) × 4 configs
- t-SNE: fold 0 only × 4 configs
"""

import sys, os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.data_loader import (
    get_loso_splits, get_loso_splits_subject_norm, get_flat_features, AVAILABLE_SUBJECTS,
)
from src.models.baseline import MLPModel
from src.domain_adaptation.coral_model import CoralMLP
from src.domain_adaptation.coral_loss import coral_loss
from src.domain_adaptation.mmd_model import MmdMLP
from src.domain_adaptation.mmd_loss import mmd_loss
from src.domain_adaptation.dann_model import DannMLP

SEED = 42; MAX_EPOCHS = 20; BATCH_SIZE = 128; LR = 5e-4
WEIGHT_DECAY = 1e-4; VAL_RATIO = 0.2; DEVICE = torch.device("cpu")
RESULTS_DIR = "/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult"
FIG_DIR = os.path.join(RESULTS_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
CLASS_NAMES = ["Negative", "Neutral", "Positive"]


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


def _loader(X, y=None, bs=128, shuffle=False):
    if y is not None:
        ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    else:
        ds = TensorDataset(torch.tensor(X))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


def _get_alpha(epoch, max_epochs):
    p = min(epoch / max_epochs, 1.0)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def extract_features(model, method, X):
    model.eval()
    with torch.no_grad():
        X_d = X.to(DEVICE) if not isinstance(X, torch.Tensor) else X.to(DEVICE)
        if method == "baseline":
            return model.net[:8](X_d).cpu().numpy()
        elif method in ("coral", "mmd"):
            _, feat = model(X_d)
            return feat.cpu().numpy()
        else:
            _, _, feat = model(X_d, 0.0)
            return feat.cpu().numpy()


def predict(model, method, X):
    model.eval()
    with torch.no_grad():
        X_d = X.to(DEVICE) if not isinstance(X, torch.Tensor) else X.to(DEVICE)
        if method == "baseline":
            return model(X_d).argmax(1).cpu().numpy()
        elif method in ("coral", "mmd"):
            logits, _ = model(X_d)
            return logits.argmax(1).cpu().numpy()
        else:
            logits, _, _ = model(X_d, 0.0)
            return logits.argmax(1).cpu().numpy()


def train_one_fold(method, norm, split, fold_idx):
    """Train a single fold and return predictions + features."""
    X_tr_all = get_flat_features(split["X_train"])
    X_te = get_flat_features(split["X_test"])
    y_tr_all, y_te = split["y_train"], split["y_test"]

    X_tr, y_tr, X_val, y_val = _stratified_split(X_tr_all, y_tr_all)
    src_loader = _loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader = _loader(X_val, y_val, 256)
    tgt_loader = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    if method == "baseline":
        model = MLPModel().to(DEVICE)
    elif method == "coral":
        model = CoralMLP().to(DEVICE)
    elif method == "mmd":
        model = MmdMLP().to(DEVICE)
    else:
        model = DannMLP().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    label_crit = nn.CrossEntropyLoss()
    domain_crit = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tgt_iter = iter(tgt_loader)
        for X_src, y_src in src_loader:
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            optimizer.zero_grad()
            if method == "baseline":
                loss = label_crit(model(X_src), y_src)
            elif method in ("coral", "mmd"):
                try:
                    (X_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader); (X_tgt,) = next(tgt_iter)
                X_tgt = X_tgt.to(DEVICE)
                logits_s, feat_s = model(X_src)
                _, feat_t = model(X_tgt)
                da_fn = coral_loss if method == "coral" else mmd_loss
                loss = label_crit(logits_s, y_src) + da_fn(feat_s, feat_t)
            else:
                alpha = _get_alpha(epoch, MAX_EPOCHS)
                try:
                    (X_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader); (X_tgt,) = next(tgt_iter)
                X_tgt = X_tgt.to(DEVICE)
                n_s, n_t = X_src.size(0), X_tgt.size(0)
                d_s = torch.zeros(n_s, dtype=torch.long, device=DEVICE)
                d_t = torch.ones(n_t, dtype=torch.long, device=DEVICE)
                lab_s, dom_s, _ = model(X_src, alpha)
                _, dom_t, _ = model(X_tgt, alpha)
                loss = label_crit(lab_s, y_src) + domain_crit(dom_s, d_s) + domain_crit(dom_t, d_t)

            loss.backward()
            optimizer.step()
        scheduler.step()

        # validation
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

    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    fold_preds = predict(model, method, X_te_t)
    fold_feats = extract_features(model, method, X_te_t)

    # also extract source features (subsample)
    rng = np.random.RandomState(SEED)
    n_src = min(1000, len(X_tr_all))
    src_idx = rng.choice(len(X_tr_all), n_src, replace=False)
    X_src_sub = torch.tensor(X_tr_all[src_idx], dtype=torch.float32, device=DEVICE)
    src_feats = extract_features(model, method, X_src_sub)

    return {
        "preds": fold_preds,
        "truths": y_te,
        "src_feats": src_feats,
        "src_labels": y_tr_all[src_idx],
        "tgt_feats": fold_feats,
        "tgt_labels": y_te,
        "test_subject": split["test_subject"],
        "acc": (fold_preds == y_te).mean(),
    }


def plot_confusion_matrices(configs_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax_idx, (title, all_preds, all_truths) in enumerate(configs_results):
        cm = confusion_matrix(all_truths, all_preds, labels=[0, 1, 2])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        ax = axes[ax_idx]
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, f"{cm_pct[i,j]:.0f}%\n({cm[i,j]})",
                       ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices (5 Representative Folds)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_tsne(tsne_data):
    n = len(tsne_data)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]

    src_color, tgt_color = "#4C72B0", "#DD8452"

    for ax_idx, (title, info) in enumerate(tsne_data):
        ax = axes[ax_idx]
        feats = np.concatenate([info["src_feats"], info["tgt_feats"]], axis=0)
        domain = np.array([0]*len(info["src_labels"]) + [1]*len(info["tgt_labels"]))

        print(f"  t-SNE for '{title}' on {feats.shape[0]} samples, dim={feats.shape[1]}...")
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(feats)-1), max_iter=1000)
        emb = tsne.fit_transform(feats)

        for d_val, color, dlabel in [(0, src_color, "Source"), (1, tgt_color, "Target")]:
            mask = domain == d_val
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=5, alpha=0.4, label=dlabel)

        acc_tgt = (info["tgt_labels"] == np.array([0]*len(info["tgt_labels"]))).mean()  # dummy
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend(fontsize=9, loc="upper right")
    plt.suptitle("t-SNE: Source vs Target Feature Distribution (Subject 1 as Target)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "tsne_visualization.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", choices=["confusion", "tsne", "both"], default="both")
    args = parser.parse_args()

    # Only need fold 0 for t-SNE
    CM_FOLDS = [0, 5, 11] if args.fig in ("confusion", "both") else []
    TSNE_FOLD = 0

    configs = [
        ("MLP baseline\n(mixed norm)", "baseline", "mixed"),
        ("MLP+MMD\n(mixed norm)", "mmd", "mixed"),
        ("MLP baseline\n(per-subj norm)", "baseline", "per_subject"),
        ("MLP+MMD\n(per-subj norm)", "mmd", "per_subject"),
    ]

    cm_results = []
    tsne_data = []

    for title, method, norm in configs:
        splits = get_loso_splits(normalize=True) if norm == "mixed" else get_loso_splits_subject_norm()
        print(f"\n{'='*60}")
        print(f"Training: {method} ({norm})")
        print(f"{'='*60}")

        all_preds, all_truths = [], []

        fold_indices = sorted(set(CM_FOLDS + [TSNE_FOLD]))
        for fi in fold_indices:
            print(f"  Fold {fi+1}/15 (subject={splits[fi]['test_subject']})...")
            result = train_one_fold(method, norm, splits[fi], fi)
            print(f"    acc={result['acc']:.4f}")

            if fi in CM_FOLDS and args.fig in ("confusion", "both"):
                all_preds.extend(result["preds"].tolist())
                all_truths.extend(result["truths"].tolist())

            if fi == TSNE_FOLD and args.fig in ("tsne", "both"):
                tsne_data.append((title, {
                    "src_feats": result["src_feats"],
                    "src_labels": result["src_labels"],
                    "tgt_feats": result["tgt_feats"],
                    "tgt_labels": result["tgt_labels"],
                }))

        if args.fig in ("confusion", "both"):
            cm_results.append((title, all_preds, all_truths))

    if args.fig in ("confusion", "both") and cm_results:
        print("\nGenerating confusion matrices...")
        plot_confusion_matrices(cm_results)

    if args.fig in ("tsne", "both") and tsne_data:
        print("\nGenerating t-SNE...")
        # Add raw features as first panel
        splits = get_loso_splits(normalize=True)
        split = splits[0]
        X_src_raw = get_flat_features(split["X_train"])
        X_tgt_raw = get_flat_features(split["X_test"])
        rng = np.random.RandomState(SEED)
        n_src = min(1000, len(X_src_raw))
        src_idx = rng.choice(len(X_src_raw), n_src, replace=False)
        raw_info = {
            "src_feats": X_src_raw[src_idx],
            "src_labels": split["y_train"][src_idx],
            "tgt_feats": X_tgt_raw,
            "tgt_labels": split["y_test"],
        }
        tsne_data.insert(0, ("Raw features\n(mixed norm)", raw_info))
        plot_tsne(tsne_data)

    print("\nDone!")