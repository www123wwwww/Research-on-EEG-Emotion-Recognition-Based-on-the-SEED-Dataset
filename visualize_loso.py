"""
Visualization script for LOSO experiments.
Generates all figures for the report.

Usage:
    python3 visualize_loso.py --fig all     # generate all figures
    python3 visualize_loso.py --fig bar      # comparison bar chart
    python3 visualize_loso.py --fig heatmap  # per-subject heatmap
    python3 visualize_loso.py --fig lambda   # lambda sensitivity curves
    python3 visualize_loso.py --fig confusion  # confusion matrices
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SUBJECTS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
SUBJ_LABELS = [f"S{s}" for s in SUBJECTS]


def load_result(method, norm, lambda_da=1.0):
    fname = f"{method}_{norm}_lambda{lambda_da}_results.json"
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Comparison Bar Chart ─────────────────────────────────────────────

def plot_comparison_bar():
    methods = [
        ("MLP", "baseline", "mixed", 1.0),
        ("MLP + CORAL\n(λ=1.0)", "coral", "mixed", 1.0),
        ("MLP + MMD\n(λ=1.0)", "mmd", "mixed", 1.0),
        ("MLP + DANN\n(λ=1.0)", "dann", "mixed", 1.0),
        ("MLP*", "baseline", "per_subject", 1.0),
        ("MLP* + CORAL\n(λ=5.0)", "coral", "per_subject", 5.0),
        ("MLP* + MMD\n(λ=10.0)", "mmd", "per_subject", 10.0),
        ("MLP* + DANN\n(λ=5.0)", "dann", "per_subject", 5.0),
    ]

    labels = []
    means = []
    stds = []
    colors = []

    mix_color = "#4C72B0"
    ps_color = "#DD8452"

    for name, key, norm, lam in methods:
        r = load_result(key, norm, lambda_da=lam)
        if r is None:
            labels.append(name)
            means.append(0)
            stds.append(0)
            colors.append(mix_color if norm == "mixed" else ps_color)
            continue
        labels.append(name)
        means.append(r["mean_acc"] * 100)
        stds.append(r["std_acc"] * 100)
        colors.append(mix_color if norm == "mixed" else ps_color)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.7)

    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[0] * 0.3,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("LOSO Cross-Subject Emotion Recognition: All Methods Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=mix_color, label="Mixed normalization"),
                       Patch(facecolor=ps_color, label="Per-subject normalization")]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "comparison_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ── Figure 2: Per-Subject Accuracy Heatmap ──────────────────────────────────────

def plot_heatmap():
    key_methods = [
        ("MLP (mixed)", "baseline", "mixed", 1.0),
        ("MLP+CORAL (mixed, λ=5)", "coral", "mixed", 5.0),
        ("MLP+MMD (mixed, λ=1)", "mmd", "mixed", 1.0),
        ("MLP+DANN (mixed, λ=1)", "dann", "mixed", 1.0),
        ("MLP (per-subj)", "baseline", "per_subject", 1.0),
        ("MLP+MMD (per-subj, λ=10)", "mmd", "per_subject", 10.0),
        ("MLP+CORAL (per-subj, λ=5)", "coral", "per_subject", 5.0),
        ("MLP+DANN (per-subj, λ=5)", "dann", "per_subject", 5.0),
    ]

    data = []
    labels = []
    for name, method_key, norm, lam in key_methods:
        r = load_result(method_key, norm, lambda_da=lam)
        if r is not None:
            data.append([a * 100 for a in r["accs"]])
            labels.append(name)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=30, vmax=90)

    ax.set_xticks(np.arange(len(SUBJ_LABELS)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(SUBJ_LABELS, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(SUBJ_LABELS)):
            val = data[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xlabel("Test Subject", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    ax.set_title("Per-Subject Accuracy (%) — Heatmap", fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "heatmap_per_subject.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ── Figure 3: Lambda Sensitivity Curves ──────────────────────────────────────────

def plot_lambda_sensitivity():
    lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    methods = [("CORAL", "coral"), ("MMD", "mmd"), ("DANN", "dann")]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for (name, key), color, marker in zip(methods, colors, markers):
        means, stds = [], []
        for lam in lambdas:
            r = load_result(key, "per_subject", lambda_da=lam)
            if r:
                means.append(r["mean_acc"] * 100)
                stds.append(r["std_acc"] * 100)
            else:
                means.append(np.nan)
                stds.append(np.nan)
        valid_lambdas = [l for l, m in zip(lambdas, means) if not np.isnan(m)]
        valid_means = [m for m in means if not np.isnan(m)]
        valid_stds = [s for s, m in zip(stds, means) if not np.isnan(m)]
        ax.errorbar(valid_lambdas, valid_means, yerr=valid_stds, marker=marker, color=color,
                    label=name, capsize=4, linewidth=2, markersize=7)

    # Also add baseline horizontal line
    baseline_ps = load_result("baseline", "per_subject")
    if baseline_ps:
        ax.axhline(y=baseline_ps["mean_acc"] * 100, color="gray", linestyle="--",
                    linewidth=1.5, label=f"Baseline (per-subj) {baseline_ps['mean_acc']*100:.1f}%")

    ax.set_xlabel("Lambda", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Lambda Sensitivity Analysis (Per-Subject Normalization, Extended Range)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(65, 82)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "lambda_sensitivity.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ── Figure 4: Normalization Ablation Bar Chart ───────────────────────────────────

def plot_ablation_bar():
    baseline_mix = load_result("baseline", "mixed")
    baseline_ps = load_result("baseline", "per_subject")
    mmd_mix = load_result("mmd", "mixed")
    mmd_ps_best = load_result("mmd", "per_subject", lambda_da=10.0)
    coral_ps_best = load_result("coral", "per_subject", lambda_da=5.0)
    dann_ps_best = load_result("dann", "per_subject", lambda_da=5.0)

    labels = ["Mixed norm\nbaseline", "Per-subj norm\nbaseline",
              "Mixed norm\n+MMD(λ=1)", "Per-subj norm\n+MMD(λ=10)",
              "Per-subj norm\n+CORAL(λ=5)", "Per-subj norm\n+DANN(λ=5)"]
    values = [
        baseline_mix["mean_acc"] * 100 if baseline_mix else 0,
        baseline_ps["mean_acc"] * 100 if baseline_ps else 0,
        mmd_mix["mean_acc"] * 100 if mmd_mix else 0,
        mmd_ps_best["mean_acc"] * 100 if mmd_ps_best else 0,
        coral_ps_best["mean_acc"] * 100 if coral_ps_best else 0,
        dann_ps_best["mean_acc"] * 100 if dann_ps_best else 0,
    ]
    stds = [
        baseline_mix["std_acc"] * 100 if baseline_mix else 0,
        baseline_ps["std_acc"] * 100 if baseline_ps else 0,
        mmd_mix["std_acc"] * 100 if mmd_mix else 0,
        mmd_ps_best["std_acc"] * 100 if mmd_ps_best else 0,
        coral_ps_best["std_acc"] * 100 if coral_ps_best else 0,
        dann_ps_best["std_acc"] * 100 if dann_ps_best else 0,
    ]

    colors = ["#4C72B0", "#55A868", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=stds, capsize=4, color=colors, edgecolor="white", width=0.7)

    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[0] * 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Normalization Ablation: What Contributes Most?",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ablation_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ── Figure 5: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices():
    from sklearn.metrics import confusion_matrix

    DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

    key_configs = [
        ("MLP baseline\n(mixed norm)", "baseline", "mixed", 1.0),
        ("MLP+MMD\n(mixed norm)", "mmd", "mixed", 1.0),
        ("MLP baseline\n(per-subject norm)", "baseline", "per_subject", 1.0),
        ("MLP+MMD\n(per-subject norm)", "mmd", "per_subject", 1.0),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    class_names = ["Negative", "Neutral", "Positive"]

    for ax, (title, method_key, norm, lam) in zip(axes, key_configs):
        all_y_true = []
        all_y_pred = []

        if norm == "mixed":
            from src.data_loader import get_loso_splits
            splits = get_loso_splits(normalize=True)
        else:
            from src.data_loader import get_loso_splits_subject_norm
            splits = get_loso_splits_subject_norm()

        for split in splits:
            X_te = get_flat_features(split["X_test"])
            y_te = split["y_test"]
            X_tr_all = get_flat_features(split["X_train"])
            y_tr_all = split["y_train"]

            ind_path = os.path.join(RESULTS_DIR,
                                     f"{method_key}_{norm}_lambda{lam}_results.json")
            if not os.path.exists(ind_path):
                continue

            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            if method_key == "baseline":
                from src.models.baseline import MLPModel
                model = MLPModel().to(torch.device("cpu"))
                state_key_prefix = ""
            elif method_key == "mmd":
                from src.domain_adaptation.mmd_model import MmdMLP
                model = MmdPLP().to(torch.device("cpu")) if False else MmdMLP().to(torch.device("cpu"))
                state_key_prefix = ""
            else:
                continue

            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long)),
                batch_size=256, shuffle=False
            )

            model.eval()
            preds = []
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    if method_key == "baseline":
                        out = model(X_b)
                    elif method_key in ("mmd", "coral"):
                        out, _ = model(X_b)
                    elif method_key == "dann":
                        out, _, _ = model(X_b, alpha=0.0)
                    preds.extend(out.argmax(1).numpy())
            all_y_pred.extend(preds)
            all_y_true.extend(y_te if isinstance(y_te, list) else y_te.tolist() if hasattr(y_te, 'tolist') else list(y_te))

        if len(all_y_true) == 0:
            ax.set_title(title + "\n(no predictions)")
            continue

        cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")

        for i in range(3):
            for j in range(3):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, f"{cm_pct[i, j]:.0f}%\n({cm[i, j]})",
                       ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ── Figure 6: t-SNE Visualization ────────────────────────────────────────────────

def plot_tsne():
    from sklearn.manifold import TSNE
    from src.data_loader import get_loso_splits, get_loso_splits_subject_norm, get_flat_features, AVAILABLE_SUBJECTS
    import torch
    from src.models.baseline import MLPModel
    from src.domain_adaptation.mmd_model import MmdMLP
    from src.domain_adaptation.dann_model import DannMLP

    SEED_VAL = 42

    configs = [
        ("Raw features\n(mixed norm)", None, "mixed", None),
        ("MLP features\n(mixed norm)", "baseline", "mixed", None),
        ("MLP+MMD features\n(mixed norm)", "mmd", "mixed", None),
        ("MLP+MMD features\n(per-subj norm)", "mmd", "per_subject", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for ax, (title, method_key, norm, _) in zip(axes, configs):
        if norm == "mixed":
            splits = get_loso_splits(normalize=True)
        else:
            splits = get_loso_splits_subject_norm()

        split = splits[0]  # test on subject 1
        X_tr_all = get_flat_features(split["X_train"])
        X_te = get_flat_features(split["X_test"])
        y_tr_all = split["y_train"]
        y_te = split["y_test"]

        domain_labels = np.array([0] * len(X_tr_all) + [1] * len(X_te))

        if method_key is None:
            features = np.concatenate([X_tr_all, X_te], axis=0)
            emotion_labels = np.concatenate([y_tr_all, y_te])
        else:
            result_path = os.path.join(RESULTS_DIR, f"{method_key}_{norm}_lambda1.0_results.json")
            if not os.path.exists(result_path):
                ax.set_title(title + "\n(no model)")
                continue

            device = torch.device("cpu")
            if method_key == "baseline":
                model = MLPModel().to(device)
            elif method_key == "mmd":
                model = MmdMLP().to(device)
            elif method_key == "dann":
                model = DannMLP().to(device)

            all_features = []
            all_labels = []

            loader = DataLoader(
                TensorDataset(torch.tensor(X_tr_all), torch.tensor(y_tr_all, dtype=torch.long)),
                batch_size=256, shuffle=False
            )
            model.eval()
            with torch.no_grad():
                for X_b, y_b in loader:
                    X_b = X_b.to(device)
                    if method_key == "baseline":
                        _ = model(X_b)
                        feat = model.net[:-1](X_b) if hasattr(model, 'net') else None
                        if feat is None:
                            feat = model(X_b)
                    elif method_key in ("mmd", "coral"):
                        _, feat = model(X_b)
                    elif method_key == "dann":
                        _, _, feat = model(X_b, alpha=0.0)
                    all_features.append(feat.cpu().numpy())
                    all_labels.extend(y_b.numpy())

            X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
            with torch.no_grad():
                if method_key == "baseline":
                    feat = model.net[:-1](X_te_t) if hasattr(model, 'net') else model(X_te_t)
                elif method_key in ("mmd", "coral"):
                    _, feat = model(X_te_t)
                elif method_key == "dann":
                    _, _, feat = model(X_te_t, alpha=0.0)
            all_features.append(feat.cpu().numpy())
            all_labels.extend(y_te.tolist() if hasattr(y_te, 'tolist') else list(y_te))

            features = np.concatenate(all_features, axis=0)
            emotion_labels = np.array(all_labels)
            domain_labels = np.array([0] * len(y_tr_all) + [1] * len(y_te))

        n_source = min(2000, len(features))
        rng = np.random.RandomState(SEED_VAL)
        idx = rng.choice(len(features), n_source, replace=False)

        tsne = TSNE(n_components=2, random_state=SEED_VAL, perplexity=30)
        emb = tsne.fit_transform(features[idx])

        domain_colors = ["#4C72B0" if domain_labels[idx][i] == 0 else "#DD8452"
                         for i in range(n_source)]
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=domain_colors, s=5, alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0',
                              markersize=8, label='Source domain'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#DD8452',
                              markersize=8, label='Target domain')]
    axes[0].legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.suptitle("t-SNE: Domain Alignment Visualization (Test Subject 1)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "tsne_visualization.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", choices=["bar", "heatmap", "lambda", "ablation",
                                           "confusion", "tsne", "all"],
                       default="all")
    args = parser.parse_args()

    if args.fig in ("bar", "all"):
        plot_comparison_bar()
    if args.fig in ("heatmap", "all"):
        plot_heatmap()
    if args.fig in ("lambda", "all"):
        plot_lambda_sensitivity()
    if args.fig in ("ablation", "all"):
        plot_ablation_bar()
    if args.fig in ("confusion", "all"):
        try:
            plot_confusion_matrices()
        except Exception as e:
            print(f"Warning: confusion matrix failed: {e}")
            print("Skipping confusion matrix (requires re-running models)")
    if args.fig in ("tsne", "all"):
        try:
            plot_tsne()
        except Exception as e:
            print(f"Warning: t-SNE failed: {e}")
            print("Skipping t-SNE (requires re-running models)")

    print("\nDone! Figures saved to:", FIG_DIR)