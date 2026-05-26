"""
Visualisation: Subject-Dependent comparison including cross-session MLP.

Generates two figures in results/cross_session/:
  1. subject_dependent_comparison.png
       Grouped bar chart: all 4 methods × Subject-Dependent accuracy,
       with error bars and per-fold scatter overlay.

  2. subject_dependent_per_fold.png
       Line chart showing per-fold accuracy for SVM, MLP baseline, and
       cross-session MLP across all 36 (subject × session) folds,
       to show where the new method gains or loses.

Usage:
    python src/cross_session/visualize_cross_session.py
    python src/cross_session/visualize_cross_session.py --show
"""

import sys
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

RESULTS_DIR    = os.path.join(_ROOT, "results")
RESULTS_CS_DIR = os.path.join(_ROOT, "results", "cross_session")


def _load(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


# ── Figure 1: bar chart of all 4 methods ──────────────────────────────────────

def plot_bar_comparison(save=True, show=False):
    import matplotlib.pyplot as plt

    entries = [
        ("SVM\n(baseline)",          "subject_dependent_svm_results.npy",                RESULTS_DIR,    "#4C72B0"),
        ("MLP\n(baseline)",          "subject_dependent_mlp_results.npy",                RESULTS_DIR,    "#55A868"),
        ("MLP+\nAttention",          "subject_dependent_attn_mlp_results.npy",           RESULTS_DIR,    "#C44E52"),
        ("MLP+\nCross-Session",      "subject_dependent_mlp_cross_session_results.npy",  RESULTS_CS_DIR, "#8172B2"),
    ]

    labels, means, stds, all_accs, colors = [], [], [], [], []
    for label, fname, folder, color in entries:
        r = _load(os.path.join(folder, fname))
        labels.append(label)
        means.append(r["mean_acc"] * 100 if r else 0)
        stds.append(r["std_acc"]   * 100 if r else 0)
        all_accs.append(r["accs"]  * 100 if r else np.array([]))
        colors.append(color)

    x   = np.arange(len(labels))
    rng = np.random.RandomState(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=colors, edgecolor="white", linewidth=1.2,
                  error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))

    for i, accs in enumerate(all_accs):
        if len(accs) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(accs))
        ax.scatter(x[i] + jitter, accs, s=14, color="white",
                   edgecolors=colors[i], linewidths=0.7, alpha=0.8, zorder=3)

    for bar, mean, std in zip(bars, means, stds):
        if mean == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.8,
                f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8, label="Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Subject-Dependent Accuracy (%)", fontsize=12)
    ax.set_title("Subject-Dependent Emotion Recognition — SEED Dataset\n"
                 "Cross-Session Pre-training vs Baselines",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_CS_DIR, "subject_dependent_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    if show:
        plt.show()
    return fig


# ── Figure 2: per-fold line chart ─────────────────────────────────────────────

def plot_per_fold(save=True, show=False):
    import matplotlib.pyplot as plt

    svm = _load(os.path.join(RESULTS_DIR,    "subject_dependent_svm_results.npy"))
    mlp = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    cs  = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))

    if svm is None and mlp is None and cs is None:
        print("[skip] No results found yet.")
        return None

    # Build fold labels: sub1_sess1 … sub14_sess3
    from src.data_loader import AVAILABLE_SUBJECTS, SESSIONS
    fold_labels = [f"s{s}\nss{t}" for s in AVAILABLE_SUBJECTS for t in SESSIONS]
    x = np.arange(len(fold_labels))

    fig, ax = plt.subplots(figsize=(14, 4))

    for r, label, color, marker in [
        (svm, "SVM",              "#4C72B0", "o"),
        (mlp, "MLP baseline",     "#55A868", "s"),
        (cs,  "MLP+Cross-Session","#8172B2", "^"),
    ]:
        if r is None:
            continue
        ax.plot(x, r["accs"] * 100, marker=marker, markersize=5,
                linewidth=1.4, color=color, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Fold (subject × session)", fontsize=11)
    ax.set_title("Per-fold Subject-Dependent Accuracy Across All 36 Folds",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_CS_DIR, "subject_dependent_per_fold.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    if show:
        plt.show()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    plot_bar_comparison(save=True, show=args.show)
    plot_per_fold(save=True, show=args.show)


if __name__ == "__main__":
    main()
