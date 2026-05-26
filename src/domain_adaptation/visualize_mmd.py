"""
Comprehensive visualisation: all LOSO methods (baselines + CORAL + MMD).

Generates two figures in results/domain_adaptation/:
  1. loso_comparison_mmd.png  — bar chart for all 5 methods with error bars
                                 and per-fold scatter overlay
  2. loso_da_delta.png        — delta bar chart showing improvement of CORAL
                                 and MMD over the MLP baseline, fold by fold

Usage:
    python src/domain_adaptation/visualize_mmd.py
    python src/domain_adaptation/visualize_mmd.py --show
"""

import sys
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

RESULTS_DIR    = os.path.join(_ROOT, "results")
RESULTS_DA_DIR = os.path.join(_ROOT, "results", "domain_adaptation")


def _load(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: full comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_methods(save=True, show=False):
    import matplotlib.pyplot as plt

    entries = [
        ("SVM\n(baseline)",    "loso_svm_results.npy",      RESULTS_DIR,    "#4C72B0"),
        ("MLP\n(baseline)",    "loso_mlp_results.npy",      RESULTS_DIR,    "#55A868"),
        ("MLP+\nAttention",    "loso_attn_mlp_results.npy", RESULTS_DIR,    "#C44E52"),
        ("MLP+\nCORAL",        "loso_mlp_coral_results.npy",RESULTS_DA_DIR, "#8172B2"),
        ("MLP+\nMMD (ours)",   "loso_mlp_mmd_results.npy",  RESULTS_DA_DIR, "#CCB974"),
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

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=colors, edgecolor="white", linewidth=1.2,
                  error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))

    # Per-fold scatter
    for i, accs in enumerate(all_accs):
        if len(accs) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(accs))
        ax.scatter(x[i] + jitter, accs, s=18, color="white",
                   edgecolors=colors[i], linewidths=0.8, alpha=0.85, zorder=3)

    # Annotate mean values
    for bar, mean, std in zip(bars, means, stds):
        if mean == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.8,
                f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Bracket highlighting the two DA methods
    da_left  = x[3] - 0.35
    da_right = x[4] + 0.35
    bracket_y = max(means[3], means[4]) + max(stds[3], stds[4]) + 5
    ax.annotate("", xy=(da_right, bracket_y), xytext=(da_left, bracket_y),
                arrowprops=dict(arrowstyle="-", color="#555", lw=1.5))
    ax.text((da_left + da_right) / 2, bracket_y + 0.5,
            "Domain Adaptation", ha="center", fontsize=9,
            color="#555", fontstyle="italic")

    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8, label="Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("LOSO Accuracy (%)", fontsize=12)
    ax.set_title("Cross-Subject (LOSO) Emotion Recognition — SEED Dataset\n"
                 "Baselines vs Domain Adaptation Methods",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 85)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DA_DIR, "loso_comparison_mmd.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: per-fold delta chart (DA methods vs MLP baseline)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fold_delta(save=True, show=False):
    import matplotlib.pyplot as plt

    mlp   = _load(os.path.join(RESULTS_DIR,    "loso_mlp_results.npy"))
    coral = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_coral_results.npy"))
    mmd   = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_mmd_results.npy"))

    if mlp is None:
        print("[skip] MLP baseline result not found.")
        return None

    n_folds = len(mlp["accs"])
    folds   = np.arange(1, n_folds + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if coral is not None:
        delta_coral = (coral["accs"] - mlp["accs"]) * 100
        ax.plot(folds, delta_coral, marker="o", color="#8172B2",
                linewidth=1.8, markersize=6, label="MLP+CORAL − MLP")
        ax.fill_between(folds, delta_coral, 0,
                        where=(delta_coral >= 0), alpha=0.15, color="#8172B2")
        ax.fill_between(folds, delta_coral, 0,
                        where=(delta_coral < 0),  alpha=0.15, color="#C44E52")

    if mmd is not None:
        delta_mmd = (mmd["accs"] - mlp["accs"]) * 100
        ax.plot(folds, delta_mmd, marker="s", color="#CCB974",
                linewidth=1.8, markersize=6, label="MLP+MMD − MLP")
        ax.fill_between(folds, delta_mmd, 0,
                        where=(delta_mmd >= 0), alpha=0.12, color="#CCB974")
        ax.fill_between(folds, delta_mmd, 0,
                        where=(delta_mmd < 0),  alpha=0.12, color="#C44E52")

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Sub{i}" for i in folds], fontsize=9)
    ax.set_xlabel("Test Subject (LOSO fold)", fontsize=11)
    ax.set_ylabel("Accuracy Δ vs MLP (%)", fontsize=11)
    ax.set_title("Per-fold Improvement of Domain Adaptation over MLP Baseline",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DA_DIR, "loso_da_delta.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    plot_all_methods(save=True, show=args.show)
    plot_fold_delta(save=True,  show=args.show)


if __name__ == "__main__":
    main()
