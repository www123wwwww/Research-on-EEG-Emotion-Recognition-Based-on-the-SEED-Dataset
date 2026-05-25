"""
Visualisation: compare all LOSO methods including MLP + CORAL.

Generates:
  results/domain_adaptation/loso_comparison_coral.png
    — grouped bar chart of all four methods on the LOSO task,
      with error bars (±1 std) and per-fold accuracy scatter overlay.

Usage:
    python src/domain_adaptation/visualize_coral.py
    python src/domain_adaptation/visualize_coral.py --show   # display interactively
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


def plot_loso_comparison(save: bool = True, show: bool = False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    entries = [
        ("SVM\n(baseline)",      "loso_svm_results.npy",      RESULTS_DIR,    "#4C72B0"),
        ("MLP\n(baseline)",      "loso_mlp_results.npy",      RESULTS_DIR,    "#55A868"),
        ("MLP+\nAttention",      "loso_attn_mlp_results.npy", RESULTS_DIR,    "#C44E52"),
        ("MLP+\nCORAL (ours)",   "loso_mlp_coral_results.npy",RESULTS_DA_DIR, "#8172B2"),
    ]

    labels, means, stds, all_accs, colors = [], [], [], [], []
    for label, fname, folder, color in entries:
        r = _load(os.path.join(folder, fname))
        labels.append(label)
        means.append(r["mean_acc"] * 100 if r else 0)
        stds.append(r["std_acc"]   * 100 if r else 0)
        all_accs.append(r["accs"]  * 100 if r else np.array([]))
        colors.append(color)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=colors, edgecolor="white", linewidth=1.2,
                  error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))

    # Per-fold scatter overlay (shows variance distribution)
    rng = np.random.RandomState(0)
    for i, accs in enumerate(all_accs):
        if len(accs) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(accs))
        ax.scatter(x[i] + jitter, accs, s=18, color="white",
                   edgecolors=colors[i], linewidths=0.8,
                   alpha=0.85, zorder=3)

    # Annotate mean values on top of bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.8,
                f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Highlight the CORAL bar with a bounding box
    coral_bar = bars[3]
    ax.annotate("Domain\nAdaptation",
                xy=(coral_bar.get_x() + coral_bar.get_width() / 2,
                    coral_bar.get_height() + stds[3] + 3.5),
                ha="center", fontsize=8.5, color="#8172B2",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("LOSO Accuracy (%)", fontsize=12)
    ax.set_title("Cross-Subject (LOSO) Emotion Recognition on SEED\n"
                 "Baseline vs Domain Adaptation (CORAL)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8, label="Chance (33.3%)")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DA_DIR, "loso_comparison_coral.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    plot_loso_comparison(save=True, show=args.show)


if __name__ == "__main__":
    main()
