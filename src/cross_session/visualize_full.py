"""
Full Subject-Dependent visualisation: all 5 methods.

Generates in results/cross_session/:
  1. subject_dependent_full.png
       Bar chart — 5 methods grouped by "baseline" vs "cross-session",
       with error bars and per-fold scatter overlay.

  2. subject_dependent_delta.png
       Delta line chart — SVM+CS and MLP+CS improvement over their
       respective baselines, fold by fold, to show which subjects benefit.

Usage:
    python src/cross_session/visualize_full.py
    python src/cross_session/visualize_full.py --show
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


# ── Figure 1: grouped bar chart ───────────────────────────────────────────────

def plot_bar_full(save=True, show=False):
    import matplotlib.pyplot as plt

    entries = [
        # (label, filename, folder, color)
        ("SVM\n(baseline)",         "subject_dependent_svm_results.npy",               RESULTS_DIR,    "#4C72B0"),
        ("MLP\n(baseline)",         "subject_dependent_mlp_results.npy",               RESULTS_DIR,    "#55A868"),
        ("MLP+\nAttention",         "subject_dependent_attn_mlp_results.npy",          RESULTS_DIR,    "#C44E52"),
        ("SVM+\nCross-Session",     "subject_dependent_svm_cross_session_results.npy", RESULTS_CS_DIR, "#8172B2"),
        ("MLP+\nCross-Session",     "subject_dependent_mlp_cross_session_results.npy", RESULTS_CS_DIR, "#E377C2"),
    ]

    labels, means, stds, all_accs, colors = [], [], [], [], []
    for label, fname, folder, color in entries:
        r = _load(os.path.join(folder, fname))
        labels.append(label)
        means.append(r["mean_acc"] * 100 if r else 0)
        stds.append (r["std_acc"]  * 100 if r else 0)
        all_accs.append(r["accs"] * 100   if r else np.array([]))
        colors.append(color)

    x   = np.arange(len(labels))
    rng = np.random.RandomState(0)

    fig, ax = plt.subplots(figsize=(10, 5))
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

    # Bracket over the two cross-session methods
    cs_left  = x[3] - 0.35
    cs_right = x[4] + 0.35
    bracket_y = max(means[3], means[4]) + max(stds[3], stds[4]) + 5
    ax.annotate("", xy=(cs_right, bracket_y), xytext=(cs_left, bracket_y),
                arrowprops=dict(arrowstyle="-", color="#555", lw=1.5))
    ax.text((cs_left + cs_right) / 2, bracket_y + 0.5,
            "Cross-Session", ha="center",
            fontsize=9, color="#555", fontstyle="italic")

    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8,
               label="Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Subject-Dependent Accuracy (%)", fontsize=12)
    ax.set_title("Subject-Dependent Emotion Recognition — SEED Dataset\n"
                 "Data Effect (SVM+CS) vs Method Effect (MLP+CS)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_CS_DIR, "subject_dependent_full.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    if show:
        plt.show()
    return fig


# ── Figure 2: per-fold delta chart ────────────────────────────────────────────

def plot_delta(save=True, show=False):
    import matplotlib.pyplot as plt

    svm    = _load(os.path.join(RESULTS_DIR,    "subject_dependent_svm_results.npy"))
    mlp    = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    svm_cs = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_svm_cross_session_results.npy"))
    mlp_cs = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))

    if svm is None and mlp is None:
        print("[skip] Baseline results not found.")
        return None

    n_folds = len(svm["accs"]) if svm else len(mlp["accs"])
    folds   = np.arange(1, n_folds + 1)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if svm is not None and svm_cs is not None:
        delta = (svm_cs["accs"] - svm["accs"]) * 100
        ax.plot(folds, delta, marker="o", color="#8172B2", linewidth=1.8,
                markersize=5, label="SVM+CrossSession − SVM  (data effect)")
        ax.fill_between(folds, delta, 0,
                        where=(delta >= 0), alpha=0.13, color="#8172B2")
        ax.fill_between(folds, delta, 0,
                        where=(delta <  0), alpha=0.10, color="#C44E52")

    if mlp is not None and mlp_cs is not None:
        delta = (mlp_cs["accs"] - mlp["accs"]) * 100
        ax.plot(folds, delta, marker="s", color="#E377C2", linewidth=1.8,
                markersize=5, label="MLP+CrossSession − MLP  (data + method effect)")
        ax.fill_between(folds, delta, 0,
                        where=(delta >= 0), alpha=0.13, color="#E377C2")
        ax.fill_between(folds, delta, 0,
                        where=(delta <  0), alpha=0.10, color="#C44E52")

    # Build fold labels from existing subject/session order
    from src.data_loader import AVAILABLE_SUBJECTS, SESSIONS
    fold_labels = [f"s{s}\nss{t}" for s in AVAILABLE_SUBJECTS for t in SESSIONS]
    ax.set_xticks(folds)
    ax.set_xticklabels(fold_labels, fontsize=7)
    ax.set_xlabel("Fold (subject × session)", fontsize=11)
    ax.set_ylabel("Accuracy Δ vs baseline (%)", fontsize=11)
    ax.set_title("Per-fold Improvement: Data Effect vs Method Effect",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_CS_DIR, "subject_dependent_delta.png")
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

    plot_bar_full(save=True, show=args.show)
    plot_delta   (save=True, show=args.show)


if __name__ == "__main__":
    main()
