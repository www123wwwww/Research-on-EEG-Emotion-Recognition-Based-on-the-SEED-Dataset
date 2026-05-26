"""
Visualisation for Cross-Session CORAL (Subject-Dependent).

Generates two figures in results/cross_session/
(new filenames — does NOT overwrite subject_dependent_full.png or
subject_dependent_delta.png):

  1. subject_dependent_with_coral.png
       6-method grouped bar chart.  Baseline + plain cross-session methods
       carried over from existing results; CORAL method added as the 6th bar.

  2. subject_dependent_coral_delta.png
       Two-panel delta line chart per fold:
         Top panel   : CORAL total gain vs MLP baseline
                       + plain Cross-Session gain vs MLP baseline (for context)
         Bottom panel: CORAL-only alignment gain
                       = (MLP+CS+CORAL) − (MLP+CS)
                       isolates the domain-adaptation contribution

Usage:
    python src/cross_session/visualize_cross_session_coral.py
    python src/cross_session/visualize_cross_session_coral.py --show
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


# ── Figure 1: 6-method bar chart ─────────────────────────────────────────────

def plot_bar(save=True, show=False):
    import matplotlib.pyplot as plt

    entries = [
        # (label, filename, folder, color)
        ("SVM\n(baseline)",          "subject_dependent_svm_results.npy",                     RESULTS_DIR,    "#4C72B0"),
        ("MLP\n(baseline)",          "subject_dependent_mlp_results.npy",                     RESULTS_DIR,    "#55A868"),
        ("MLP+\nAttention",          "subject_dependent_attn_mlp_results.npy",                RESULTS_DIR,    "#C44E52"),
        ("SVM+\nCross-Sess",         "subject_dependent_svm_cross_session_results.npy",       RESULTS_CS_DIR, "#8172B2"),
        ("MLP+\nCross-Sess",         "subject_dependent_mlp_cross_session_results.npy",       RESULTS_CS_DIR, "#E377C2"),
        ("MLP+CS\n+CORAL",           "subject_dependent_mlp_cross_session_coral_results.npy", RESULTS_CS_DIR, "#D55E00"),
    ]

    labels, means, stds, all_accs, colors = [], [], [], [], []
    for label, fname, folder, color in entries:
        r = _load(os.path.join(folder, fname))
        labels.append(label)
        means.append(r["mean_acc"] * 100 if r else 0)
        stds.append (r["std_acc"]  * 100 if r else 0)
        all_accs.append(r["accs"]  * 100 if r else np.array([]))
        colors.append(color)

    x   = np.arange(len(labels))
    rng = np.random.RandomState(0)

    fig, ax = plt.subplots(figsize=(12, 5))
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

    # Bracket over cross-session methods (bars 3–5)
    cs_left  = x[3] - 0.35
    cs_right = x[5] + 0.35
    bracket_y = max(means[3:]) + max(stds[3:]) + 5.5
    ax.annotate("", xy=(cs_right, bracket_y), xytext=(cs_left, bracket_y),
                arrowprops=dict(arrowstyle="-", color="#555", lw=1.5))
    ax.text((cs_left + cs_right) / 2, bracket_y + 0.5,
            "Cross-Session Methods", ha="center",
            fontsize=9, color="#555", fontstyle="italic")

    # Bracket over CORAL vs plain MLP+CS (bars 4–5)
    da_left  = x[4] - 0.30
    da_right = x[5] + 0.30
    da_y     = max(means[4], means[5]) + max(stds[4], stds[5]) + 1.5
    ax.annotate("", xy=(da_right, da_y), xytext=(da_left, da_y),
                arrowprops=dict(arrowstyle="-", color="#D55E00", lw=1.2))
    coral_diff = means[5] - means[4]
    ax.text((da_left + da_right) / 2, da_y + 0.4,
            f"CORAL Δ={'+'if coral_diff>=0 else ''}{coral_diff:.1f}%",
            ha="center", fontsize=8, color="#D55E00", fontstyle="italic")

    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8,
               label="Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Subject-Dependent Accuracy (%)", fontsize=12)
    ax.set_title("Subject-Dependent Emotion Recognition — SEED Dataset\n"
                 "Effect of Cross-Session Domain Adaptation (CORAL)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_CS_DIR, "subject_dependent_with_coral.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    if show:
        plt.show()
    return fig


# ── Figure 2: two-panel delta line chart ─────────────────────────────────────

def plot_delta(save=True, show=False):
    import matplotlib.pyplot as plt
    from src.data_loader import AVAILABLE_SUBJECTS, SESSIONS

    mlp      = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    mlp_cs   = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))
    mlp_cs_da= _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_coral_results.npy"))

    if mlp is None:
        print("[skip] Baseline MLP results not found.")
        return None
    if mlp_cs_da is None:
        print("[skip] CORAL results not found — run train_cross_session_coral.py first.")
        return None

    n_folds     = len(mlp["accs"])
    folds       = np.arange(1, n_folds + 1)
    fold_labels = [f"s{s}\nss{t}" for s in AVAILABLE_SUBJECTS for t in SESSIONS]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    # ── Top panel: total gain vs MLP baseline ────────────────────────────────
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if mlp_cs is not None:
        d_cs = (mlp_cs["accs"] - mlp["accs"]) * 100
        ax1.plot(folds, d_cs, marker="s", color="#E377C2", linewidth=1.6,
                 markersize=5, label="MLP+CrossSession − MLP  (transfer effect)", zorder=2)
        ax1.fill_between(folds, d_cs, 0,
                         where=(d_cs >= 0), alpha=0.10, color="#E377C2")
        ax1.fill_between(folds, d_cs, 0,
                         where=(d_cs <  0), alpha=0.08, color="#C44E52")

    d_da = (mlp_cs_da["accs"] - mlp["accs"]) * 100
    ax1.plot(folds, d_da, marker="o", color="#D55E00", linewidth=2.0,
             markersize=6, label="MLP+CS+CORAL − MLP  (transfer + DA effect)", zorder=3)
    ax1.fill_between(folds, d_da, 0,
                     where=(d_da >= 0), alpha=0.13, color="#D55E00")
    ax1.fill_between(folds, d_da, 0,
                     where=(d_da <  0), alpha=0.08, color="#C44E52")

    ax1.set_ylabel("Accuracy Δ vs MLP baseline (%)", fontsize=11)
    ax1.set_title("Per-fold Improvement: Cross-Session Transfer vs CORAL-Augmented Transfer",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Bottom panel: CORAL-only contribution ────────────────────────────────
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if mlp_cs is not None:
        d_coral_only = (mlp_cs_da["accs"] - mlp_cs["accs"]) * 100
        pos = d_coral_only >= 0
        ax2.bar(folds[pos],  d_coral_only[pos],  color="#D55E00", alpha=0.7, width=0.6,
                label="CORAL improves this fold")
        ax2.bar(folds[~pos], d_coral_only[~pos], color="#C44E52", alpha=0.7, width=0.6,
                label="CORAL hurts this fold")
        ax2.plot(folds, d_coral_only, color="#D55E00", linewidth=1.4,
                 marker="o", markersize=4, zorder=3)

        n_pos = pos.sum()
        ax2.text(0.02, 0.92,
                 f"CORAL helps: {n_pos}/{n_folds} folds  |  "
                 f"mean Δ={'+'if d_coral_only.mean()>=0 else ''}{d_coral_only.mean():.2f}%",
                 transform=ax2.transAxes, fontsize=9, color="#333")
    else:
        ax2.text(0.5, 0.5, "MLP+CrossSession results not found",
                 transform=ax2.transAxes, ha="center", fontsize=10, color="gray")

    ax2.set_ylabel("CORAL-only Δ (%)\n(vs MLP+CrossSession)", fontsize=11)
    ax2.set_xlabel("Fold (subject × session)", fontsize=11)
    ax2.set_xticks(folds)
    ax2.set_xticklabels(fold_labels, fontsize=7)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_CS_DIR, "subject_dependent_coral_delta.png")
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

    plot_bar  (save=True, show=args.show)
    plot_delta(save=True, show=args.show)


if __name__ == "__main__":
    main()
