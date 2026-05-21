"""
Visualisation for attention weights (Direction 1).

Generates two figures:
  1. Band attention bar chart  — which frequency band is most important
  2. Channel attention topomap — which electrodes are most important (brain heatmap)

Usage:
    python src/visualize.py --split subject_dependent   # uses attn_mlp results
    python src/visualize.py --split loso
    python src/visualize.py --split both                # generate all 4 plots
"""

import os
import sys
import argparse
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CHANNEL_POS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "dataset", "channel_62_pos.locs"
)
BANDS = ["delta\n(1-4Hz)", "theta\n(4-8Hz)", "alpha\n(8-13Hz)",
         "beta\n(13-30Hz)", "gamma\n(30-50Hz)"]
BAND_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def load_channel_positions():
    """
    Parse channel_62_pos.locs → arrays of (x, y, label) for 62 channels.
    Format: index  angle  radius  label
    Converted from polar to Cartesian for 2-D plotting.
    """
    xs, ys, labels = [], [], []
    with open(CHANNEL_POS_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            angle_deg = float(parts[1])
            radius    = float(parts[2])
            label     = parts[3]
            angle_rad = np.deg2rad(angle_deg)
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            xs.append(x)
            ys.append(y)
            labels.append(label)
    return np.array(xs), np.array(ys), labels


def plot_band_attention(band_weights: np.ndarray, split: str, save: bool = True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(BANDS, band_weights, color=BAND_COLORS, edgecolor="white",
                  linewidth=1.2, width=0.6)

    # annotate values
    for bar, val in zip(bars, band_weights):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(band_weights) * 1.2)
    ax.set_ylabel("Mean Attention Weight", fontsize=11)
    ax.set_title(f"Band Attention Weights ({split.replace('_', ' ').title()})",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"band_attention_{split}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    return fig


def plot_channel_attention(chan_weights: np.ndarray, split: str, save: bool = True):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    xs, ys, labels = load_channel_positions()

    # interpolate onto a grid for smooth heatmap
    xi = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 300)
    yi = np.linspace(ys.min() - 0.05, ys.max() + 0.05, 300)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xs, ys), chan_weights, (xi, yi), method="cubic")

    # mask outside head circle
    head_r = max(xs.max(), ys.max()) + 0.02
    dist = np.sqrt(xi**2 + yi**2)
    zi[dist > head_r] = np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    norm = Normalize(vmin=np.nanmin(zi), vmax=np.nanmax(zi))
    mesh = ax.contourf(xi, yi, zi, levels=64, cmap="RdYlBu_r", norm=norm)
    plt.colorbar(mesh, ax=ax, fraction=0.04, label="Attention Weight")

    # draw head outline
    circle = plt.Circle((0, 0), head_r, color="black", fill=False, linewidth=2)
    ax.add_patch(circle)
    # nose
    ax.plot([0, 0], [head_r, head_r + 0.04], color="black", linewidth=2)
    # ears
    ax.plot([-head_r - 0.02, -head_r - 0.02], [-0.04, 0.04], color="black", linewidth=2)
    ax.plot([ head_r + 0.02,  head_r + 0.02], [-0.04, 0.04], color="black", linewidth=2)

    # electrode dots
    sc = ax.scatter(xs, ys, c=chan_weights, cmap="RdYlBu_r", norm=norm,
                    s=60, edgecolors="black", linewidth=0.6, zorder=5)

    # label top-N channels
    top_n = 10
    top_idx = np.argsort(chan_weights)[-top_n:]
    for idx in top_idx:
        ax.annotate(labels[idx], (xs[idx], ys[idx]),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Channel Attention Weights ({split.replace('_', ' ').title()})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"channel_attention_{split}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    return fig


def plot_accuracy_comparison(save: bool = True):
    """Bar chart comparing all models on both split strategies."""
    import matplotlib.pyplot as plt

    models  = ["SVM\n(baseline)", "MLP\n(baseline)", "MLP+Attention\n(ours)"]
    keys    = ["svm", "mlp", "attn_mlp"]
    splits  = ["subject_dependent", "loso"]
    s_labels= ["Subject-Dependent", "LOSO"]
    colors  = ["#4C72B0", "#DD8452"]

    sd_accs, sd_stds = [], []
    lo_accs, lo_stds = [], []

    for k in keys:
        r_sd = _load(f"subject_dependent_{k}_results.npy")
        r_lo = _load(f"loso_{k}_results.npy")
        sd_accs.append(r_sd["mean_acc"] * 100 if r_sd else 0)
        sd_stds.append(r_sd["std_acc"]  * 100 if r_sd else 0)
        lo_accs.append(r_lo["mean_acc"] * 100 if r_lo else 0)
        lo_stds.append(r_lo["std_acc"]  * 100 if r_lo else 0)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, sd_accs, width, yerr=sd_stds, capsize=4,
                label="Subject-Dependent", color=colors[0])
    b2 = ax.bar(x + width/2, lo_accs, width, yerr=lo_stds, capsize=4,
                label="LOSO", color=colors[1])

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison on SEED Dataset", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    return fig


def _load(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["subject_dependent", "loso", "both"],
                        default="both")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["subject_dependent", "loso"] if args.split == "both" else [args.split]

    for split in splits:
        result = _load(f"{split}_attn_mlp_results.npy")
        if result is None:
            print(f"[skip] No attn_mlp result for split={split}. Run train.py first.")
            continue
        if "band_weights" in result:
            plot_band_attention(result["band_weights"], split)
        if "chan_weights" in result:
            plot_channel_attention(result["chan_weights"], split)

    plot_accuracy_comparison()

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
