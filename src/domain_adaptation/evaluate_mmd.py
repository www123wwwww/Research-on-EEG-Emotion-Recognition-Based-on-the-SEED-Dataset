"""
Comprehensive comparison table: all LOSO methods including MMD and CORAL.

Reads from:
  results/                   — SVM, MLP, MLP+Attention baselines
  results/domain_adaptation/ — CORAL and MMD results

Usage:
    python src/domain_adaptation/evaluate_mmd.py
    python src/domain_adaptation/evaluate_mmd.py --save    # writes summary_all_da.csv
    python src/domain_adaptation/evaluate_mmd.py --fold    # per-fold breakdown
"""

import sys
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

RESULTS_DIR    = os.path.join(_ROOT, "results")
RESULTS_DA_DIR = os.path.join(_ROOT, "results", "domain_adaptation")


def _load(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def _fmt(result):
    if result is None:
        return "（未运行）"
    return f"{result['mean_acc']*100:.2f}% ± {result['std_acc']*100:.2f}%"


def _delta(baseline, other):
    """Show improvement of `other` over `baseline`."""
    if baseline is None or other is None:
        return "—"
    diff = (other["mean_acc"] - baseline["mean_acc"]) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f}%"


def print_comparison_table():
    svm      = _load(os.path.join(RESULTS_DIR,    "loso_svm_results.npy"))
    mlp      = _load(os.path.join(RESULTS_DIR,    "loso_mlp_results.npy"))
    attn     = _load(os.path.join(RESULTS_DIR,    "loso_attn_mlp_results.npy"))
    coral    = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_coral_results.npy"))
    mmd      = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_mmd_results.npy"))

    rows = [
        ("SVM (baseline)",      svm,   None),
        ("MLP (baseline)",      mlp,   None),
        ("MLP + Attention",     attn,  None),
        ("MLP + CORAL",         coral, mlp),
        ("MLP + MMD  (ours)",   mmd,   mlp),
    ]

    w = [22, 28, 14]
    sep    = "=" * (sum(w) + 4)
    header = f"{'Model':<{w[0]}} {'LOSO Accuracy':>{w[1]}} {'vs MLP':>{w[2]}}"

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, base in rows:
        delta = _delta(base, res) if base is not None else "—"
        print(f"{name:<{w[0]}} {_fmt(res):>{w[1]}} {delta:>{w[2]}}")
    print(sep)

    if coral is not None:
        print(f"  CORAL λ = {coral.get('lambda_coral', '?')}")
    if mmd is not None:
        print(f"  MMD   λ = {mmd.get('lambda_mmd', '?')}")

    return rows


def print_per_fold(result, label):
    if result is None:
        print(f"\n  [未找到结果: {label}，请先运行对应训练脚本]")
        return
    print(f"\nPer-fold accuracies  [{label}]")
    for i, a in enumerate(result["accs"]):
        print(f"  Fold {i+1:2d}: {a*100:.2f}%")
    print(f"  Mean : {result['mean_acc']*100:.2f}%  "
          f"Std : {result['std_acc']*100:.2f}%")


def save_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "LOSO Acc", "Delta vs MLP"])
        for name, res, base in rows:
            w.writerow([
                name,
                _fmt(res),
                _delta(base, res) if base is not None else "—",
            ])
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save full comparison table to CSV")
    parser.add_argument("--fold", action="store_true",
                        help="Print per-fold breakdown for MLP, CORAL, MMD")
    args = parser.parse_args()

    rows = print_comparison_table()

    if args.fold:
        mlp   = _load(os.path.join(RESULTS_DIR,    "loso_mlp_results.npy"))
        coral = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_coral_results.npy"))
        mmd   = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_mmd_results.npy"))
        print_per_fold(mlp,   "MLP baseline")
        print_per_fold(coral, "MLP + CORAL")
        print_per_fold(mmd,   "MLP + MMD")

    if args.save:
        csv_path = os.path.join(RESULTS_DA_DIR, "summary_all_da.csv")
        save_csv(rows, csv_path)


if __name__ == "__main__":
    main()
