"""
Comparison table: baseline models vs MLP + CORAL (LOSO only).

Reads from:
  results/                      — original baseline results
  results/domain_adaptation/    — CORAL results (run train_coral.py first)

Usage:
    python src/domain_adaptation/evaluate_coral.py
    python src/domain_adaptation/evaluate_coral.py --save   # also writes summary_coral.csv
    python src/domain_adaptation/evaluate_coral.py --fold   # per-fold breakdown
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
        return "—"
    return f"{result['mean_acc']*100:.2f}% ± {result['std_acc']*100:.2f}%"


def _delta(baseline, coral):
    if baseline is None or coral is None:
        return "—"
    diff = (coral["mean_acc"] - baseline["mean_acc"]) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f}%"


def print_comparison_table():
    svm      = _load(os.path.join(RESULTS_DIR, "loso_svm_results.npy"))
    mlp      = _load(os.path.join(RESULTS_DIR, "loso_mlp_results.npy"))
    attn_mlp = _load(os.path.join(RESULTS_DIR, "loso_attn_mlp_results.npy"))
    coral    = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_coral_results.npy"))

    rows = [
        ("SVM (baseline)",        svm,      None),
        ("MLP (baseline)",        mlp,      None),
        ("MLP + Attention",       attn_mlp, None),
        ("MLP + CORAL (ours)",    coral,    mlp),
    ]

    col_w = [22, 26, 12]
    header = (f"{'Model':<{col_w[0]}} {'LOSO Accuracy':>{col_w[1]}} "
              f"{'vs MLP':>{col_w[2]}}")
    sep = "=" * (sum(col_w) + 2)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, baseline in rows:
        delta = _delta(baseline, res) if baseline is not None else "—"
        print(f"{name:<{col_w[0]}} {_fmt(res):>{col_w[1]}} {delta:>{col_w[2]}}")
    print(sep)

    if coral is not None:
        lam = coral.get("lambda_coral", "?")
        print(f"  CORAL λ = {lam}")

    return rows


def print_per_fold(result, label):
    if result is None:
        print(f"  [no result for {label}]")
        return
    print(f"\nPer-fold accuracies  [{label}]")
    for i, a in enumerate(result["accs"]):
        print(f"  Fold {i+1:2d}: {a*100:.2f}%")
    print(f"  Mean: {result['mean_acc']*100:.2f}%  "
          f"Std: {result['std_acc']*100:.2f}%")


def save_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "LOSO Acc", "Delta vs MLP"])
        for name, res, baseline in rows:
            w.writerow([name, _fmt(res), _delta(baseline, res) if baseline else "—"])
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fold", action="store_true")
    args = parser.parse_args()

    rows = print_comparison_table()

    if args.fold:
        coral = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_coral_results.npy"))
        mlp   = _load(os.path.join(RESULTS_DIR, "loso_mlp_results.npy"))
        print_per_fold(mlp,   "MLP baseline")
        print_per_fold(coral, "MLP + CORAL")

    if args.save:
        csv_path = os.path.join(RESULTS_DA_DIR, "summary_coral.csv")
        save_csv(rows, csv_path)


if __name__ == "__main__":
    main()
