"""
Subject-Dependent comparison including Cross-Session CORAL.

Reads existing result files (never overwrites them) and adds the new
MLP + Cross-Session CORAL result for comparison.

Table columns:
  Method | Accuracy | vs SVM baseline | vs MLP+CrossSession

Results written to:
  results/cross_session/summary_with_coral.csv   (new file, does not touch summary_full.csv)

Usage:
    python src/cross_session/evaluate_cross_session_coral.py
    python src/cross_session/evaluate_cross_session_coral.py --save
    python src/cross_session/evaluate_cross_session_coral.py --fold
"""

import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

RESULTS_DIR    = os.path.join(_ROOT, "results")
RESULTS_CS_DIR = os.path.join(_ROOT, "results", "cross_session")


def _load(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def _fmt(r):
    if r is None:
        return "（未运行）"
    return f"{r['mean_acc']*100:.2f}% ± {r['std_acc']*100:.2f}%"


def _delta(base, other):
    if base is None or other is None:
        return "—"
    diff = (other["mean_acc"] - base["mean_acc"]) * 100
    return f"{'+'if diff>=0 else ''}{diff:.2f}%"


def print_comparison_table():
    svm        = _load(os.path.join(RESULTS_DIR,    "subject_dependent_svm_results.npy"))
    mlp        = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    attn       = _load(os.path.join(RESULTS_DIR,    "subject_dependent_attn_mlp_results.npy"))
    svm_cs     = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_svm_cross_session_results.npy"))
    mlp_cs     = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))
    mlp_cs_da  = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_coral_results.npy"))

    # (name, result, delta_base_for_col3, delta_base_for_col4)
    # col3 = vs SVM baseline, col4 = vs MLP+CrossSession
    rows = [
        ("SVM (baseline)",                    svm,       None,   None),
        ("MLP (baseline)",                    mlp,       None,   None),
        ("MLP + Attention",                   attn,      None,   None),
        ("SVM + Cross-Session",               svm_cs,    svm,    None),
        ("MLP + Cross-Session",               mlp_cs,    mlp,    None),
        ("MLP + Cross-Session + CORAL (ours)",mlp_cs_da, mlp,    mlp_cs),
    ]

    w = [36, 26, 16, 22]
    sep    = "=" * (sum(w) + 6)
    header = (f"{'Model':<{w[0]}} {'Accuracy':>{w[1]}}"
              f" {'vs MLP base':>{w[2]}} {'vs MLP+CrossSess':>{w[3]}}")

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, base_mlp, base_cs in rows:
        d1 = _delta(base_mlp, res) if base_mlp is not None else "—"
        d2 = _delta(base_cs,  res) if base_cs  is not None else "—"
        print(f"{name:<{w[0]}} {_fmt(res):>{w[1]}} {d1:>{w[2]}} {d2:>{w[3]}}")
    print(sep)

    if mlp_cs is not None and mlp_cs_da is not None:
        diff = (mlp_cs_da["mean_acc"] - mlp_cs["mean_acc"]) * 100
        print(f"\n  CORAL alignment effect (vs MLP+CrossSession): "
              f"{'+'if diff>=0 else ''}{diff:.2f}%")

    return rows


def print_per_fold(result, label):
    if result is None:
        print(f"\n  [未找到结果: {label}]")
        return
    print(f"\nPer-fold accuracies  [{label}]")
    for i, a in enumerate(result["accs"]):
        print(f"  Fold {i+1:2d}: {a*100:.2f}%")
    print(f"  Mean : {result['mean_acc']*100:.2f}%  Std : {result['std_acc']*100:.2f}%")


def save_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Accuracy", "vs MLP baseline", "vs MLP+CrossSession"])
        for name, res, base_mlp, base_cs in rows:
            d1 = _delta(base_mlp, res) if base_mlp is not None else "—"
            d2 = _delta(base_cs,  res) if base_cs  is not None else "—"
            w.writerow([name, _fmt(res), d1, d2])
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save to results/cross_session/summary_with_coral.csv")
    parser.add_argument("--fold", action="store_true",
                        help="Print per-fold accuracies for CORAL method")
    args = parser.parse_args()

    rows = print_comparison_table()

    if args.fold:
        cs_dir = RESULTS_CS_DIR
        da_path = os.path.join(cs_dir, "subject_dependent_mlp_cross_session_coral_results.npy")
        print_per_fold(_load(da_path), "MLP + Cross-Session + CORAL")

    if args.save:
        save_csv(rows, os.path.join(RESULTS_CS_DIR, "summary_with_coral.csv"))


if __name__ == "__main__":
    main()
