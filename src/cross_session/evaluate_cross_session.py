"""
Comparison table: Subject-Dependent baselines vs cross-session pretrained MLP.

Reads from:
  results/                 — SVM, MLP, MLP+Attention (existing, untouched)
  results/cross_session/   — MLP cross-session pretrain result

Usage:
    python src/cross_session/evaluate_cross_session.py
    python src/cross_session/evaluate_cross_session.py --save
    python src/cross_session/evaluate_cross_session.py --fold
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


def _delta(baseline, other):
    if baseline is None or other is None:
        return "—"
    diff = (other["mean_acc"] - baseline["mean_acc"]) * 100
    return f"{'+'if diff>=0 else ''}{diff:.2f}%"


def print_comparison_table():
    svm  = _load(os.path.join(RESULTS_DIR,    "subject_dependent_svm_results.npy"))
    mlp  = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    attn = _load(os.path.join(RESULTS_DIR,    "subject_dependent_attn_mlp_results.npy"))
    cs   = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))

    rows = [
        ("SVM (baseline)",             svm,  None),
        ("MLP (baseline)",             mlp,  None),
        ("MLP + Attention",            attn, None),
        ("MLP + Cross-Session (ours)", cs,   svm),   # compare against SVM (the winner)
    ]

    w = [28, 26, 14]
    sep    = "=" * (sum(w) + 4)
    header = f"{'Model':<{w[0]}} {'Subject-Dep. Accuracy':>{w[1]}} {'vs SVM':>{w[2]}}"

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, base in rows:
        delta = _delta(base, res) if base is not None else "—"
        print(f"{name:<{w[0]}} {_fmt(res):>{w[1]}} {delta:>{w[2]}}")
    print(sep)
    return rows


def print_per_fold(result, label):
    if result is None:
        print(f"\n  [未找到结果: {label}]")
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
        w.writerow(["Model", "Subject-Dep. Acc", "vs SVM"])
        for name, res, base in rows:
            w.writerow([name, _fmt(res),
                        _delta(base, res) if base is not None else "—"])
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fold", action="store_true")
    args = parser.parse_args()

    rows = print_comparison_table()

    if args.fold:
        svm = _load(os.path.join(RESULTS_DIR,    "subject_dependent_svm_results.npy"))
        mlp = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
        cs  = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))
        print_per_fold(svm, "SVM baseline")
        print_per_fold(mlp, "MLP baseline")
        print_per_fold(cs,  "MLP + Cross-Session")

    if args.save:
        save_csv(rows, os.path.join(RESULTS_CS_DIR, "summary_cross_session.csv"))


if __name__ == "__main__":
    main()
