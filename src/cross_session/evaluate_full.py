"""
Full Subject-Dependent comparison: all 5 methods.

Reads from:
  results/               — SVM, MLP, MLP+Attention baselines (untouched)
  results/cross_session/ — SVM+CrossSession, MLP+CrossSession results

The key comparison this table enables:
  SVM baseline  vs  SVM+CrossSession   → pure data-addition effect (no method change)
  MLP baseline  vs  MLP+CrossSession   → pre-train/fine-tune effect on MLP
  SVM+CrossSession vs MLP+CrossSession → does two-phase training add value beyond more data?

Usage:
    python src/cross_session/evaluate_full.py
    python src/cross_session/evaluate_full.py --save
    python src/cross_session/evaluate_full.py --fold
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
    svm      = _load(os.path.join(RESULTS_DIR,    "subject_dependent_svm_results.npy"))
    mlp      = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    attn     = _load(os.path.join(RESULTS_DIR,    "subject_dependent_attn_mlp_results.npy"))
    svm_cs   = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_svm_cross_session_results.npy"))
    mlp_cs   = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))

    # baseline column shows delta vs same model's baseline
    rows = [
        ("SVM (baseline)",              svm,    None),
        ("MLP (baseline)",              mlp,    None),
        ("MLP + Attention",             attn,   None),
        ("SVM + Cross-Session",         svm_cs, svm),
        ("MLP + Cross-Session (ours)",  mlp_cs, mlp),
    ]

    w = [28, 26, 16]
    sep    = "=" * (sum(w) + 4)
    header = (f"{'Model':<{w[0]}} {'Subject-Dep. Accuracy':>{w[1]}}"
              f" {'vs own baseline':>{w[2]}}")

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, base in rows:
        delta = _delta(base, res) if base is not None else "—"
        print(f"{name:<{w[0]}} {_fmt(res):>{w[1]}} {delta:>{w[2]}}")
    print(sep)

    # Extra comparison: does MLP+CS beat SVM+CS?
    if svm_cs is not None and mlp_cs is not None:
        diff = (mlp_cs["mean_acc"] - svm_cs["mean_acc"]) * 100
        sign = "+" if diff >= 0 else ""
        print(f"\n  MLP+CrossSession vs SVM+CrossSession : "
              f"{sign}{diff:.2f}%  "
              f"({'pre-train/fine-tune adds value' if diff > 0 else 'data effect dominates'})")

    return rows


def print_per_fold(result, label):
    if result is None:
        print(f"\n  [未找到结果: {label}]")
        return
    print(f"\nPer-fold accuracies  [{label}]")
    for i, a in enumerate(result["accs"]):
        print(f"  Fold {i+1:2d}: {a*100:.2f}%")
    print(f"  Mean : {result['mean_acc']*100:.2f}%  "
          f"Std  : {result['std_acc']*100:.2f}%")


def save_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Subject-Dep. Acc", "vs own baseline"])
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
        cs_dir = RESULTS_CS_DIR
        bl_dir = RESULTS_DIR
        for fname, folder, label in [
            ("subject_dependent_svm_results.npy",                bl_dir, "SVM baseline"),
            ("subject_dependent_mlp_results.npy",                bl_dir, "MLP baseline"),
            ("subject_dependent_svm_cross_session_results.npy",  cs_dir, "SVM + Cross-Session"),
            ("subject_dependent_mlp_cross_session_results.npy",  cs_dir, "MLP + Cross-Session"),
        ]:
            print_per_fold(_load(os.path.join(folder, fname)), label)

    if args.save:
        save_csv(rows, os.path.join(RESULTS_CS_DIR, "summary_full.csv"))


if __name__ == "__main__":
    main()
