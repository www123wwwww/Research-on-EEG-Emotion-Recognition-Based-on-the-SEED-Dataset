"""
Subject-Dependent comparison including AdaBN variants.

Reads existing result files (never overwrites them).

Prints a table covering:
  - MLP baseline
  - MLP + Cross-Session (pretrain + finetune, no adaptation)
  - MLP + Cross-Session + AdaBN only  (pretrain → AdaBN → direct eval)
  - MLP + Cross-Session + AdaBN + FT  (pretrain → AdaBN → finetune)
  - MLP + Cross-Session + CORAL       (if available)

Saved to:
  results/cross_session/summary_with_adabn.csv  (new file)

Usage:
    python src/cross_session/evaluate_cross_session_adabn.py
    python src/cross_session/evaluate_cross_session_adabn.py --save
    python src/cross_session/evaluate_cross_session_adabn.py --fold
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
    mlp            = _load(os.path.join(RESULTS_DIR,    "subject_dependent_mlp_results.npy"))
    mlp_cs         = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_results.npy"))
    adabn_only     = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_adabn_only_results.npy"))
    adabn_ft       = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_adabn_results.npy"))
    coral          = _load(os.path.join(RESULTS_CS_DIR, "subject_dependent_mlp_cross_session_coral_results.npy"))

    # (name, result, baseline_for_delta1, baseline_for_delta2)
    # delta1 = vs MLP baseline, delta2 = vs MLP+CrossSession
    rows = [
        ("MLP (baseline)",                mlp,        None,   None),
        ("MLP + Cross-Session",           mlp_cs,     mlp,    None),
        ("MLP + CS + AdaBN (no FT)",      adabn_only, mlp,    mlp_cs),
        ("MLP + CS + AdaBN + Finetune",   adabn_ft,   mlp,    mlp_cs),
        ("MLP + CS + CORAL",              coral,      mlp,    mlp_cs),
    ]

    w = [34, 26, 16, 22]
    sep    = "=" * (sum(w) + 6)
    header = (f"{'Model':<{w[0]}} {'Accuracy':>{w[1]}}"
              f" {'vs MLP base':>{w[2]}} {'vs MLP+CrossSess':>{w[3]}}")

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, base1, base2 in rows:
        d1 = _delta(base1, res) if base1 is not None else "—"
        d2 = _delta(base2, res) if base2 is not None else "—"
        print(f"{name:<{w[0]}} {_fmt(res):>{w[1]}} {d1:>{w[2]}} {d2:>{w[3]}}")
    print(sep)

    # Insight: AdaBN-only shows how much free (label-free) gain is available
    if mlp_cs is not None and adabn_only is not None and adabn_ft is not None:
        free_gain = (adabn_only["mean_acc"] - mlp_cs["mean_acc"]) * 100
        ft_gain   = (adabn_ft ["mean_acc"] - adabn_only["mean_acc"]) * 100
        print(f"\n  AdaBN label-free gain (no FT)         : "
              f"{'+'if free_gain>=0 else ''}{free_gain:.2f}%")
        print(f"  Additional finetune gain (after AdaBN): "
              f"{'+'if ft_gain>=0 else ''}{ft_gain:.2f}%")

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
        for name, res, base1, base2 in rows:
            d1 = _delta(base1, res) if base1 is not None else "—"
            d2 = _delta(base2, res) if base2 is not None else "—"
            w.writerow([name, _fmt(res), d1, d2])
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save to results/cross_session/summary_with_adabn.csv")
    parser.add_argument("--fold", action="store_true",
                        help="Print per-fold accuracies for AdaBN variants")
    args = parser.parse_args()

    rows = print_comparison_table()

    if args.fold:
        cs_dir = RESULTS_CS_DIR
        for fname, label in [
            ("subject_dependent_mlp_cross_session_adabn_only_results.npy",
             "MLP + CS + AdaBN (no FT)"),
            ("subject_dependent_mlp_cross_session_adabn_results.npy",
             "MLP + CS + AdaBN + Finetune"),
        ]:
            print_per_fold(_load(os.path.join(cs_dir, fname)), label)

    if args.save:
        save_csv(rows, os.path.join(RESULTS_CS_DIR, "summary_with_adabn.csv"))


if __name__ == "__main__":
    main()
