"""
Full LOSO comparison: baselines + CORAL + MMD + DANN.

Reads from:
  results/                   — SVM, MLP, MLP+Attention baselines
  results/domain_adaptation/ — CORAL, MMD, DANN results

Usage:
    python src/domain_adaptation/evaluate_dann.py
    python src/domain_adaptation/evaluate_dann.py --save
    python src/domain_adaptation/evaluate_dann.py --fold
"""

import os
import sys
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
    svm   = _load(os.path.join(RESULTS_DIR,    "loso_svm_results.npy"))
    mlp   = _load(os.path.join(RESULTS_DIR,    "loso_mlp_results.npy"))
    attn  = _load(os.path.join(RESULTS_DIR,    "loso_attn_mlp_results.npy"))
    coral = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_coral_results.npy"))
    mmd   = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_mmd_results.npy"))
    dann  = _load(os.path.join(RESULTS_DA_DIR, "loso_mlp_dann_results.npy"))

    rows = [
        ("SVM (baseline)",      svm,   None),
        ("MLP (baseline)",      mlp,   None),
        ("MLP + Attention",     attn,  None),
        ("── Domain Adaptation ──────────────────────────", None, None),
        ("MLP + CORAL",         coral, mlp),
        ("MLP + MMD",           mmd,   mlp),
        ("MLP + DANN  (ours)",  dann,  mlp),
    ]

    w = [28, 26, 12]
    sep    = "=" * (sum(w) + 4)
    header = f"{'Model':<{w[0]}} {'LOSO Accuracy':>{w[1]}} {'vs MLP':>{w[2]}}"

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, res, base in rows:
        if res is None and base is None and name.startswith("──"):
            print(f"\n  {name}")
            continue
        delta = _delta(base, res) if base is not None else "—"
        print(f"{name:<{w[0]}} {_fmt(res):>{w[1]}} {delta:>{w[2]}}")
    print(sep)

    for key, r in [("CORAL", coral), ("MMD", mmd), ("DANN", dann)]:
        if r is not None:
            lk = {"CORAL": "lambda_coral", "MMD": "lambda_mmd",
                  "DANN": "lambda_dann"}.get(key, "lambda")
            print(f"  {key} λ = {r.get(lk, '?')}")

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
        w.writerow(["Model", "LOSO Acc", "vs MLP"])
        for name, res, base in rows:
            if res is None and base is None:
                continue
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
        for fname, label in [
            ("loso_mlp_results.npy",       "MLP baseline"),
            ("loso_mlp_coral_results.npy",  "MLP + CORAL"),
            ("loso_mlp_mmd_results.npy",    "MLP + MMD"),
            ("loso_mlp_dann_results.npy",   "MLP + DANN"),
        ]:
            folder = RESULTS_DIR if "coral" not in fname and "mmd" not in fname \
                     and "dann" not in fname else RESULTS_DA_DIR
            if "mlp_results" in fname:
                folder = RESULTS_DIR
            print_per_fold(_load(os.path.join(folder, fname)), label)

    if args.save:
        save_csv(rows, os.path.join(RESULTS_DA_DIR, "summary_all_loso.csv"))


if __name__ == "__main__":
    main()
