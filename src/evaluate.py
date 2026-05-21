"""
Evaluation utilities and result summary table.

Usage:
    python src/evaluate.py           # print comparison table for all saved results
    python src/evaluate.py --save    # also save table to results/summary.csv
"""

import os
import sys
import argparse
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

MODEL_ORDER = ["svm", "mlp", "attn_mlp"]
SPLIT_ORDER = ["subject_dependent", "loso"]
MODEL_NAMES = {"svm": "SVM (baseline)", "mlp": "MLP (baseline)", "attn_mlp": "MLP+Attention"}
SPLIT_NAMES = {"subject_dependent": "Subject-Dependent", "loso": "LOSO (cross-subject)"}


def load_result(split: str, model: str):
    path = os.path.join(RESULTS_DIR, f"{split}_{model}_results.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def print_summary_table():
    rows = []
    header = f"{'Model':<22} {'Subject-Dep. Acc':>18} {'LOSO Acc':>18}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for model in MODEL_ORDER:
        sd = load_result("subject_dependent", model)
        lo = load_result("loso", model)
        sd_str = f"{sd['mean_acc']*100:.2f}% ± {sd['std_acc']*100:.2f}%" if sd else "—"
        lo_str = f"{lo['mean_acc']*100:.2f}% ± {lo['std_acc']*100:.2f}%" if lo else "—"
        row = f"{MODEL_NAMES[model]:<22} {sd_str:>18} {lo_str:>18}"
        print(row)
        rows.append((MODEL_NAMES[model], sd_str, lo_str))

    print("=" * len(header))
    return rows


def print_per_fold(split: str, model: str):
    result = load_result(split, model)
    if result is None:
        print(f"No result found: {split}/{model}")
        return
    accs = result["accs"]
    print(f"\nPer-fold accuracies  [{SPLIT_NAMES[split]}  |  {MODEL_NAMES[model]}]")
    for i, a in enumerate(accs):
        print(f"  Fold {i+1:2d}: {a*100:.2f}%")
    print(f"  Mean: {accs.mean()*100:.2f}%  Std: {accs.std()*100:.2f}%")


def save_csv(rows, path: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Subject-Dependent Acc", "LOSO Acc"])
        w.writerows(rows)
    print(f"\nTable saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save summary table to CSV")
    parser.add_argument("--fold", action="store_true", help="Print per-fold breakdown")
    args = parser.parse_args()

    rows = print_summary_table()

    if args.fold:
        for split in SPLIT_ORDER:
            for model in MODEL_ORDER:
                print_per_fold(split, model)

    if args.save:
        csv_path = os.path.join(RESULTS_DIR, "summary.csv")
        save_csv(rows, csv_path)


if __name__ == "__main__":
    main()
