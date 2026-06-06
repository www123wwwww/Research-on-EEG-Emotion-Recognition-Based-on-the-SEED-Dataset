"""
Evaluation script: read all experiment results and produce summary tables.

Reads from: /hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult/
"""

import json
import os
import glob

RESULTS_DIR = "/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult"

SUBJECTS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]


def load_result(method, norm, lambda_da=1.0, weighted_multisource=False, temperature=None):
    fname = f"{method}_{norm}_lambda{lambda_da}_results.json"
    if weighted_multisource and temperature is not None:
        fname = f"{method}_{norm}_weighted_t{temperature}_lambda{lambda_da}_results.json"
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(r):
    if r is None:
        return "--"
    return f"{r['mean_acc']*100:.2f}% +/- {r['std_acc']*100:.2f}%"


def delta(base, other):
    if base is None or other is None:
        return "--"
    d = (other["mean_acc"] - base["mean_acc"]) * 100
    return f"{'+'if d>=0 else ''}{d:.2f}%"


def print_main_table():
    print("\n" + "=" * 100)
    print("Table 1: LOSO Accuracy Comparison (epoch=20, lr=5e-4, no early stopping)")
    print("=" * 100)

    baseline_mix = load_result("baseline", "mixed")
    baseline_ps = load_result("baseline", "per_subject")
    coral_mix = load_result("coral", "mixed")
    mmd_mix = load_result("mmd", "mixed")
    dann_mix = load_result("dann", "mixed")
    coral_ps = load_result("coral", "per_subject")
    mmd_ps = load_result("mmd", "per_subject")
    dann_ps = load_result("dann", "per_subject")

    rows = [
        ("MLP baseline", "mixed", baseline_mix, None),
        ("MLP + CORAL", "mixed", coral_mix, baseline_mix),
        ("MLP + MMD", "mixed", mmd_mix, baseline_mix),
        ("MLP + DANN", "mixed", dann_mix, baseline_mix),
        ("---", "", None, None),
        ("MLP baseline", "per-subject", baseline_ps, None),
        ("MLP + CORAL", "per-subject", coral_ps, baseline_ps),
        ("MLP + MMD", "per-subject", mmd_ps, baseline_ps),
        ("MLP + DANN", "per-subject", dann_ps, baseline_ps),
    ]

    print(f"{'Method':<35} {'Norm':<14} {'Accuracy':>20} {'vs Baseline':>14}")
    print("-" * 85)
    for name, norm, r, base in rows:
        if name == "---":
            print()
            continue
        print(f"{name:<35} {norm:<14} {fmt(r):>20} {delta(base, r):>14}")

    print("\n--- Cross-comparison: per-subject norm vs mixed norm ---")
    if baseline_mix and baseline_ps:
        print(f"Normalization gain (baseline):     {delta(baseline_mix, baseline_ps)}")
    if mmd_mix and mmd_ps:
        print(f"Normalization + DA gain (MMD):   {delta(mmd_mix, mmd_ps)}")
    if coral_mix and coral_ps:
        print(f"Normalization + DA gain (CORAL):  {delta(coral_mix, coral_ps)}")
    if dann_mix and dann_ps:
        print(f"Normalization + DA gain (DANN):   {delta(dann_mix, dann_ps)}")


def print_normalization_ablation():
    print("\n" + "=" * 80)
    print("Table 2: Normalization Ablation (isolating normalization vs DA effects)")
    print("=" * 80)

    baseline_mix = load_result("baseline", "mixed")
    baseline_ps = load_result("baseline", "per_subject")
    mmd_mix = load_result("mmd", "mixed")
    mmd_ps = load_result("mmd", "per_subject")
    coral_mix = load_result("coral", "mixed")
    coral_ps = load_result("coral", "per_subject")
    dann_mix = load_result("dann", "mixed")
    dann_ps = load_result("dann", "per_subject")

    comparisons = [
        ("Normalization alone", baseline_mix, baseline_ps),
        ("DA alone (CORAL)", baseline_mix, coral_mix),
        ("DA alone (MMD)", baseline_mix, mmd_mix),
        ("DA alone (DANN)", baseline_mix, dann_mix),
        ("Both (CORAL+norm)", baseline_mix, coral_ps),
        ("Both (MMD+norm)", baseline_mix, mmd_ps),
        ("Both (DANN+norm)", baseline_mix, dann_ps),
    ]

    print(f"{'Effect':<30} {'Mixed Acc':>12} {'PerSubj Acc':>12} {'Gain':>10}")
    print("-" * 66)
    base_ref = baseline_mix
    for label, r_mix, r_ps in comparisons:
        if r_mix is not None:
            print(f"{label:<30} {r_mix['mean_acc']*100:.2f}%      ", end="")
        else:
            print(f"{label:<30} {'--':>12} ", end="")
        if r_ps is not None:
            print(f"{r_ps['mean_acc']*100:.2f}%      ", end="")
        else:
            print(f"{'--':>12} ", end="")
        if base_ref is not None and r_ps is not None:
            d = (r_ps["mean_acc"] - base_ref["mean_acc"]) * 100
            print(f"{'+'if d>=0 else ''}{d:.2f}%")
        else:
            print("--")


def print_lambda_sensitivity():
    print("\n" + "=" * 80)
    print("Table 3: Lambda Sensitivity (per-subject norm)")
    print("=" * 80)

    lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    methods = [("CORAL", "coral"), ("MMD", "mmd"), ("DANN", "dann")]

    header = f"{'Method':<10}" + "".join(f"{'λ='+str(l):<12}" for l in lambdas)
    print(header)
    print("-" * len(header))
    for name, key in methods:
        row = f"{name:<10}"
        for lam in lambdas:
            r = load_result(key, "per_subject", lambda_da=lam)
            if r:
                row += f"{r['mean_acc']*100:.2f}%      "
            else:
                row += f"{'--':<12}"
        print(row)


def print_per_fold_table():
    print("\n" + "=" * 120)
    print("Table 4: Per-fold (Per-subject) Accuracy")
    print("=" * 120)

    key_results = [
        ("MLP (mixed)", load_result("baseline", "mixed")),
        ("MLP+MMD (mixed)", load_result("mmd", "mixed")),
        ("MLP+DANN (mixed)", load_result("dann", "mixed")),
        ("MLP (per-subj)", load_result("baseline", "per_subject")),
        ("MLP+MMD (per-subj)", load_result("mmd", "per_subject")),
        ("MLP+DANN (per-subj)", load_result("dann", "per_subject")),
    ]

    header = f"{'Method':<20}" + "".join(f"{'sub'+str(s):>7}" for s in SUBJECTS) + f"{'Mean':>8}{'Std':>8}"
    print(header)
    print("-" * len(header))
    for name, r in key_results:
        if r is None:
            continue
        row = f"{name:<20}"
        for s in SUBJECTS:
            accs_fold = r.get("accs", [])
            idx = SUBJECTS.index(s) if s in SUBJECTS else -1
            if idx < len(accs_fold):
                row += f"{accs_fold[idx]*100:7.2f}"
            else:
                row += f"{'--':>7}"
        row += f"{r['mean_acc']*100:8.2f}{r['std_acc']*100:8.2f}"
        print(row)


def save_csv():
    import csv
    baseline_mix = load_result("baseline", "mixed")
    baseline_ps = load_result("baseline", "per_subject")

    rows = []
    for method in ["baseline", "coral", "mmd", "dann"]:
        for norm in ["mixed", "per_subject"]:
            for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
                r = load_result(method, norm, lambda_da=lam)
                if r is None:
                    continue
                base = baseline_mix if norm == "mixed" else baseline_ps
                rows.append({
                    "method": method,
                    "norm": norm,
                    "lambda": lam,
                    "mean_acc": f"{r['mean_acc']*100:.2f}",
                    "std_acc": f"{r['std_acc']*100:.2f}",
                    "vs_baseline": delta(base, r),
                    "per_fold_accs": ",".join(f"{a*100:.2f}" for a in r.get("accs", [])),
                })

    path = os.path.join(RESULTS_DIR, "summary_all.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "norm", "lambda", "mean_acc", "std_acc", "vs_baseline", "per_fold_accs"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved -> {path}")


if __name__ == "__main__":
    print_main_table()
    print_normalization_ablation()
    print_lambda_sensitivity()
    print_per_fold_table()
    save_csv()