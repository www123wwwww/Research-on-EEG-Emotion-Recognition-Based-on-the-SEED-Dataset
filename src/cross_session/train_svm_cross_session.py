"""
SVM + cross-session data for Subject-Dependent evaluation.

Strategy (for each fold = one (subject, session) pair):

  Training data = ALL data from the OTHER 2 sessions of the SAME subject
                  (both their train and test portions)
                + training portion of the TARGET session

  Test data     = test portion of the TARGET session (unchanged)

This is the SVM counterpart to train_cross_session.py.
SVM cannot do pre-train + fine-tune (it has no weights to initialise),
so the natural equivalent is simply to add more same-subject data to
the training set in a single fit() call.

Comparing SVM+CrossSession vs MLP+CrossSession answers:
  "Is the gain from more same-subject data (data effect)
   or from the two-phase pre-train/fine-tune paradigm (method effect)?"

Results saved to results/cross_session/  — never touches results/.

Usage:
    python src/cross_session/train_svm_cross_session.py
    python src/cross_session/train_svm_cross_session.py --quiet
"""

import sys
import os
import argparse
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from src.data_loader import (
    get_subject_dependent_splits, get_flat_features,
    _load_npz, SESSIONS,
)
from src.models.baseline import SVMModel

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "results", "cross_session")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── data helper ───────────────────────────────────────────────────────────────

def _load_other_sessions_flat(subject: int, target_session: int):
    """
    Load ALL data (train + test merged) from the two sessions of `subject`
    that are NOT `target_session`, then flatten and normalise each session
    independently (using that session's own train statistics).

    Returns:
        X : (N_other, 310)  float32
        y : (N_other,)      int64
    """
    X_parts, y_parts = [], []
    for sess in SESSIONS:
        if sess == target_session:
            continue
        X_tr, y_tr, X_te, y_te = _load_npz(subject, sess)
        X_tr_flat = get_flat_features(X_tr)
        X_te_flat = get_flat_features(X_te)
        mu    = X_tr_flat.mean(axis=0)
        sigma = X_tr_flat.std(axis=0) + 1e-8
        X_parts.append((X_tr_flat - mu) / sigma)
        X_parts.append((X_te_flat - mu) / sigma)
        y_parts.append(y_tr)
        y_parts.append(y_te)
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ── per-fold pipeline ─────────────────────────────────────────────────────────

def run_svm_cross_session_fold(split: dict) -> float:
    """
    Train a single SVM on:
        other-session data  +  target-session training data
    then evaluate on the target-session test data.
    """
    subject        = split["subject"]
    target_session = split["session"]

    # Target session (already normalised by get_subject_dependent_splits)
    X_target_train = get_flat_features(split["X_train"])
    y_target_train = split["y_train"]
    X_test         = get_flat_features(split["X_test"])
    y_test         = split["y_test"]

    # Other sessions of same subject
    X_other, y_other = _load_other_sessions_flat(subject, target_session)

    # Combine into one training set
    X_train = np.concatenate([X_other, X_target_train])
    y_train = np.concatenate([y_other, y_target_train])

    model = SVMModel(C=1.0)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# ── full experiment ───────────────────────────────────────────────────────────

def run_experiment(verbose: bool = True) -> dict:
    splits = get_subject_dependent_splits(normalize=True)
    accs   = []

    for i, split in enumerate(splits):
        label = f"sub{split['subject']}_sess{split['session']}"
        acc   = run_svm_cross_session_fold(split)
        accs.append(acc)
        if verbose:
            print(f"  [{i+1:2d}/{len(splits)}] {label:18s}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n>>> Subject-Dependent | SVM (cross-session data): "
          f"acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "model":    "svm_cross_session",
        "split":    "subject_dependent",
        "accs":     np.array(accs),
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
    }
    out = os.path.join(RESULTS_DIR, "subject_dependent_svm_cross_session_results.npy")
    np.save(out, result, allow_pickle=True)
    print(f"    Saved → {out}")
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SVM with cross-session same-subject data (Subject-Dependent)."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("Strategy : SVM trained on other-session data + target-session train data")
    print("           (single fit — no pre-train/fine-tune)\n")

    run_experiment(verbose=not args.quiet)


if __name__ == "__main__":
    main()
