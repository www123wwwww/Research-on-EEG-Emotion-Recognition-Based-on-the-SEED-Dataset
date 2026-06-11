"""
Group A: Subject-dependent experiment with unified hyperparameters.
Uses random stratified validation (same as v1 C/D/F groups).
Saves to v1 results directory.
"""
import sys, os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.data_loader import get_subject_dependent_splits, get_flat_features

RESULTS_DIR = os.path.join(_HERE, "results", "loso_domain_adaptation")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_EPOCHS = 20
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stratified_split(X, y, val_ratio=0.2):
    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Group A: Subject-Dependent (unified hyperparams, stratified val)")
    print(f"Hyperparams: epochs={MAX_EPOCHS}, lr={LR}, bs={BATCH_SIZE}, wd={WEIGHT_DECAY}, seed={SEED}")

    splits = get_subject_dependent_splits(normalize=True)
    splits_data = []

    for sp in splits:
        X_tr = get_flat_features(sp["X_train"])
        X_te = get_flat_features(sp["X_test"])
        y_tr, y_te = sp["y_train"], sp["y_test"]
        X_train, y_train, X_val, y_val = _stratified_split(X_tr, y_tr)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)),
                                batch_size=256, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long)),
                                 batch_size=256, shuffle=False)
        splits_data.append({
            "subject": sp["subject"], "session": sp["session"],
            "train_loader": train_loader, "val_loader": val_loader,
            "test_loader": test_loader, "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_te),
        })

    from src.models.baseline import MLPModel

    accs = []
    per_session = []
    for i, sd in enumerate(splits_data):
        model = MLPModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_val, best_state = -1.0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for X_b, y_b in sd["train_loader"]:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(X_b), y_b).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for X_v, y_v in sd["val_loader"]:
                    X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                    c += (model(X_v).argmax(1) == y_v).sum().item()
                    t += len(y_v)
            if c / t > best_val:
                best_val = c / t
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for X_t, y_t in sd["test_loader"]:
                X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                c += (model(X_t).argmax(1) == y_t).sum().item()
                t += len(y_t)
        acc = c / t
        accs.append(acc)
        per_session.append({
            "subject": int(sd["subject"]), "session": int(sd["session"]),
            "acc": float(acc), "n_train": sd["n_train"], "n_val": sd["n_val"], "n_test": sd["n_test"],
        })
        print(f"  [{i+1}/{len(splits_data)}] sub{sd['subject']}_s{sd['session']}  "
              f"train={sd['n_train']} val={sd['n_val']} test={sd['n_test']}  acc={acc*100:.2f}%")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\n>>> Group A: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    result = {
        "group": "A", "model": "MLP", "split": "subject_dependent", "norm": "per_subject",
        "val_method": "stratified",
        "mean_acc": float(mean_acc), "std_acc": float(std_acc),
        "accs": [float(a) for a in accs],
        "per_session": per_session,
        "hyperparams": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                         "weight_decay": WEIGHT_DECAY, "seed": SEED, "val_ratio": 0.2},
    }
    path = os.path.join(RESULTS_DIR, "group_A_subject_dependent_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")

    # Also compute per-subject averages
    subjs = sorted(set(s["subject"] for s in per_session))
    print("\nPer-subject averages:")
    for subj in subjs:
        sa = [s["acc"] for s in per_session if s["subject"] == subj]
        print(f"  Subject {subj}: {np.mean(sa)*100:.2f}% ± {np.std(sa)*100:.2f}% (n={len(sa)})")


if __name__ == "__main__":
    main()