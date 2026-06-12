"""
Baseline models: SVM and MLP.

Both accept flat features (N, 310).
SVMModel wraps sklearn; MLPModel is a PyTorch nn.Module.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC


# ──────────────────────────────────────────────
# SVM (sklearn)
# ──────────────────────────────────────────────

class SVMModel:
    """RBF-kernel SVM with optional internal standard scaling."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str = "scale"):
        self.clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.clf.score(X, y)


# ──────────────────────────────────────────────
# MLP (PyTorch)
# ──────────────────────────────────────────────

class MLPModel(nn.Module):
    """
    3-layer MLP: 310 → 256 → 128 → 3

    Dropout + BatchNorm for regularisation.
    """

    def __init__(self, input_dim: int = 310, hidden1: int = 256,
                 hidden2: int = 128, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
