"""
MmdMLP: MLP split into feature_extractor + classifier for MMD training.

Architecture identical to CoralMLP (and baseline MLPModel):
    310 → 256 → 128  (feature_extractor)
    128 → 3          (classifier)

Kept as a separate file so the MMD experiment is fully self-contained
and does not depend on the CORAL experiment files.
"""

import torch
import torch.nn as nn


class MmdMLP(nn.Module):
    """
    MLP with exposed penultimate features for MMD domain adaptation.

    forward(x) → (logits, features)
        logits   : (N, 3)   for CrossEntropy loss
        features : (N, 128) for MMD loss
    """

    def __init__(self, input_dim: int = 310, hidden1: int = 256,
                 hidden2: int = 128, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden2, n_classes)

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)   # (N, 128)
        logits   = self.classifier(features)   # (N, 3)
        return logits, features
