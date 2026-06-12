"""
CoralMLP: MLP split into feature_extractor + classifier.

Architecture is identical to the baseline MLPModel (310→256→128→3),
but the forward pass exposes the 128-dim intermediate representation so
the CORAL loss can be applied on those features.

Splitting at the penultimate layer (128-d) rather than the raw input
lets CORAL act on learned representations rather than raw DE features,
which typically yields better alignment.
"""

import torch
import torch.nn as nn


class CoralMLP(nn.Module):
    """
    MLP with a two-stage forward pass for domain adaptation.

    Layers:
        feature_extractor : 310 → 256 → 128   (same as baseline MLP hidden layers)
        classifier        : 128 → 3

    forward(x) returns (logits, features):
        logits   — (N, 3)   used for classification loss
        features — (N, 128) used for CORAL loss
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
