"""
DANN model: Feature Extractor + Label Classifier + Domain Classifier.

Core idea
─────────
A Gradient Reversal Layer (GRL) sits between the feature extractor and the
domain classifier.  During the forward pass it is an identity function.
During backprop it multiplies the incoming gradient by −α, which causes
the feature extractor to be updated in the direction that MAXIMISES domain
classification error — i.e. it is forced to produce features that the domain
classifier cannot distinguish as "source" or "target".

The label classifier sees the same features and is trained normally with
cross-entropy.  The two objectives compete:

  feature extractor ←→ label classifier    (cooperate: minimise label loss)
  feature extractor ←→ domain classifier   (adversarial: feature extractor
                                             tries to fool the domain clf)

α is scheduled from 0 → 1 during training so domain adaptation pressure
builds up gradually (prevents destabilising the label classifier early on).

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks",
    JMLR 2016.

Architecture
────────────
  feature_extractor  : 310 → 256 → 128   (shared trunk)
  label_classifier   : 128 → 3            (emotion: negative/neutral/positive)
  domain_classifier  : 128 → GRL(α) → 64 → 2   (source=0 / target=1)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ── Gradient Reversal Layer ───────────────────────────────────────────────────

class _GRLFunction(Function):
    """Custom autograd function: identity forward, reversed+scaled backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha          # save scalar (not a tensor, so don't use save_for_backward)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None   # None for alpha's grad (not a tensor input)


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with strength α."""
    return _GRLFunction.apply(x, alpha)


# ── Full DANN model ───────────────────────────────────────────────────────────

class DannMLP(nn.Module):
    """
    MLP with adversarial domain adaptation via GRL.

    forward(x, alpha) returns (label_logits, domain_logits, features):
        label_logits  : (N, 3)   used for emotion classification loss
        domain_logits : (N, 2)   used for domain adversarial loss (via GRL)
        features      : (N, 128) penultimate representations (for analysis)
    """

    def __init__(self, input_dim: int = 310, hidden1: int = 256,
                 hidden2: int = 128, n_classes: int = 3,
                 dom_hidden: int = 64, dropout: float = 0.3):
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

        self.label_classifier = nn.Linear(hidden2, n_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden2, dom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dom_hidden, 2),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        features      = self.feature_extractor(x)
        label_logits  = self.label_classifier(features)
        domain_logits = self.domain_classifier(grad_reverse(features, alpha))
        return label_logits, domain_logits, features
