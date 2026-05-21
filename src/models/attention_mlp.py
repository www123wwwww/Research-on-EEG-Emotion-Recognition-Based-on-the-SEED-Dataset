"""
Improved model: MLP with Band Attention + Channel Attention (Direction 1).

Architecture follows the Squeeze-and-Excitation (SE-Net) paradigm:
  - Squeeze: aggregate global statistics across the complementary axis
  - Excitation: two-layer FC bottleneck → Sigmoid gate
  - Scale: element-wise multiplication (non-reductive)

Input shape: (N, 5, 62) — 5 frequency bands × 62 EEG channels

Pipeline:
  1. BandAttention   — squeeze over channels → excitation → gate bands
                       keeps shape (N, 5, 62),  returns weights (N, 5)
  2. ChannelAttention — squeeze over bands → excitation → gate channels
                       keeps shape (N, 5, 62),  returns weights (N, 62)
  3. Flatten → (N, 310)
  4. MLP  310 → 256 → 128 → 3   (identical to baseline for fair comparison)

Why Sigmoid (not Softmax)?
  Sigmoid allows each band/channel to be independently up- or down-weighted.
  Softmax forces a zero-sum competition and collapses interpretability when
  one band is globally dominant. Sigmoid weights ∈ (0,1) are directly
  readable as "how much this band/channel is amplified."

Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandAttention(nn.Module):
    """
    SE-style attention over frequency bands.

    Squeeze : global average pool across channels → (N, 5)
    Excite  : FC(5→16) → ReLU → FC(16→5) → Sigmoid  → gates ∈ (0,1)^5
    Scale   : x = x * gates.unsqueeze(-1)            → (N, 5, 62)

    Input:  (N, 5, 62)
    Output: re-weighted (N, 5, 62),  gates (N, 5)
    """

    def __init__(self, n_bands: int = 5, reduction: int = 2):
        super().__init__()
        mid = max(n_bands // reduction, 2)
        self.excitation = nn.Sequential(
            nn.Linear(n_bands, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_bands),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (N, 5, 62)
        squeezed = x.mean(dim=2)               # (N, 5)  — squeeze over channels
        gates = self.excitation(squeezed)       # (N, 5)  — excitation
        out = x * gates.unsqueeze(-1)           # (N, 5, 62)  — scale
        return out, gates


class ChannelAttention(nn.Module):
    """
    SE-style attention over EEG channels (electrodes).

    Squeeze : global average pool across bands → (N, 62)
    Excite  : FC(62→16) → ReLU → FC(16→62) → Sigmoid  → gates ∈ (0,1)^62
    Scale   : x = x * gates.unsqueeze(1)              → (N, 5, 62)

    Input:  (N, 5, 62)
    Output: re-weighted (N, 5, 62),  gates (N, 62)
    """

    def __init__(self, n_channels: int = 62, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 8)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (N, 5, 62)
        squeezed = x.mean(dim=1)               # (N, 62)  — squeeze over bands
        gates = self.excitation(squeezed)       # (N, 62)  — excitation
        out = x * gates.unsqueeze(1)            # (N, 5, 62)  — scale
        return out, gates


class AttentionMLP(nn.Module):
    """
    Full model: BandAttention → ChannelAttention → flatten → MLP.

    The MLP head (310→256→128→3) is identical to baseline MLPModel,
    so any accuracy difference is solely attributable to the attention module.

    forward() returns (logits, band_gates, channel_gates) for visualisation.
    """

    def __init__(self, n_bands: int = 5, n_channels: int = 62,
                 hidden1: int = 256, hidden2: int = 128,
                 n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.band_attn    = BandAttention(n_bands)
        self.channel_attn = ChannelAttention(n_channels)
        feature_dim = n_bands * n_channels   # 310, same as baseline

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: (N, 5, 62)
        x, band_gates = self.band_attn(x)        # (N, 5, 62), (N, 5)
        x, chan_gates = self.channel_attn(x)     # (N, 5, 62), (N, 62)
        x = x.reshape(x.size(0), -1)             # (N, 310)
        logits = self.classifier(x)               # (N, 3)
        return logits, band_gates, chan_gates
