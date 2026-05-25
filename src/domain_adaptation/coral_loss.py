"""
CORAL Loss (Correlation Alignment).

Aligns the second-order statistics (covariance matrices) of source and target
domain features so the model learns domain-invariant representations.

Reference:
    Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain
    Adaptation", ECCV Workshop 2016.

Formula:
    L_CORAL = ||C_s - C_t||_F^2 / (4 * d^2)

where C_s, C_t are the covariance matrices of source and target features,
d is the feature dimension, and ||·||_F is the Frobenius norm.

Why divide by 4d²?
    Normalises the loss to be scale-invariant with respect to feature
    dimension, so lambda_coral doesn't need re-tuning when d changes.
"""

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL loss between source and target feature batches.

    Args:
        source: (N_s, d)  features from source domain (labeled, training subjects)
        target: (N_t, d)  features from target domain (unlabeled, test subject)

    Returns:
        Scalar tensor. Returns 0.0 if either batch has fewer than 2 samples
        (covariance is undefined for a single sample).
    """
    ns, d = source.shape
    nt = target.shape[0]

    if ns < 2 or nt < 2:
        return torch.tensor(0.0, device=source.device, requires_grad=False)

    # Remove mean (center the features)
    xs = source - source.mean(dim=0, keepdim=True)   # (N_s, d)
    xt = target - target.mean(dim=0, keepdim=True)   # (N_t, d)

    # Unbiased covariance matrices
    Cs = (xs.T @ xs) / (ns - 1)   # (d, d)
    Ct = (xt.T @ xt) / (nt - 1)   # (d, d)

    # Frobenius norm squared, normalised by 4d²
    loss = ((Cs - Ct) ** 2).sum() / (4.0 * d * d)
    return loss
