"""
MMD Loss — Maximum Mean Discrepancy (multi-kernel RBF variant).

MMD measures how different two distributions are by comparing their means
in a high-dimensional kernel space (RKHS).  Unlike CORAL (which only aligns
covariance matrices), multi-kernel MMD captures distributional differences
at multiple scales simultaneously.

Biased estimator:
    MMD²(X_s, X_t) = E[k(xs,xs')] - 2·E[k(xs,xt)] + E[k(xt,xt')]

Kernel:
    k(x,y) = Σ_{σ} exp( -||x-y||² / (2σ²) )   (sum over multiple bandwidths)

Why multiple bandwidths?
    A single bandwidth may miss structure at other scales.
    Summing kernels over [0.5, 1, 2, 5, 10] covers a wide range without
    needing to tune σ per dataset.

Reference:
    Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
    Long et al., "Learning Transferable Features with Deep Adaptation Networks",
    ICML 2015.
"""

import torch


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient pairwise squared Euclidean distances.
    Uses the identity ||x-y||² = ||x||² + ||y||² - 2·<x,y>.

    Args:
        x: (n, d)
        y: (m, d)
    Returns:
        (n, m)  dist[i,j] = ||x[i] - y[j]||²
    """
    xx = (x * x).sum(dim=1, keepdim=True)      # (n, 1)
    yy = (y * y).sum(dim=1, keepdim=True).T    # (1, m)
    xy = x @ y.T                               # (n, m)
    return (xx + yy - 2.0 * xy).clamp(min=0.0)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return torch.exp(-_pairwise_sq_dist(x, y) / (2.0 * bandwidth ** 2))


def mmd_loss(source: torch.Tensor, target: torch.Tensor,
             bandwidths: list = None) -> torch.Tensor:
    """
    Compute multi-kernel MMD² between source and target feature batches.

    Args:
        source     : (N_s, d)  labeled source domain features
        target     : (N_t, d)  unlabeled target domain features
        bandwidths : list of RBF σ values; defaults to [0.5, 1.0, 2.0, 5.0, 10.0]

    Returns:
        Scalar tensor (MMD²).  Returns 0.0 for degenerate batches (< 2 samples).
    """
    if source.size(0) < 2 or target.size(0) < 2:
        return torch.tensor(0.0, device=source.device)

    if bandwidths is None:
        bandwidths = [0.5, 1.0, 2.0, 5.0, 10.0]

    # Accumulate kernel matrices over all bandwidths
    K_ss = sum(_rbf_kernel(source, source, bw) for bw in bandwidths)
    K_tt = sum(_rbf_kernel(target, target, bw) for bw in bandwidths)
    K_st = sum(_rbf_kernel(source, target, bw) for bw in bandwidths)

    # Biased MMD² estimator
    return K_ss.mean() - 2.0 * K_st.mean() + K_tt.mean()
