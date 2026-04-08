"""
Gaussian kernel evaluation (Atom A2).

Computes K(q, μ, Σ) = exp(-0.5 * D_M / τ) where D_M is the Mahalanobis distance.
Supports diagonal covariance (Phase 1) and full Cholesky (Phase 2+).
"""

import torch
import torch.nn.functional as F


def gaussian_kernel_diag(
    query: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate Gaussian kernel with diagonal covariance.

    Args:
        query:   [batch, d_s] — query points in splatting space
        mu:      [batch, n, d_s] — Gaussian means
        log_var: [batch, n, d_s] — log-variance (Σ_ii = exp(log_var))
        tau:     scalar — learned temperature

    Returns:
        K: [batch, n] — kernel values in (0, 1]
    """
    # Expand query: [batch, 1, d_s]
    q = query.unsqueeze(1)

    # Difference: [batch, n, d_s]
    diff = q - mu

    # Inverse variance: 1/Σ_ii
    inv_var = torch.exp(-log_var)  # [batch, n, d_s]

    # Mahalanobis distance (diagonal case): Σ (diff_k^2 / var_k)
    mahal = (diff * diff * inv_var).sum(dim=-1)  # [batch, n]

    # Temperature-scaled kernel
    K = torch.exp(-0.5 * mahal / tau)  # [batch, n]

    return K


def gaussian_kernel_diag_pairwise(
    queries: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate kernel at EACH Gaussian's own position (for context broadcast).
    Query i = mu_i (each Gaussian queries the field at its own location).

    Args:
        queries: [batch, n, d_s] — one query per Gaussian (typically = mu)
        mu:      [batch, n, d_s] — Gaussian means
        log_var: [batch, n, d_s] — log-variance
        tau:     scalar — temperature

    Returns:
        K: [batch, n, n] — K[b, i, j] = kernel of Gaussian j evaluated at query i
    """
    # queries: [batch, n, 1, d_s]
    q = queries.unsqueeze(2)
    # mu: [batch, 1, n, d_s]
    m = mu.unsqueeze(1)

    diff = q - m  # [batch, n, n, d_s]
    inv_var = torch.exp(-log_var).unsqueeze(1)  # [batch, 1, n, d_s]

    mahal = (diff * diff * inv_var).sum(dim=-1)  # [batch, n, n]
    K = torch.exp(-0.5 * mahal / tau)

    return K
