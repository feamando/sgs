"""
Alpha-compositing rendering equation (Atom A3).

Meaning(q) = Σ_i f_i * w_i
where w_i = α_i * K_i * T_i
and   T_i = Π_{j<i} (1 - α_j * K_j)

Proven properties (Lean 4 verified):
- Σ w_i = 1 - T_{n+1} ≤ 1
- T is monotonically non-increasing
- Gradients flow to all parameters when w_i > 0
- This scheme is strictly more expressive than softmax attention
"""

import torch


def render(
    features: torch.Tensor,
    alpha: torch.Tensor,
    kernel_vals: torch.Tensor,
    return_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Alpha-compositing rendering equation.

    Args:
        features:    [batch, n, d_f] — feature vectors per Gaussian
        alpha:       [batch, n] — opacity values in (0, 1)
        kernel_vals: [batch, n] — kernel evaluations K(q, μ_i, Σ_i)
        return_weights: whether to return the blending weights

    Returns:
        meaning: [batch, d_f] — rendered meaning vector
        weights: [batch, n] or None — blending weights (if requested)
    """
    # Effective opacity: a_i = α_i * K_i
    eff_opacity = alpha * kernel_vals  # [batch, n]

    # Clamp for numerical stability (prevent log(0))
    eff_opacity = eff_opacity.clamp(min=0.0, max=1.0 - 1e-6)

    # Transmittance via log-cumsum for numerical stability
    # T_i = exp(Σ_{j<i} log(1 - a_j))
    log_one_minus_a = torch.log1p(-eff_opacity)  # [batch, n] — log(1 - a_i)
    log_cum = torch.cumsum(log_one_minus_a, dim=1)  # [batch, n]

    # Shift right: T_1 = 1 (log(1) = 0), T_i = exp(sum of log(1-a_j) for j < i)
    log_transmittance = torch.cat([
        torch.zeros_like(log_cum[:, :1]),  # T_1 = exp(0) = 1
        log_cum[:, :-1],                    # T_2 = exp(log(1-a_1)), etc.
    ], dim=1)
    transmittance = torch.exp(log_transmittance)  # [batch, n]

    # Blending weights: w_i = a_i * T_i
    weights = eff_opacity * transmittance  # [batch, n]

    # Weighted sum of features
    meaning = (weights.unsqueeze(-1) * features).sum(dim=1)  # [batch, d_f]

    if return_weights:
        return meaning, weights
    return meaning, None


def render_mean_pool(
    features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Simple mean-pooling baseline (no rendering equation).

    Args:
        features: [batch, n, d_f]
        mask:     [batch, n] — 1 for real tokens, 0 for padding

    Returns:
        meaning: [batch, d_f]
    """
    if mask is not None:
        features = features * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return features.sum(dim=1) / lengths
    return features.mean(dim=1)


def render_softmax_attention(
    features: torch.Tensor,
    query: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Softmax attention baseline (same representations, different composition).

    Args:
        features: [batch, n, d_f]
        query:    [batch, d_s] — query vector
        mu:       [batch, n, d_s] — Gaussian means as keys
        mask:     [batch, n] — padding mask

    Returns:
        meaning: [batch, d_f]
    """
    d_s = query.size(-1)
    # Dot-product scores
    scores = torch.bmm(
        mu, query.unsqueeze(-1)
    ).squeeze(-1) / (d_s ** 0.5)  # [batch, n]

    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float('-inf'))

    weights = torch.softmax(scores, dim=-1)  # [batch, n]
    meaning = (weights.unsqueeze(-1) * features).sum(dim=1)  # [batch, d_f]
    return meaning
