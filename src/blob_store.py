"""
BlobStore — Knowledge blob storage and retrieval for Hierarchical SGS.

Each blob is a Gaussian in the same splatting space as word tokens:
  - mu:      [n_blobs, d_s]  centroid meaning
  - log_var: [n_blobs, d_s]  semantic breadth (diagonal covariance)
  - alpha:   [n_blobs]       confidence / authority
  - features:[n_blobs, d_f]  semantic content

Two-pass rendering:
  Pass 1 (blobs): renders top-k retrieved blobs via alpha-compositing
  Pass 2 (words): renders word tokens in remaining transmittance capacity

Proven in Lean 4:
  H1: Two-pass = single-pass (partition equivalence)
  H2: T_max cap preserves expressiveness (relative weights unchanged)
  H4: Total weight is permutation-invariant (ordering is a heuristic)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlobStore(nn.Module):
    """
    Gaussian knowledge blob store with retrieval and rendering.

    Blobs are retrieved by Gaussian kernel distance to a query,
    then rendered via alpha-compositing before word-level rendering.
    """

    def __init__(
        self,
        n_blobs: int,
        d_s: int,
        d_f: int,
        k: int = 8,
        t_max: float = 0.3,
        tau_init: float = 128.0,
    ):
        super().__init__()
        self.n_blobs = n_blobs
        self.d_s = d_s
        self.d_f = d_f
        self.k = k
        self.t_max = t_max

        # Blob Gaussian parameters
        self.mu = nn.Parameter(torch.randn(n_blobs, d_s) * 0.02)
        self.log_var = nn.Parameter(torch.zeros(n_blobs, d_s))
        self.raw_alpha = nn.Parameter(torch.zeros(n_blobs))
        self.features = nn.Parameter(torch.randn(n_blobs, d_f) * 0.02)

        # Shared temperature (can be tied to model's tau)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def init_from_clusters(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        alpha: torch.Tensor,
        features: torch.Tensor,
    ):
        """Initialize blob parameters from pre-computed clusters.

        Args:
            mu:       [n_blobs, d_s] cluster centroids
            log_var:  [n_blobs, d_s] log variance from cluster spread
            alpha:    [n_blobs] confidence scores (pre-sigmoid)
            features: [n_blobs, d_f] mean feature vectors
        """
        assert mu.shape[0] == self.n_blobs
        with torch.no_grad():
            self.mu.copy_(mu)
            self.log_var.copy_(log_var)
            self.raw_alpha.copy_(alpha)
            self.features.copy_(features)

    def _gaussian_kernel(
        self,
        query: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian kernel K(query, mu, Sigma) with diagonal covariance.

        Args:
            query:   [B, d_s]
            mu:      [K, d_s]   (K = n_blobs or top-k)
            log_var: [K, d_s]
        Returns:
            K: [B, K] kernel values in (0, 1]
        """
        inv_var = torch.exp(-log_var)                    # [K, d_s]
        diff = query.unsqueeze(1) - mu.unsqueeze(0)      # [B, K, d_s]
        mahal = (diff * diff * inv_var.unsqueeze(0)).sum(-1)  # [B, K]
        return torch.exp(-0.5 * mahal / self.tau)

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k blobs by effective weight (K * alpha).

        Args:
            query: [B, d_s] query in splatting space

        Returns:
            indices: [B, k] top-k blob indices
            scores:  [B, k] effective scores (K * alpha)
        """
        # Compute kernel over all blobs
        K = self._gaussian_kernel(query, self.mu, self.log_var)  # [B, n_blobs]
        alpha = self.alpha                                        # [n_blobs]
        scores = K * alpha.unsqueeze(0)                           # [B, n_blobs]

        # Top-k retrieval
        top_scores, top_idx = scores.topk(
            min(self.k, self.n_blobs), dim=-1
        )  # [B, k]
        return top_idx, top_scores

    def render(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and render blobs via alpha-compositing.

        Args:
            query: [B, d_s] query in splatting space

        Returns:
            blob_meaning: [B, d_f] rendered blob features
            t_residual:   [B] residual transmittance after blob rendering
        """
        B = query.shape[0]
        device = query.device

        # Retrieve top-k blobs
        top_idx, top_scores = self.retrieve(query)     # [B, k]
        k = top_idx.shape[1]

        # Gather blob features for top-k
        # top_idx: [B, k] → index into self.features [n_blobs, d_f]
        blob_feats = self.features[top_idx.reshape(-1)].reshape(B, k, self.d_f)

        # Compute full kernel + alpha for top-k blobs
        blob_mu = self.mu[top_idx.reshape(-1)].reshape(B, k, self.d_s)
        blob_lv = self.log_var[top_idx.reshape(-1)].reshape(B, k, self.d_s)

        # Re-evaluate kernel precisely for top-k
        inv_var = torch.exp(-blob_lv)                                    # [B, k, d_s]
        diff = query.unsqueeze(1) - blob_mu                             # [B, k, d_s]
        mahal = (diff * diff * inv_var).sum(-1)                          # [B, k]
        K_topk = torch.exp(-0.5 * mahal / self.tau)                     # [B, k]

        blob_alpha = self.alpha[top_idx.reshape(-1)].reshape(B, k)      # [B, k]

        # Effective opacity (clamped by t_max budget)
        eff_opacity = (blob_alpha * K_topk).clamp(max=self.t_max / k)   # [B, k]

        # Alpha-compositing: compute transmittance and weights
        # T_j = Π_{i<j} (1 - eff_opacity_i)
        log_1m = torch.log1p(-eff_opacity.clamp(max=1.0 - 1e-6))       # [B, k]
        log_cum = log_1m.cumsum(dim=-1)                                  # [B, k]
        log_T = torch.cat(
            [torch.zeros(B, 1, device=device), log_cum[:, :-1]], dim=-1
        )                                                                 # [B, k]
        T = log_T.exp()                                                   # [B, k]

        # Rendering weights
        weights = eff_opacity * T                                         # [B, k]

        # Render blob meaning
        blob_meaning = (weights.unsqueeze(-1) * blob_feats).sum(dim=1)   # [B, d_f]

        # Residual transmittance
        t_residual = log_cum[:, -1].exp()                                 # [B]

        return blob_meaning, t_residual

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        return {
            "blob_mu": self.mu.numel(),
            "blob_log_var": self.log_var.numel(),
            "blob_raw_alpha": self.raw_alpha.numel(),
            "blob_features": self.features.numel(),
            "blob_log_tau": 1,
            "total": self.count_parameters(),
        }
