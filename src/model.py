"""
SGS Encoder — Full model with multi-pass rendering.

Architecture:
  Input tokens → Activate Gaussians → Multi-pass rendering → Meaning vector

Each pass: Render → Context broadcast → Update (μ, α, features)
"""

import torch
import torch.nn as nn

from .gaussian import SemanticGaussianVocab
from .kernel import gaussian_kernel_diag
from .rendering import render, render_mean_pool, render_softmax_attention


class SGSEncoder(nn.Module):
    """
    Semantic Gaussian Splatting encoder.

    Encodes a token sequence into a fixed-size meaning vector via
    multi-pass Gaussian rendering.
    """

    def __init__(
        self,
        vocab_size: int,
        d_s: int = 64,
        d_f: int = 300,
        n_passes: int = 4,
        tau_init: float = 64.0,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_f = d_f
        self.n_passes = n_passes

        # Gaussian vocabulary
        self.vocab = SemanticGaussianVocab(vocab_size, d_s, d_f)

        # Learned temperature
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

        # Positional modulation: small displacement per position
        self.pos_embed_mu = nn.Embedding(512, d_s)  # position → μ offset
        self.pos_embed_alpha = nn.Embedding(512, 1)  # position → α modulation

        # Per-pass update networks (for passes 2..P)
        if n_passes > 1:
            self.mu_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_f + d_f, d_s),
                    nn.Tanh(),  # Bounded output — proven necessary (Claim 5.1)
                )
                for _ in range(n_passes - 1)
            ])
            self.alpha_gate = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_f + d_f, 1),
                    nn.Sigmoid(),  # Gate ∈ (0,1) — proven strictly decreasing (Claim 5.2)
                )
                for _ in range(n_passes - 1)
            ])
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_f + d_f),
                    nn.Linear(d_f + d_f, d_f * 2),
                    nn.GELU(),
                    nn.Linear(d_f * 2, d_f),
                )
                for _ in range(n_passes - 1)
            ])

        # Initialize positional embeddings small
        nn.init.normal_(self.pos_embed_mu.weight, std=0.01)
        nn.init.zeros_(self.pos_embed_alpha.weight)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a token sequence via multi-pass Gaussian rendering.

        Args:
            token_ids: [batch, seq_len] — vocabulary indices
            mask:      [batch, seq_len] — 1 for real tokens, 0 for padding

        Returns:
            meaning: [batch, d_f] — rendered sentence meaning
        """
        batch_size, seq_len = token_ids.shape

        # 1. Activate Gaussians from vocabulary
        mu, log_var, alpha, features = self.vocab.get_params(token_ids)

        # 2. Positional modulation
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.clamp(max=511)  # Cap at max position
        mu = mu + self.pos_embed_mu(positions).unsqueeze(0)
        alpha_mod = torch.sigmoid(
            self.pos_embed_alpha(positions).squeeze(-1)
        ).unsqueeze(0)
        alpha = alpha * alpha_mod

        # 3. Apply padding mask to alpha (padding tokens get zero opacity)
        if mask is not None:
            alpha = alpha * mask.float()

        # 4. Multi-pass rendering
        for p in range(self.n_passes):
            # Query point: centroid of means (simple; works for Phase 1)
            if mask is not None:
                masked_mu = mu * mask.float().unsqueeze(-1)
                lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                query = masked_mu.sum(dim=1) / lengths  # [batch, d_s]
            else:
                query = mu.mean(dim=1)  # [batch, d_s]

            # Evaluate kernel
            K = gaussian_kernel_diag(query, mu, log_var, self.tau)  # [batch, n]

            # Mask kernel for padding
            if mask is not None:
                K = K * mask.float()

            # Render
            meaning, _ = render(features, alpha, K)  # [batch, d_f]

            # Update Gaussian parameters for next pass (not on last pass)
            if p < self.n_passes - 1:
                # Context: broadcast rendered meaning to all tokens
                context = meaning.unsqueeze(1).expand_as(features)  # [batch, n, d_f]
                combined = torch.cat([features, context], dim=-1)  # [batch, n, 2*d_f]

                # Update position (bounded by tanh)
                mu = mu + self.mu_update[p](combined)

                # Update opacity (gated — can only decrease)
                gate = self.alpha_gate[p](combined).squeeze(-1)  # [batch, n]
                alpha = alpha * gate

                # Update features (residual)
                features = features + self.ffn[p](combined)

                # Re-apply padding mask
                if mask is not None:
                    alpha = alpha * mask.float()

        return meaning


class SGSSimilarityModel(nn.Module):
    """
    STS-B model: encode two sentences, compute cosine similarity.
    """

    def __init__(self, encoder: SGSEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        ids_a: torch.Tensor,
        mask_a: torch.Tensor,
        ids_b: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between sentence pairs.

        Returns:
            similarity: [batch] — cosine similarity scaled to [0, 5]
        """
        meaning_a = self.encoder(ids_a, mask_a)  # [batch, d_f]
        meaning_b = self.encoder(ids_b, mask_b)  # [batch, d_f]

        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(meaning_a, meaning_b, dim=-1)

        # Scale to [0, 5] range (STS-B scores)
        return cos_sim * 5.0


# ═══════════════════════════════════════════════════════════
# Ablation variants
# ═══════════════════════════════════════════════════════════

class MeanPoolModel(nn.Module):
    """Baseline: mean-pool Gaussian features (no rendering)."""

    def __init__(self, vocab: SemanticGaussianVocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        _, _, _, feat_a = self.vocab.get_params(ids_a)
        _, _, _, feat_b = self.vocab.get_params(ids_b)
        mean_a = render_mean_pool(feat_a, mask_a)
        mean_b = render_mean_pool(feat_b, mask_b)
        cos = nn.functional.cosine_similarity(mean_a, mean_b, dim=-1)
        return cos * 5.0


class MeanPoolMuModel(nn.Module):
    """Baseline: mean-pool Gaussian means in splatting space."""

    def __init__(self, vocab: SemanticGaussianVocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        mu_a, _, _, _ = self.vocab.get_params(ids_a)
        mu_b, _, _, _ = self.vocab.get_params(ids_b)
        mean_a = render_mean_pool(mu_a, mask_a)
        mean_b = render_mean_pool(mu_b, mask_b)
        cos = nn.functional.cosine_similarity(mean_a, mean_b, dim=-1)
        return cos * 5.0


class SoftmaxAttentionModel(nn.Module):
    """Ablation: same Gaussians, but softmax attention instead of rendering."""

    def __init__(self, vocab: SemanticGaussianVocab, d_s: int = 64):
        super().__init__()
        self.vocab = vocab
        self.d_s = d_s

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        mu_a, _, _, feat_a = self.vocab.get_params(ids_a)
        mu_b, _, _, feat_b = self.vocab.get_params(ids_b)
        # Query = mean of means
        q_a = render_mean_pool(mu_a, mask_a)
        q_b = render_mean_pool(mu_b, mask_b)
        # Softmax attention over features using mu as keys
        mean_a = render_softmax_attention(feat_a, q_a, mu_a, mask_a)
        mean_b = render_softmax_attention(feat_b, q_b, mu_b, mask_b)
        cos = nn.functional.cosine_similarity(mean_a, mean_b, dim=-1)
        return cos * 5.0


class NoTransmittanceModel(nn.Module):
    """Ablation: rendering without transmittance (T_i = 1 for all)."""

    def __init__(self, encoder: SGSEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        # Override: render with uniform transmittance
        mean_a = self._render_no_T(ids_a, mask_a)
        mean_b = self._render_no_T(ids_b, mask_b)
        cos = nn.functional.cosine_similarity(mean_a, mean_b, dim=-1)
        return cos * 5.0

    def _render_no_T(self, token_ids, mask):
        mu, log_var, alpha, features = self.encoder.vocab.get_params(token_ids)
        if mask is not None:
            alpha = alpha * mask.float()
        query = render_mean_pool(mu, mask)
        K = gaussian_kernel_diag(query, mu, log_var, self.encoder.tau)
        if mask is not None:
            K = K * mask.float()
        # No transmittance: weights = alpha * K (no T factor)
        weights = alpha * K
        # Normalize to sum to 1 (otherwise scale is wrong)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        meaning = (weights.unsqueeze(-1) * features).sum(dim=1)
        return meaning
