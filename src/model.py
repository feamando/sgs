"""
SGS Encoder — Full model with multi-pass rendering.

Phase 1:   Single-head centroid query, diagonal covariance
Phase 1.5: Multi-head learned query, IDF init, fair baselines
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
    multi-pass Gaussian rendering with optional multi-head viewpoints.
    """

    def __init__(
        self,
        vocab_size: int,
        d_s: int = 64,
        d_f: int = 300,
        n_passes: int = 4,
        tau_init: float = 64.0,
        n_heads: int = 1,
        log_passes: bool = False,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_f = d_f
        self.n_passes = n_passes
        self.n_heads = n_heads
        self.log_passes = log_passes
        self._pass_logs = []

        # Gaussian vocabulary
        self.vocab = SemanticGaussianVocab(vocab_size, d_s, d_f)

        # Learned temperature
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

        # Positional modulation
        self.pos_embed_mu = nn.Embedding(512, d_s)
        self.pos_embed_alpha = nn.Embedding(512, 1)

        # Multi-head viewpoint projections (Phase 1.5)
        if n_heads > 1:
            self.query_proj = nn.ModuleList([
                nn.Linear(d_s, d_s, bias=True) for _ in range(n_heads)
            ])
            self.output_proj = nn.Linear(n_heads * d_f, d_f)
            # Initialize close to identity
            for proj in self.query_proj:
                nn.init.eye_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            self.query_proj = None
            self.output_proj = None

        # Per-pass update networks (for passes 2..P)
        if n_passes > 1:
            self.mu_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_f + d_f, d_s),
                    nn.Tanh(),
                )
                for _ in range(n_passes - 1)
            ])
            self.alpha_gate = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_f + d_f, 1),
                    nn.Sigmoid(),
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

    def _compute_centroid(self, mu, mask):
        """Compute masked centroid of means."""
        if mask is not None:
            masked_mu = mu * mask.float().unsqueeze(-1)
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            return masked_mu.sum(dim=1) / lengths
        return mu.mean(dim=1)

    def _render_single_head(self, query, mu, log_var, alpha, features, mask):
        """Render from a single query viewpoint."""
        K = gaussian_kernel_diag(query, mu, log_var, self.tau)
        if mask is not None:
            K = K * mask.float()
        meaning, weights = render(features, alpha, K, return_weights=True)
        return meaning, weights

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        self._pass_logs = []

        # 1. Activate Gaussians
        mu, log_var, alpha, features = self.vocab.get_params(token_ids)

        # 2. Positional modulation
        positions = torch.arange(seq_len, device=token_ids.device).clamp(max=511)
        mu = mu + self.pos_embed_mu(positions).unsqueeze(0)
        alpha_mod = torch.sigmoid(
            self.pos_embed_alpha(positions).squeeze(-1)
        ).unsqueeze(0)
        alpha = alpha * alpha_mod

        # 3. Padding mask
        if mask is not None:
            alpha = alpha * mask.float()

        # 4. Multi-pass rendering
        for p in range(self.n_passes):
            centroid = self._compute_centroid(mu, mask)

            if self.n_heads > 1 and self.query_proj is not None:
                # Multi-head: each head has its own projected query
                head_meanings = []
                for h in range(self.n_heads):
                    query_h = self.query_proj[h](centroid)
                    meaning_h, _ = self._render_single_head(
                        query_h, mu, log_var, alpha, features, mask,
                    )
                    head_meanings.append(meaning_h)
                meaning = self.output_proj(torch.cat(head_meanings, dim=-1))
            else:
                # Single head: query = centroid
                meaning, weights = self._render_single_head(
                    centroid, mu, log_var, alpha, features, mask,
                )

            # Per-pass diagnostics
            if self.log_passes:
                with torch.no_grad():
                    self._pass_logs.append({
                        'pass': p,
                        'alpha_mean': alpha.mean().item(),
                        'alpha_std': alpha.std().item(),
                        'mu_norm': mu.norm(dim=-1).mean().item(),
                        'feat_norm': features.norm(dim=-1).mean().item(),
                        'tau': self.tau.item(),
                    })

            # Update Gaussian parameters for next pass
            if p < self.n_passes - 1:
                context = meaning.unsqueeze(1).expand_as(features)
                combined = torch.cat([features, context], dim=-1)

                mu = mu + self.mu_update[p](combined)
                gate = self.alpha_gate[p](combined).squeeze(-1)
                alpha = alpha * gate
                features = features + self.ffn[p](combined)

                if mask is not None:
                    alpha = alpha * mask.float()

        return meaning


class SGSSimilarityModel(nn.Module):
    """STS-B model: encode two sentences, compute cosine similarity."""

    def __init__(self, encoder: SGSEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        meaning_a = self.encoder(ids_a, mask_a)
        meaning_b = self.encoder(ids_b, mask_b)
        cos_sim = nn.functional.cosine_similarity(meaning_a, meaning_b, dim=-1)
        return cos_sim * 5.0


# ═══════════════════════════════════════════════════════════
# Baselines and ablation variants
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
        q_a = render_mean_pool(mu_a, mask_a)
        q_b = render_mean_pool(mu_b, mask_b)
        mean_a = render_softmax_attention(feat_a, q_a, mu_a, mask_a)
        mean_b = render_softmax_attention(feat_b, q_b, mu_b, mask_b)
        cos = nn.functional.cosine_similarity(mean_a, mean_b, dim=-1)
        return cos * 5.0


class FairSoftmaxModel(nn.Module):
    """
    Fair softmax baseline — matched architecture to SGS-2pass.

    Includes position embeddings, learned temperature, 2-layer
    softmax attention + FFN. Same parameter budget as SGS.
    """

    def __init__(self, vocab: SemanticGaussianVocab, d_s: int = 64, d_f: int = 300, n_layers: int = 2):
        super().__init__()
        self.vocab = vocab
        self.d_s = d_s
        self.d_f = d_f

        # Position embeddings (same as SGS)
        self.pos_embed = nn.Embedding(512, d_s)
        nn.init.normal_(self.pos_embed.weight, std=0.01)

        # Learned temperature (same as SGS)
        self.log_tau = nn.Parameter(torch.tensor(float(d_s)).log())

        # Multi-layer softmax attention + FFN
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(d_f, d_s),
                'k_proj': nn.Linear(d_s, d_s),
                'ffn': nn.Sequential(
                    nn.LayerNorm(d_f + d_f),
                    nn.Linear(d_f + d_f, d_f * 2),
                    nn.GELU(),
                    nn.Linear(d_f * 2, d_f),
                ),
            }))

    def _encode(self, token_ids, mask):
        mu, log_var, alpha, features = self.vocab.get_params(token_ids)
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).clamp(max=511)
        mu = mu + self.pos_embed(positions).unsqueeze(0)

        for layer in self.layers:
            # Compute query from current features
            centroid = render_mean_pool(features, mask)
            query = layer['q_proj'](centroid)  # [batch, d_s]
            keys = layer['k_proj'](mu)  # [batch, n, d_s]

            # Softmax attention
            tau = self.log_tau.exp()
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (tau ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), float('-inf'))
            weights = torch.softmax(scores, dim=-1)

            # Weighted sum
            context = (weights.unsqueeze(-1) * features).sum(dim=1)

            # FFN update on features
            ctx_broadcast = context.unsqueeze(1).expand_as(features)
            combined = torch.cat([features, ctx_broadcast], dim=-1)
            features = features + layer['ffn'](combined)

        # Final pooling via attention
        centroid = render_mean_pool(features, mask)
        return centroid

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        mean_a = self._encode(ids_a, mask_a)
        mean_b = self._encode(ids_b, mask_b)
        cos = nn.functional.cosine_similarity(mean_a, mean_b, dim=-1)
        return cos * 5.0


class GaussianKernelSoftmaxModel(nn.Module):
    """
    M2 ablation: Gaussian kernel VALUES fed through softmax (not rendering).

    Isolates whether the advantage is from the Gaussian kernel function
    or from the alpha-compositing mechanism.

    weights = softmax(K(q, μ_i, Σ_i)) instead of rendering equation.
    Same kernel, same Gaussians, but softmax normalization instead of transmittance.
    """

    def __init__(self, vocab: SemanticGaussianVocab, d_s: int = 64, tau_init: float = 64.0):
        super().__init__()
        self.vocab = vocab
        self.d_s = d_s
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
        self.pos_embed_mu = nn.Embedding(512, d_s)
        nn.init.normal_(self.pos_embed_mu.weight, std=0.01)

    @property
    def tau(self):
        return self.log_tau.exp()

    def _encode(self, token_ids, mask):
        mu, log_var, _, features = self.vocab.get_params(token_ids)
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).clamp(max=511)
        mu = mu + self.pos_embed_mu(positions).unsqueeze(0)

        # Query = centroid (same as SGS)
        if mask is not None:
            masked_mu = mu * mask.float().unsqueeze(-1)
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            query = masked_mu.sum(dim=1) / lengths
        else:
            query = mu.mean(dim=1)

        # Evaluate Gaussian kernel (same as SGS)
        K = gaussian_kernel_diag(query, mu, log_var, self.tau)

        # SOFTMAX over kernel values instead of alpha-compositing
        if mask is not None:
            K = K.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(K, dim=-1)  # [batch, n]

        meaning = (weights.unsqueeze(-1) * features).sum(dim=1)
        return meaning

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        a = self._encode(ids_a, mask_a)
        b = self._encode(ids_b, mask_b)
        cos = nn.functional.cosine_similarity(a, b, dim=-1)
        return cos * 5.0


class HybridSGSSoftmaxModel(nn.Module):
    """
    M6: SGS rendering for pass 1, softmax attention for pass 2.

    Combines SGS's inductive bias (structural composition) with
    softmax's capacity (flexible reweighting).
    """

    def __init__(self, vocab: SemanticGaussianVocab, d_s: int = 64, d_f: int = 300):
        super().__init__()
        self.vocab = vocab
        self.d_s = d_s
        self.d_f = d_f

        # Shared
        self.log_tau = nn.Parameter(torch.tensor(float(d_s)).log())
        self.pos_embed_mu = nn.Embedding(512, d_s)
        self.pos_embed_alpha = nn.Embedding(512, 1)
        nn.init.normal_(self.pos_embed_mu.weight, std=0.01)
        nn.init.zeros_(self.pos_embed_alpha.weight)

        # Pass 1 → Pass 2 bridge: update features with SGS context
        self.bridge_ffn = nn.Sequential(
            nn.LayerNorm(d_f + d_f),
            nn.Linear(d_f + d_f, d_f * 2),
            nn.GELU(),
            nn.Linear(d_f * 2, d_f),
        )

        # Pass 2: softmax attention layer
        self.attn_q = nn.Linear(d_f, d_s)
        self.attn_k = nn.Linear(d_s, d_s)
        self.attn_ffn = nn.Sequential(
            nn.LayerNorm(d_f + d_f),
            nn.Linear(d_f + d_f, d_f * 2),
            nn.GELU(),
            nn.Linear(d_f * 2, d_f),
        )

    @property
    def tau(self):
        return self.log_tau.exp()

    def _encode(self, token_ids, mask):
        mu, log_var, alpha, features = self.vocab.get_params(token_ids)
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).clamp(max=511)
        mu = mu + self.pos_embed_mu(positions).unsqueeze(0)
        alpha_mod = torch.sigmoid(self.pos_embed_alpha(positions).squeeze(-1)).unsqueeze(0)
        alpha = alpha * alpha_mod
        if mask is not None:
            alpha = alpha * mask.float()

        # === Pass 1: SGS rendering ===
        centroid = mu.mean(dim=1) if mask is None else (
            (mu * mask.float().unsqueeze(-1)).sum(dim=1) /
            mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        )
        K = gaussian_kernel_diag(centroid, mu, log_var, self.tau)
        if mask is not None:
            K = K * mask.float()
        sgs_meaning, _ = render(features, alpha, K)

        # Bridge: update features with SGS context
        context = sgs_meaning.unsqueeze(1).expand_as(features)
        combined = torch.cat([features, context], dim=-1)
        features_v2 = features + self.bridge_ffn(combined)

        # === Pass 2: softmax attention ===
        query = self.attn_q(sgs_meaning)  # [batch, d_s]
        keys = self.attn_k(mu)  # [batch, n, d_s]
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (self.d_s ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn_meaning = (weights.unsqueeze(-1) * features_v2).sum(dim=1)

        # Final: residual combination
        final = sgs_meaning + self.attn_ffn(
            torch.cat([sgs_meaning, attn_meaning], dim=-1)
        )
        return final

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        a = self._encode(ids_a, mask_a)
        b = self._encode(ids_b, mask_b)
        cos = nn.functional.cosine_similarity(a, b, dim=-1)
        return cos * 5.0


class NoTransmittanceModel(nn.Module):
    """Ablation: rendering without transmittance (T_i = 1 for all)."""

    def __init__(self, encoder: SGSEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, ids_a, mask_a, ids_b, mask_b):
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
        weights = alpha * K
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        meaning = (weights.unsqueeze(-1) * features).sum(dim=1)
        return meaning
