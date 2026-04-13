"""
Causal SGS Language Model — Radiance Planck (100M).

Architecture: Semantic Gaussian Splatting with alpha-compositing rendering.
Each token is a Gaussian in splatting space. Multi-head, multi-pass
causal rendering composes meaning from visible tokens.

Key design:
  - Causal mask (diagonal=0): position t sees tokens 0..t, predicts t+1
  - Factored kernel: avoids materializing [B, L, L, d_s] diff tensors
  - Heads processed sequentially to limit VRAM
  - Transmittance via log-cumsum for numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGSLanguageModel(nn.Module):
    """
    Causal language model using Semantic Gaussian Splatting.

    At each position, multi-head alpha-compositing rendering composes
    a meaning vector from causally visible token Gaussians. Multi-pass
    refinement updates Gaussian parameters between rendering passes.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_s: int = 128,
        d_f: int = 1000,      # ~100M params with default settings
        n_passes: int = 3,
        n_heads: int = 4,
        max_len: int = 512,
        tau_init: float = 128.0,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_s = d_s
        self.d_f = d_f
        self.n_passes = n_passes
        self.n_heads = n_heads
        self.max_len = max_len

        # ── Gaussian vocabulary (learned from scratch) ──
        self.tok_mu = nn.Embedding(vocab_size, d_s)
        self.tok_log_var = nn.Embedding(vocab_size, d_s)
        self.tok_raw_alpha = nn.Embedding(vocab_size, 1)
        self.tok_features = nn.Embedding(vocab_size, d_f)

        # ── Position embeddings (added to μ in splatting space) ──
        self.pos_mu = nn.Embedding(max_len, d_s)

        # ── Learned temperature ──
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

        # ── Multi-head query projections ──
        self.query_proj = nn.ModuleList(
            [nn.Linear(d_s, d_s) for _ in range(n_heads)]
        )

        # ── Head combination → d_f ──
        self.head_proj = nn.Linear(n_heads * d_f, d_f)

        # ── Per-pass update networks (between rendering passes) ──
        self.mu_update = nn.ModuleList()
        self.alpha_gate = nn.ModuleList()
        self.pass_ffn = nn.ModuleList()
        for _ in range(n_passes - 1):
            self.mu_update.append(
                nn.Sequential(nn.Linear(d_f * 2, d_s), nn.Tanh())
            )
            self.alpha_gate.append(
                nn.Sequential(nn.Linear(d_f * 2, 1), nn.Sigmoid())
            )
            self.pass_ffn.append(
                nn.Sequential(
                    nn.LayerNorm(d_f * 2),
                    nn.Linear(d_f * 2, d_f * ffn_mult),
                    nn.GELU(),
                    nn.Linear(d_f * ffn_mult, d_f),
                )
            )

        # ── Output ──
        self.ln_f = nn.LayerNorm(d_f)
        self.lm_head = nn.Linear(d_f, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    # ────────────────────────────────────────────────────────────
    # Initialization
    # ────────────────────────────────────────────────────────────

    def _init_weights(self):
        # Token embeddings
        nn.init.normal_(self.tok_mu.weight, std=0.02)
        nn.init.zeros_(self.tok_log_var.weight)       # Σ = I
        nn.init.zeros_(self.tok_raw_alpha.weight)      # α = 0.5
        nn.init.normal_(self.tok_features.weight, std=0.02)

        # Position embeddings — small so they modulate, not dominate
        nn.init.normal_(self.pos_mu.weight, std=0.01)

        # Query projections — start near identity
        for proj in self.query_proj:
            nn.init.eye_(proj.weight)
            nn.init.zeros_(proj.bias)

        # Head projection — scaled for residual-like accumulation
        nn.init.normal_(
            self.head_proj.weight,
            std=0.02 / math.sqrt(2 * self.n_passes),
        )
        nn.init.zeros_(self.head_proj.bias)

        # Per-pass MLPs
        for modules in [self.mu_update, self.alpha_gate, self.pass_ffn]:
            for block in modules:
                for m in block.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        # LM head
        nn.init.normal_(self.lm_head.weight, std=0.02)

    # ────────────────────────────────────────────────────────────
    # Core ops
    # ────────────────────────────────────────────────────────────

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def _pairwise_kernel(
        self,
        queries: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Memory-efficient pairwise Gaussian kernel.

        Factored computation avoids materializing [B, L, L, d_s]:
          M[t,j] = q_t² · iv_j  −  2·q_t · (μ_j·iv_j)  +  μ_j²·iv_j

        Args:
            queries: [B, L, d_s]
            mu:      [B, L, d_s]
            log_var: [B, L, d_s]
        Returns:
            K: [B, L, L]  kernel values in (0, 1]
        """
        iv = torch.exp(-log_var)                               # [B, L, d_s]
        mu_iv = mu * iv                                        # [B, L, d_s]
        mu2_iv_sum = (mu * mu_iv).sum(-1)                      # [B, L]
        q_sq = queries * queries                               # [B, L, d_s]

        # Three [B, L, L] terms via bmm — no [B, L, L, d_s] intermediate
        term1 = torch.bmm(q_sq, iv.transpose(1, 2))           # [B, L, L]
        term2 = torch.bmm(queries, mu_iv.transpose(1, 2))     # [B, L, L]

        mahal = term1 - 2.0 * term2 + mu2_iv_sum.unsqueeze(1) # [B, L, L]
        return torch.exp(-0.5 * mahal / self.tau)

    def _causal_render(
        self,
        features: torch.Tensor,
        alpha: torch.Tensor,
        K: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal alpha-compositing rendering.

        For each position t, renders meaning from visible tokens:
          w[t,j] = α_j · K[t,j] · T[t,j]
          T[t,j] = Π_{k<j} (1 − α_k · K[t,k])   (only over visible k)
          meaning[t] = Σ_j w[t,j] · f[j]

        Args:
            features:    [B, L, d_f]
            alpha:       [B, L]
            K:           [B, L, L]
            causal_mask: [L, L]
        Returns:
            meaning: [B, L, d_f]
        """
        # Effective opacity per (query_pos, key_pos)
        eff_a = alpha.unsqueeze(1) * K                         # [B, L, L]
        eff_a = eff_a * causal_mask                            # zero out future
        eff_a = eff_a.clamp(min=0.0, max=1.0 - 1e-6)

        # Transmittance via log-cumsum (numerically stable)
        log_1ma = torch.log1p(-eff_a)                          # [B, L, L]
        log_cum = log_1ma.cumsum(dim=-1)                       # [B, L, L]

        # Shift right: T[t, 0] = 1
        log_T = torch.cat(
            [torch.zeros_like(log_cum[:, :, :1]), log_cum[:, :, :-1]],
            dim=-1,
        )
        T = log_T.exp()                                        # [B, L, L]

        # Blending weights and render
        weights = eff_a * T                                    # [B, L, L]
        meaning = torch.bmm(weights, features)                 # [B, L, d_f]
        return meaning

    # ────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Causal language model forward pass.

        Args:
            token_ids: [B, L] token indices
        Returns:
            logits: [B, L, vocab_size]
                    logits[:, t, :] predicts token at position t+1
        """
        B, L = token_ids.shape
        device = token_ids.device

        # ── Activate Gaussians ──
        mu = self.tok_mu(token_ids)                            # [B, L, d_s]
        log_var = self.tok_log_var(token_ids)                  # [B, L, d_s]
        alpha = torch.sigmoid(
            self.tok_raw_alpha(token_ids).squeeze(-1)
        )                                                      # [B, L]
        features = self.tok_features(token_ids)                # [B, L, d_f]

        # ── Position modulation ──
        pos = torch.arange(L, device=device)
        mu = mu + self.pos_mu(pos).unsqueeze(0)

        # ── Causal mask: position t sees tokens 0..t ──
        causal_mask = torch.tril(torch.ones(L, L, device=device))

        # ── Multi-pass rendering ──
        meaning = None
        for p in range(self.n_passes):
            # Heads processed sequentially to limit VRAM
            head_meanings = []
            for h in range(self.n_heads):
                q_h = self.query_proj[h](mu)                   # [B, L, d_s]
                K_h = self._pairwise_kernel(q_h, mu, log_var)  # [B, L, L]
                m_h = self._causal_render(
                    features, alpha, K_h, causal_mask
                )                                               # [B, L, d_f]
                head_meanings.append(m_h)

            # Combine heads
            meaning = self.head_proj(
                torch.cat(head_meanings, dim=-1)
            )                                                   # [B, L, d_f]
            meaning = self.dropout(meaning)

            # Update Gaussian params for next pass
            if p < self.n_passes - 1:
                ctx = torch.cat([features, meaning], dim=-1)    # [B, L, 2·d_f]
                mu = mu + self.mu_update[p](ctx)
                alpha = alpha * self.alpha_gate[p](ctx).squeeze(-1)
                features = features + self.pass_ffn[p](ctx)

        # ── Output ──
        logits = self.lm_head(self.ln_f(meaning))              # [B, L, V]
        return logits

    # ────────────────────────────────────────────────────────────
    # Generation
    # ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k sampling."""
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new):
            ctx = ids[:, -self.max_len :]
            logits = self.forward(ctx)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids

    # ────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        """Parameter count by component."""
        groups = {
            "tok_mu": self.tok_mu.weight.numel(),
            "tok_log_var": self.tok_log_var.weight.numel(),
            "tok_raw_alpha": self.tok_raw_alpha.weight.numel(),
            "tok_features": self.tok_features.weight.numel(),
            "pos_mu": self.pos_mu.weight.numel(),
            "log_tau": 1,
            "query_proj": sum(
                p.numel() for proj in self.query_proj for p in proj.parameters()
            ),
            "head_proj": sum(p.numel() for p in self.head_proj.parameters()),
            "mu_update": sum(
                p.numel() for m in self.mu_update for p in m.parameters()
            ),
            "alpha_gate": sum(
                p.numel() for m in self.alpha_gate for p in m.parameters()
            ),
            "pass_ffn": sum(
                p.numel() for m in self.pass_ffn for p in m.parameters()
            ),
            "ln_f": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": self.lm_head.weight.numel(),
        }
        groups["total"] = sum(groups.values())
        return groups
