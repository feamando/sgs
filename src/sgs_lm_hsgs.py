"""
Hierarchical SGS Language Model — Radiance Planck 1.1 / Hertz 1.1.

Extends the base SGS-LM with a BlobStore for two-pass rendering:
  Pass 1: Render knowledge blobs (consume transmittance)
  Pass 2: Render word tokens in remaining capacity

Mathematical foundation (all Lean 4 verified):
  H1: Two-pass = single-pass (partition equivalence)
  H2: T_max cap preserves expressiveness
  H4: Total weight is permutation-invariant
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgs_lm import SGSLanguageModel, migrate_state_dict
from src.blob_store import BlobStore


class HSGSLanguageModel(nn.Module):
    """
    Hierarchical SGS Language Model.

    Wraps a base SGSLanguageModel and adds a BlobStore.
    At each position, blob rendering provides a semantic backdrop;
    word rendering fills in specifics within the remaining transmittance.
    """

    def __init__(
        self,
        base_model: SGSLanguageModel,
        blob_store: BlobStore,
        blob_proj: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.blobs = blob_store

        d_f = base_model.d_f

        # Project blob meaning into the same space as word meaning
        if blob_proj and blob_store.d_f != d_f:
            self.blob_proj = nn.Linear(blob_store.d_f, d_f)
        else:
            self.blob_proj = nn.Identity()

        # Learned gate: how much to trust blob vs word rendering
        # Initialized to pass through T_residual directly
        self.blob_gate = nn.Sequential(
            nn.Linear(d_f * 2, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.blob_gate[0].weight)
        nn.init.constant_(self.blob_gate[0].bias, 0.0)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        blob_store: BlobStore,
        device: torch.device = torch.device("cpu"),
        **model_kwargs,
    ) -> "HSGSLanguageModel":
        """Load base model from checkpoint and wrap with blob store."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = migrate_state_dict(state)

        vocab_size = state["tok_mu.weight"].shape[0]
        base = SGSLanguageModel(vocab_size=vocab_size, **model_kwargs)
        base.load_state_dict(state)

        model = cls(base, blob_store)
        return model.to(device)

    def _compute_query(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute blob query from input tokens (mean of token mu's).

        Args:
            token_ids: [B, L]
        Returns:
            query: [B, d_s]
        """
        mu = self.base.tok_mu(token_ids)                         # [B, L, d_s]
        pos = torch.arange(token_ids.shape[1], device=token_ids.device)
        mu = mu + self.base.pos_mu(pos).unsqueeze(0)
        return mu.mean(dim=1)                                    # [B, d_s]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Two-pass forward: blob rendering + word rendering.

        Args:
            token_ids: [B, L] token indices
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = token_ids.shape
        device = token_ids.device

        # ── Pass 1: Blob rendering ──
        query = self._compute_query(token_ids)                   # [B, d_s]
        blob_meaning, t_residual = self.blobs.render(query)      # [B, d_f], [B]
        blob_meaning = self.blob_proj(blob_meaning)              # [B, d_f]

        # ── Pass 2: Word rendering (standard SGS forward) ──
        # Run the full base model forward pass
        word_meaning = self._base_forward_meaning(token_ids)     # [B, L, d_f]

        # ── Combine: blob backdrop + word specifics ──
        # Expand blob meaning to all positions
        blob_expanded = blob_meaning.unsqueeze(1).expand_as(word_meaning)

        # Gate: learn how to combine blob and word meaning
        combined_ctx = torch.cat([blob_expanded, word_meaning], dim=-1)
        gate = self.blob_gate(combined_ctx)                      # [B, L, 1]

        # T_residual scales word contribution (from H1 proof)
        t_res = t_residual.view(B, 1, 1)                        # [B, 1, 1]

        # Final meaning: blob contribution + residual-scaled word contribution
        meaning = (1 - t_res) * gate * blob_expanded + t_res * word_meaning

        # ── Output ──
        logits = self.base.lm_head(self.base.ln_f(meaning))     # [B, L, V]
        return logits

    def _base_forward_meaning(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the base SGS model but return meaning instead of logits."""
        B, L = token_ids.shape
        device = token_ids.device

        mu = self.base.tok_mu(token_ids)
        log_var = self.base.tok_log_var(token_ids)
        alpha = torch.sigmoid(self.base.tok_raw_alpha(token_ids).squeeze(-1))
        features = self.base.tok_features(token_ids)

        pos = torch.arange(L, device=device)
        mu = mu + self.base.pos_mu(pos).unsqueeze(0)

        causal_mask = torch.tril(torch.ones(L, L, device=device))

        # Fused multi-head query projection (matches current base model layout).
        H = self.base.n_heads
        d_s = self.base.d_s

        meaning = None
        for p in range(self.base.n_passes):
            # All-heads-in-one matmul: [B, L, d_s] -> [B, L, H*d_s] -> [B*H, L, d_s]
            all_q = self.base.query_proj(mu)
            all_q = all_q.view(B, L, H, d_s).permute(0, 2, 1, 3).reshape(B * H, L, d_s)

            mu_h = mu.unsqueeze(1).expand(B, H, L, d_s).reshape(B * H, L, d_s)
            lv_h = log_var.unsqueeze(1).expand(B, H, L, d_s).reshape(B * H, L, d_s)
            al_h = alpha.unsqueeze(1).expand(B, H, L).reshape(B * H, L)
            ft_h = features.unsqueeze(1).expand(B, H, L, self.base.d_f).reshape(
                B * H, L, self.base.d_f
            )

            K_all = self.base._pairwise_kernel(all_q, mu_h, lv_h)
            m_all = self.base._causal_render(ft_h, al_h, K_all, causal_mask)

            m_all = m_all.view(B, H, L, self.base.d_f).permute(0, 2, 1, 3)
            m_cat = m_all.reshape(B, L, H * self.base.d_f)

            meaning = self.base.head_proj(m_cat)
            meaning = self.base.dropout(meaning)

            if p < self.base.n_passes - 1:
                ctx = torch.cat([features, meaning], dim=-1)
                mu = mu + self.base.mu_update[p](ctx)
                alpha = alpha * self.base.alpha_gate[p](ctx).squeeze(-1)
                features = features + self.base.pass_ffn[p](ctx)

        return meaning

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with blob-conditioned rendering."""
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new):
            ctx = ids[:, -self.base.max_len:]
            logits = self.forward(ctx)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        base = self.base.param_breakdown()
        blobs = self.blobs.param_breakdown()
        gate_params = sum(p.numel() for p in self.blob_gate.parameters())
        proj_params = sum(p.numel() for p in self.blob_proj.parameters()) if not isinstance(self.blob_proj, nn.Identity) else 0
        return {
            **{f"base_{k}": v for k, v in base.items()},
            **{f"blob_{k}": v for k, v in blobs.items()},
            "blob_gate": gate_params,
            "blob_proj": proj_params,
        }
