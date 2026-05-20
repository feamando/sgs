"""
Conversation memory via dynamic blob store (Planck 1.4).

Implements per-turn blob writing and hybrid retrieval for long
conversations. Each (user, assistant) turn pair is encoded as a blob
and stored in a pre-allocated buffer. At prompt time, recent turns are
included verbatim and older turns are retrieved by similarity.

Architecture:
  - DynamicBlobStore: fixed-capacity ring buffer of blob embeddings
  - TurnEncoder: encodes a text turn into (mu_s, features) using Planck
  - HybridRetriever: combines last-N verbatim + top-k similarity with
    recency decay
"""

import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Turn:
    """One conversation turn (user + assistant)."""
    turn_idx: int
    user_msg: str
    assistant_msg: str
    timestamp: float = 0.0


@dataclass
class ConversationSession:
    """All turns in one conversation."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)


class DynamicBlobStore(nn.Module):
    """
    Pre-allocated blob store that grows as conversation turns arrive.

    Fixed capacity (max_blobs). When full, oldest blob is overwritten
    (ring buffer). Retrieval uses Gaussian kernel distance, same as
    the static BlobStore.
    """

    def __init__(
        self,
        max_blobs: int = 512,
        d_s: int = 128,
        d_f: int = 1000,
        k: int = 8,
        tau_init: float = 128.0,
    ):
        super().__init__()
        self.max_blobs = max_blobs
        self.d_s = d_s
        self.d_f = d_f
        self.k = k

        # Pre-allocated storage (not nn.Parameters, just buffers)
        self.register_buffer("mu", torch.zeros(max_blobs, d_s))
        self.register_buffer("features", torch.zeros(max_blobs, d_f))
        self.register_buffer("timestamps", torch.zeros(max_blobs))
        self.register_buffer("valid", torch.zeros(max_blobs, dtype=torch.bool))

        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self._write_ptr = 0
        self._total_written = 0

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum().item())

    def write(self, mu_s: torch.Tensor, features: torch.Tensor, timestamp: float = 0.0):
        """
        Write one blob to the store.

        Args:
            mu_s: [d_s] centroid in splatting space
            features: [d_f] feature vector
            timestamp: turn timestamp (for recency decay)
        """
        idx = self._write_ptr % self.max_blobs
        self.mu[idx] = mu_s.detach()
        self.features[idx] = features.detach()
        self.timestamps[idx] = timestamp
        self.valid[idx] = True
        self._write_ptr += 1
        self._total_written += 1

    def retrieve(
        self,
        query: torch.Tensor,
        current_time: float = 0.0,
        decay: float = 0.1,
        query_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k blobs by similarity with recency decay.

        Score = cos_sim(query_features, blob_features) * exp(-decay * age)

        Uses feature-space similarity (d_f=1000) for semantic matching.
        Falls back to mu-space (d_s=128) if query_features not provided.

        Args:
            query: [d_s] query in splatting space (fallback)
            current_time: current turn index (for age computation)
            decay: exponential decay rate on age
            query_features: [d_f] query in feature space (preferred)

        Returns:
            indices: [k] blob indices
            scores: [k] retrieval scores
            features: [k, d_f] retrieved blob features
        """
        if self.n_valid == 0:
            empty = torch.zeros(0, dtype=torch.long, device=query.device)
            return empty, torch.zeros(0, device=query.device), torch.zeros(0, self.d_f, device=query.device)

        valid_mask = self.valid
        valid_ts = self.timestamps[valid_mask]
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        # Use feature space for similarity (much better for semantic matching)
        if query_features is not None:
            valid_feat = self.features[valid_mask]
            q_norm = query_features / query_features.norm().clamp(min=1e-8)
            f_norms = valid_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            f_normed = valid_feat / f_norms
            cos_sim = (q_norm.unsqueeze(0) @ f_normed.T).squeeze(0)
        else:
            valid_mu = self.mu[valid_mask]
            query_norm = query / query.norm().clamp(min=1e-8)
            mu_norms = valid_mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mu_normed = valid_mu / mu_norms
            cos_sim = (query_norm.unsqueeze(0) @ mu_normed.T).squeeze(0)

        # Recency weighting
        age = current_time - valid_ts
        recency = torch.exp(-decay * age.clamp(min=0))

        scores = cos_sim * recency

        # Top-k
        actual_k = min(self.k, scores.shape[0])
        top_scores, top_local_idx = scores.topk(actual_k)
        top_global_idx = valid_idx[top_local_idx]
        top_features = self.features[top_global_idx]

        return top_global_idx, top_scores, top_features

    def clear(self):
        """Reset the store."""
        self.mu.zero_()
        self.features.zero_()
        self.timestamps.zero_()
        self.valid.zero_()
        self._write_ptr = 0
        self._total_written = 0


class TurnEncoder:
    """
    Encodes conversation turns into blob embeddings using Planck's
    token embedding table.

    Takes the mean of token embeddings across the turn text to produce
    a single (mu_s, features) pair.
    """

    def __init__(self, model, tokenizer):
        """
        Args:
            model: SGSLanguageModel (or HSGSLanguageModel with .base)
            tokenizer: SentencePiece processor
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Get the base model if wrapped
        base = model.base if hasattr(model, "base") else model
        self.tok_mu = base.tok_mu
        self.tok_features = base.tok_features
        self.pos_mu = base.pos_mu
        self.d_s = base.d_s
        self.d_f = base.d_f

    @torch.no_grad()
    def encode_turn(self, user_msg: str, assistant_msg: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a (user, assistant) turn pair into a blob.

        Returns:
            mu_s: [d_s] mean position in splatting space
            features: [d_f] mean feature vector
        """
        text = f"{user_msg} {assistant_msg}"
        token_ids = self.tokenizer.encode(text, out_type=int)

        # Truncate to model's max_len
        max_len = 512
        token_ids = token_ids[:max_len]

        ids_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mu = self.tok_mu(ids_t)  # [1, L, d_s]
        features = self.tok_features(ids_t)  # [1, L, d_f]

        # Add positional modulation
        pos = torch.arange(len(token_ids), device=self.device)
        mu = mu + self.pos_mu(pos).unsqueeze(0)

        # Mean pool
        mu_mean = mu.mean(dim=1).squeeze(0)  # [d_s]
        feat_mean = features.mean(dim=1).squeeze(0)  # [d_f]

        return mu_mean, feat_mean

    @torch.no_grad()
    def encode_query(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a query string into splatting space + feature space.

        Returns:
            (mu_s [d_s], features [d_f])
        """
        token_ids = self.tokenizer.encode(text, out_type=int)[:512]
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mu = self.tok_mu(ids_t)
        features = self.tok_features(ids_t)
        pos = torch.arange(len(token_ids), device=self.device)
        mu = mu + self.pos_mu(pos).unsqueeze(0)
        return mu.mean(dim=1).squeeze(0), features.mean(dim=1).squeeze(0)


class HybridRetriever:
    """
    Combines last-N verbatim turns with top-k similarity retrieval.

    At prompt time, builds the effective context:
    1. System prompt (if any)
    2. Retrieved older turns (by similarity, recency-weighted)
    3. Last N turns verbatim
    4. Current user message
    """

    def __init__(
        self,
        blob_store: DynamicBlobStore,
        turn_encoder: TurnEncoder,
        n_recent: int = 3,
        k_retrieve: int = 5,
        decay: float = 0.05,
    ):
        self.blob_store = blob_store
        self.turn_encoder = turn_encoder
        self.n_recent = n_recent
        self.k_retrieve = k_retrieve
        self.decay = decay

    def build_context(
        self,
        session: ConversationSession,
        current_user_msg: str,
        system_prompt: str = "",
    ) -> str:
        """
        Build the effective prompt context for the LM.

        Returns a string ready to be tokenized and fed to the model.
        """
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        n_turns = len(session.turns)

        # Retrieve older turns (skip the last N which go verbatim)
        if n_turns > self.n_recent and self.blob_store.n_valid > 0:
            query_mu, query_feat = self.turn_encoder.encode_query(current_user_msg)
            _, scores, _ = self.blob_store.retrieve(
                query_mu,
                current_time=float(n_turns),
                decay=self.decay,
                query_features=query_feat,
            )
            # Get the actual turn texts for retrieved blobs
            # (blob index maps to turn index for per-session stores)
            retrieved_indices = []
            if scores.numel() > 0:
                valid_idx = torch.nonzero(self.blob_store.valid, as_tuple=False).squeeze(-1)
                top_k = min(self.k_retrieve, scores.shape[0])
                _, top_local = scores.topk(top_k)
                for li in top_local.tolist():
                    turn_idx = int(self.blob_store.timestamps[valid_idx[li]].item())
                    if turn_idx < n_turns - self.n_recent:
                        retrieved_indices.append(turn_idx)

            # Add retrieved turns (sorted by turn order)
            for ti in sorted(set(retrieved_indices)):
                if ti < len(session.turns):
                    t = session.turns[ti]
                    parts.append(f"[retrieved turn {ti}] User: {t.user_msg}")
                    parts.append(f"Assistant: {t.assistant_msg}")

        # Last N turns verbatim
        recent_start = max(0, n_turns - self.n_recent)
        for i in range(recent_start, n_turns):
            t = session.turns[i]
            parts.append(f"User: {t.user_msg}")
            parts.append(f"Assistant: {t.assistant_msg}")

        # Current turn
        parts.append(f"User: {current_user_msg}")
        parts.append("Assistant:")

        return "\n".join(parts)
