"""
Frozen Planck encoder for Raum 1.1.

Loads a Planck checkpoint (SGSLanguageModel state_dict) and exposes
per-token (mu_s, features) as frozen embeddings for the Raum bridge.
No autoregressive generation, no multi-pass rendering -- just the
token-level Gaussian parameters.
"""

import torch
import torch.nn as nn
from pathlib import Path


class FrozenPlanckEncoder(nn.Module):
    """
    Extract per-token Gaussian parameters from a trained Planck LM.

    Output shape per token:
        mu_s:     [B, N, d_s]   (semantic position)
        features: [B, N, d_f]   (feature vector)

    All parameters are frozen (no gradient flow).
    """

    def __init__(self, checkpoint_path: str | Path, device: torch.device | str = "cpu"):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt

        self.d_s = state["tok_mu.weight"].shape[1]
        self.d_f = state["tok_features.weight"].shape[1]
        self.vocab_size = state["tok_mu.weight"].shape[0]

        self.tok_mu = nn.Embedding(self.vocab_size, self.d_s)
        self.tok_features = nn.Embedding(self.vocab_size, self.d_f)

        self.tok_mu.weight.data.copy_(state["tok_mu.weight"])
        self.tok_features.weight.data.copy_(state["tok_features.weight"])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [B, N] int tensor of SentencePiece token ids
        Returns:
            (mu_s [B, N, d_s], features [B, N, d_f])
        """
        return self.tok_mu(token_ids), self.tok_features(token_ids)

    @torch.no_grad()
    def encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(token_ids)


def build_sp_word2idx(tokenizer_path: str | Path) -> dict[str, int]:
    """
    Build a word2idx from a SentencePiece model for Raum scene words.

    Each scene word (sphere, red, above, etc.) encodes to a single SP
    token. Returns {word: sp_token_id} for use in RaumDataset.
    """
    import sentencepiece as spm
    from .vocab import ALL_SCENE_WORDS

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    word2idx: dict[str, int] = {}

    for word in ALL_SCENE_WORDS:
        ids = sp.encode(word, out_type=int)
        if len(ids) == 1:
            word2idx[word] = ids[0]
        else:
            # Multi-token: use first subtoken (rare for common words)
            word2idx[word] = ids[0]

    # Fallback for unk
    word2idx["<unk>"] = sp.unk_id()
    return word2idx
