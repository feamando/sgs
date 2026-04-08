"""
Semantic Gaussian primitive (Atom A1).

Each word in the vocabulary is stored as a Gaussian distribution:
  G = (μ, Σ, α, f)
  - μ ∈ R^d_s  — mean in splatting space
  - Σ          — covariance (diagonal in Phase 1)
  - α ∈ (0,1) — base opacity/salience
  - f ∈ R^d_f  — feature vector
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


class SemanticGaussianVocab(nn.Module):
    """Vocabulary of Semantic Gaussians."""

    def __init__(self, vocab_size: int, d_s: int = 64, d_f: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_s = d_s
        self.d_f = d_f

        # Mean in splatting space: initialized from GloVe PCA
        self.mu = nn.Parameter(torch.randn(vocab_size, d_s) * 0.1)

        # Log-variance (diagonal covariance): Σ_ii = exp(log_var_i)
        # Initialized to 0 → Σ = I (unit variance in each dimension)
        self.log_var = nn.Parameter(torch.zeros(vocab_size, d_s))

        # Raw opacity (pre-sigmoid): initialized to 0 → α = 0.5
        self.raw_alpha = nn.Parameter(torch.zeros(vocab_size))

        # Feature vectors: initialized from GloVe
        self.features = nn.Parameter(torch.randn(vocab_size, d_f) * 0.1)

        # Padding index
        self.padding_idx = 0

    def init_from_glove(
        self,
        glove_vectors: np.ndarray,
        word_freqs: np.ndarray | None = None,
    ):
        """
        Initialize from pre-trained GloVe embeddings.

        Args:
            glove_vectors: [vocab_size, 300] — GloVe embeddings
            word_freqs: [vocab_size] — word frequencies (for covariance init)
        """
        n_words = min(glove_vectors.shape[0], self.vocab_size)

        # Features: directly from GloVe
        with torch.no_grad():
            self.features[:n_words] = torch.from_numpy(
                glove_vectors[:n_words]
            ).float()

        # Means: PCA of GloVe to d_s dimensions
        pca = PCA(n_components=self.d_s)
        mu_np = pca.fit_transform(glove_vectors[:n_words])
        with torch.no_grad():
            self.mu[:n_words] = torch.from_numpy(mu_np).float()

        # Covariance: rare words get larger variance (more uncertain)
        if word_freqs is not None:
            # log_var ~ -log(freq) → rare words have high variance
            freq_tensor = torch.from_numpy(word_freqs[:n_words]).float()
            freq_tensor = freq_tensor.clamp(min=1e-8)
            log_freq = torch.log(freq_tensor)
            # Normalize to reasonable range: mean=0, std=1
            log_freq = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)
            # Invert: rare (low freq → low log_freq) → positive log_var → high variance
            init_log_var = -log_freq.unsqueeze(1).expand(n_words, self.d_s)
            with torch.no_grad():
                self.log_var[:n_words] = init_log_var

        # Zero out padding
        with torch.no_grad():
            self.mu[self.padding_idx] = 0
            self.features[self.padding_idx] = 0
            self.raw_alpha[self.padding_idx] = -10  # sigmoid(-10) ≈ 0

        print(f"Initialized {n_words} Gaussians from GloVe")
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  μ range: [{mu_np.min():.2f}, {mu_np.max():.2f}]")

    def get_params(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Look up Gaussian parameters for given token indices.

        Args:
            indices: [batch, seq_len] — token indices

        Returns:
            mu:      [batch, seq, d_s]
            log_var: [batch, seq, d_s]
            alpha:   [batch, seq]
            features:[batch, seq, d_f]
        """
        mu = self.mu[indices]
        log_var = self.log_var[indices]
        alpha = torch.sigmoid(self.raw_alpha[indices])
        features = self.features[indices]
        return mu, log_var, alpha, features
