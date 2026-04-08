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
        idf_init: bool = False,
        remove_pc1: bool = False,
    ):
        """
        Initialize from pre-trained GloVe embeddings.

        Args:
            glove_vectors: [vocab_size, 300] — GloVe embeddings
            word_freqs: [vocab_size] — word frequencies (for covariance init)
            idf_init: if True, initialize opacity from IDF weights (Phase 1.5)
            remove_pc1: if True, subtract first principal component from features (SIF trick)
        """
        n_words = min(glove_vectors.shape[0], self.vocab_size)
        glove_f32 = glove_vectors[:n_words].astype(np.float32)

        # Features: from GloVe (optionally with PC1 removal)
        if remove_pc1:
            mean_vec = glove_f32.mean(axis=0, keepdims=True)
            centered = glove_f32 - mean_vec
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            pc1 = Vt[0]  # first principal component
            # Remove projection onto pc1
            proj = (glove_f32 @ pc1[:, None]) @ pc1[None, :]
            glove_features = glove_f32 - proj
            print(f"  Removed PC1 (explained {S[0]**2/np.sum(S**2):.1%} of variance)")
        else:
            glove_features = glove_f32

        with torch.no_grad():
            if self.d_f == glove_features.shape[1]:
                self.features[:n_words] = torch.from_numpy(glove_features)
            else:
                # If d_f != glove dim, pad or truncate
                d = min(self.d_f, glove_features.shape[1])
                self.features[:n_words, :d] = torch.from_numpy(glove_features[:, :d])

        # Means: PCA of GloVe to d_s dimensions (or direct if d_s == glove_dim)
        if self.d_s == glove_f32.shape[1]:
            mu_np = glove_f32
            explained = 1.0
            print(f"  d_s == d_glove ({self.d_s}), no PCA needed")
        elif self.d_s < glove_f32.shape[1]:
            pca = PCA(n_components=self.d_s)
            mu_np = pca.fit_transform(glove_f32)
            explained = pca.explained_variance_ratio_.sum()
        else:
            # d_s > glove_dim: pad with zeros
            pca = None
            mu_np = np.zeros((n_words, self.d_s), dtype=np.float32)
            mu_np[:, :glove_f32.shape[1]] = glove_f32
            explained = 1.0

        with torch.no_grad():
            self.mu[:n_words] = torch.from_numpy(mu_np.astype(np.float32))

        # Covariance: rare words get larger variance (more uncertain)
        if word_freqs is not None:
            freq_tensor = torch.from_numpy(word_freqs[:n_words]).float()
            freq_tensor = freq_tensor.clamp(min=1e-8)
            log_freq = torch.log(freq_tensor)
            log_freq = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)
            init_log_var = -log_freq.unsqueeze(1).expand(n_words, self.d_s)
            with torch.no_grad():
                self.log_var[:n_words] = init_log_var

        # Opacity: IDF initialization (Phase 1.5) or uniform
        if idf_init and word_freqs is not None:
            freq_tensor = torch.from_numpy(word_freqs[:n_words]).float()
            freq_tensor = freq_tensor.clamp(min=1e-8)
            # IDF = log(1/freq), normalized to [0.1, 0.9] via sigmoid
            idf = -torch.log(freq_tensor)
            # Scale so common words → low alpha, rare words → high alpha
            idf_norm = (idf - idf.mean()) / (idf.std() + 1e-8)
            # Map to raw_alpha (pre-sigmoid): ~0 for average, positive for rare
            with torch.no_grad():
                self.raw_alpha[:n_words] = idf_norm
            mean_alpha = torch.sigmoid(idf_norm).mean().item()
            print(f"  IDF-initialized opacity: mean α = {mean_alpha:.3f}")
        else:
            pass  # default: raw_alpha = 0 → α = 0.5

        # Zero out padding
        with torch.no_grad():
            self.mu[self.padding_idx] = 0
            self.features[self.padding_idx] = 0
            self.raw_alpha[self.padding_idx] = -10

        print(f"Initialized {n_words} Gaussians from GloVe")
        print(f"  PCA explained variance: {explained:.3f}")
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
