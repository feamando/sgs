"""
OOV (out-of-vocabulary) policy for Raum demo.

When a user types a word not in the blob library, the bridge's blob_id
head guesses. This module provides a cosine-similarity gate + nearest-
neighbor fallback to either confirm the prediction, substitute a better
match, or reject it as unresolved.
"""

import numpy as np
import torch


class OOVPolicy:
    """
    Resolves out-of-vocabulary words to blob classes via GloVe cosine NN.

    Usage:
        policy = OOVPolicy(blob_names, glove_vectors, word2idx)
        resolved_id = policy.resolve("couch", predicted_id=3)
        # Returns 3 if "couch" is close to blob_names[3], else finds NN, else None
    """

    def __init__(
        self,
        blob_names: list[str],
        glove_vectors: np.ndarray,
        word2idx: dict[str, int],
        threshold: float = 0.3,
    ):
        self.blob_names = blob_names
        self.threshold = threshold
        self.word2idx = word2idx

        # Build blob-class embedding matrix [n_blobs, d]
        d = glove_vectors.shape[1]
        self.blob_embeddings = np.zeros((len(blob_names), d), dtype=np.float32)
        for i, name in enumerate(blob_names):
            # Handle multi-word blob names: average the word vectors
            words = name.split()
            vecs = []
            for w in words:
                idx = word2idx.get(w)
                if idx is not None and idx < glove_vectors.shape[0]:
                    vecs.append(glove_vectors[idx])
            if vecs:
                self.blob_embeddings[i] = np.mean(vecs, axis=0)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.blob_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        self.blob_embeddings_normed = self.blob_embeddings / norms

        self.glove_vectors = glove_vectors

    def _get_word_vec(self, word: str) -> np.ndarray | None:
        words = word.lower().split()
        vecs = []
        for w in words:
            idx = self.word2idx.get(w)
            if idx is not None and idx < self.glove_vectors.shape[0]:
                vecs.append(self.glove_vectors[idx])
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype(np.float32)

    def resolve(self, word: str, predicted_id: int) -> int | None:
        """
        Check if predicted_id is a reasonable match for `word`.

        Returns:
            - predicted_id if cosine(word, blob[predicted_id]) > threshold
            - nearest-neighbor blob_id if cosine(word, NN) > threshold
            - None if no blob is close enough (unresolved)
        """
        vec = self._get_word_vec(word)
        if vec is None:
            return None

        vec_normed = vec / max(np.linalg.norm(vec), 1e-8)

        # Check predicted class first
        cos_pred = float(np.dot(vec_normed, self.blob_embeddings_normed[predicted_id]))
        if cos_pred >= self.threshold:
            return predicted_id

        # Find nearest neighbor
        cosines = self.blob_embeddings_normed @ vec_normed
        nn_id = int(np.argmax(cosines))
        cos_nn = float(cosines[nn_id])

        if cos_nn >= self.threshold:
            return nn_id

        return None

    def find_nearest(self, word: str) -> tuple[int | None, float]:
        """Find the nearest blob class for a word. Returns (blob_id, cosine)."""
        vec = self._get_word_vec(word)
        if vec is None:
            return None, 0.0
        vec_normed = vec / max(np.linalg.norm(vec), 1e-8)
        cosines = self.blob_embeddings_normed @ vec_normed
        nn_id = int(np.argmax(cosines))
        return nn_id, float(cosines[nn_id])
