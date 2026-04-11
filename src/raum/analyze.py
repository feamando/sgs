"""
Analysis tools for PoC-C: inspect what the space transform learns.
"""

import torch
import numpy as np


@torch.no_grad()
def analyze_mu_proj(
    model,
    word2idx: dict[str, int],
    vocab_mu: torch.Tensor,
) -> dict:
    """
    Pass individual words through the space transform and analyze.

    Returns dict with analysis results.
    """
    spatial_words = ["left", "right", "above", "below", "behind"]
    color_words = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
    object_words = ["sphere", "cube", "cylinder", "cone", "plane", "torus"]
    size_words = ["tiny", "small", "medium", "large", "huge"]

    all_groups = {
        "spatial": spatial_words,
        "color": color_words,
        "object": object_words,
        "size": size_words,
    }

    results = {}
    for group_name, words in all_groups.items():
        coords = {}
        for w in words:
            if w not in word2idx:
                continue
            idx = word2idx[w]
            mu_s = vocab_mu[idx].unsqueeze(0).unsqueeze(0)  # [1, 1, d_s]
            mu_xyz = model.mu_proj(mu_s).squeeze()           # [3]
            coords[w] = mu_xyz.cpu().numpy()
        results[group_name] = coords

    return results


def print_word_mapping(results: dict):
    """Pretty-print word → xyz mapping."""
    for group, coords in results.items():
        print(f"\n{group.upper()} words:")
        for word, xyz in coords.items():
            print(f"  {word:>10s} → x={xyz[0]:+.3f}  y={xyz[1]:+.3f}  z={xyz[2]:+.3f}")

    # Check interpretability: do spatial words separate on axes?
    spatial = results.get("spatial", {})
    if "left" in spatial and "right" in spatial:
        lr_diff = spatial["right"][0] - spatial["left"][0]
        print(f"\n  left-right x-separation: {lr_diff:+.3f} (should be positive)")
    if "above" in spatial and "below" in spatial:
        ud_diff = spatial["above"][1] - spatial["below"][1]
        print(f"  above-below y-separation: {ud_diff:+.3f} (should be positive)")


@torch.no_grad()
def analyze_sentence_positions(
    model,
    word2idx: dict[str, int],
    vocab_mu: torch.Tensor,
    sentences: list[str],
) -> list[dict]:
    """
    Encode sentences and extract coarse word positions in 3D.

    Returns list of {sentence, word_positions: {word: [x,y,z]}}.
    """
    results = []
    for sent in sentences:
        words = sent.lower().split()
        positions = {}
        for w in words:
            if w not in word2idx:
                continue
            idx = word2idx[w]
            mu_s = vocab_mu[idx].unsqueeze(0).unsqueeze(0)
            mu_xyz = model.mu_proj(mu_s).squeeze().cpu().numpy()
            positions[w] = mu_xyz
        results.append({"sentence": sent, "word_positions": positions})
    return results


@torch.no_grad()
def interpolation_analysis(
    model,
    word2idx: dict[str, int],
    vocab_mu: torch.Tensor,
    vocab_features: torch.Tensor,
    word_a: str,
    word_b: str,
    n_steps: int = 10,
) -> list[tuple[float, np.ndarray]]:
    """
    Interpolate between two words in semantic space and project to 3D.

    Returns list of (t, xyz) along the interpolation path.
    """
    idx_a = word2idx.get(word_a)
    idx_b = word2idx.get(word_b)
    if idx_a is None or idx_b is None:
        return []

    mu_a = vocab_mu[idx_a]
    mu_b = vocab_mu[idx_b]

    path = []
    for i in range(n_steps + 1):
        t = i / n_steps
        mu_interp = (1 - t) * mu_a + t * mu_b
        mu_xyz = model.mu_proj(mu_interp.unsqueeze(0).unsqueeze(0)).squeeze()
        path.append((t, mu_xyz.cpu().numpy()))
    return path
