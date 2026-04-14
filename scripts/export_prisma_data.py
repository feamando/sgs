#!/usr/bin/env python3
"""
Export SGS visualization data for Radiance Prisma.

Computes Gaussian parameters and rendering weights for example sentences,
exports as JSON for the static frontend. No trained model needed — uses
GloVe initialization (which already achieves 0.707 zero-shot Spearman).

Usage:
    python scripts/export_prisma_data.py --glove data/glove.6B.300d.txt
    python scripts/export_prisma_data.py --glove data/glove.6B.300d.txt --out prisma/data
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data import load_glove, tokenize
from src.gaussian import SemanticGaussianVocab
from src.kernel import gaussian_kernel_diag
from src.rendering import render


# Curated example sentences showing interesting SGS phenomena
EXAMPLES = [
    # Basic composition
    "The cat sat on the mat",
    "A dog runs through the park",
    "The warm coffee sat on the old wooden table",

    # Word order matters (transmittance effect)
    "Dog bites man",
    "Man bites dog",

    # Polysemy
    "I went to the bank to deposit money",
    "I sat on the bank of the river",
    "The spring flowers bloomed early",
    "The spring in the mattress broke",

    # Adjective composition
    "A big red ball",
    "A small blue car",
    "The dark stormy night",

    # Semantic similarity pairs
    "The cat is sleeping on the couch",
    "A kitten naps on the sofa",

    "The movie was really exciting",
    "The film was quite boring",

    # Complex sentences
    "Scientists discovered a new species of deep sea fish",
    "The quick brown fox jumps over the lazy dog",
    "She sold seashells by the seashore",

    # Abstract concepts
    "Freedom is the most important value",
    "Love conquers all obstacles",
    "Time flies when you are having fun",

    # Negation (where SGS struggles)
    "The weather is not cold today",
    "I am not happy about this",

    # Technical
    "Machine learning models process natural language",
    "Neural networks compute weighted sums of inputs",
    "Gaussian distributions represent uncertainty in meaning",

    # Short
    "Hello world",
    "Good morning",
    "Thank you very much",

    # Longer
    "The old professor walked slowly through the ancient library looking for a rare book",
    "Heavy rain fell across the city as people hurried home from work",
    "A small child pointed excitedly at the colorful butterfly in the garden",

    # Emotion
    "I am so excited about the concert tonight",
    "The news made everyone deeply sad",
    "She laughed until tears streamed down her face",

    # Berlin-themed :)
    "The Brandenburg Gate stands in the center of Berlin",
    "Currywurst is the most famous street food in Berlin",
    "The Berlin Wall fell in nineteen eighty nine",
]


def compute_sentence_data(
    sentence: str,
    word2idx: dict,
    vocab: SemanticGaussianVocab,
    pca_3d: PCA,
    tau: float,
    device: torch.device,
) -> dict | None:
    """Compute full SGS rendering data for one sentence."""
    tokens = tokenize(sentence, word2idx, max_len=30)
    if len(tokens) < 2:
        return None

    token_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    mu, log_var, alpha, features = vocab.get_params(token_ids)

    # Query = centroid
    query = mu.mean(dim=1)  # [1, d_s]

    # Kernel evaluation
    K = gaussian_kernel_diag(query, mu, log_var, torch.tensor(tau, device=device))

    # Rendering
    meaning, weights = render(features, alpha, K.squeeze(0).unsqueeze(0), return_weights=True)

    # Compute transmittance explicitly
    eff_opacity = (alpha * K).squeeze(0)  # [n]
    transmittance = torch.ones_like(eff_opacity)
    for i in range(1, len(transmittance)):
        transmittance[i] = transmittance[i-1] * (1 - eff_opacity[i-1].clamp(max=1-1e-6))

    # Project to 3D
    mu_np = mu.squeeze(0).detach().cpu().numpy()
    mu_3d = pca_3d.transform(mu_np)

    # Get word strings
    idx2word = {v: k for k, v in word2idx.items()}
    words = [idx2word.get(t, "?") for t in tokens]

    # Covariance in 3D (diagonal → scale per axis)
    var_np = torch.exp(log_var.squeeze(0)).detach().cpu().numpy()
    # Project variance through PCA components
    var_3d = np.abs(pca_3d.components_ @ var_np.T).T  # [n, 3]

    # Feature color: first 3 PCA of features → RGB
    feat_np = features.squeeze(0).detach().cpu().numpy()
    n_comp = min(3, feat_np.shape[0], feat_np.shape[1])
    if n_comp >= 3:
        feat_pca = PCA(n_components=3).fit_transform(feat_np)
    else:
        # Too few tokens for PCA — use fixed warm colors
        feat_pca = np.zeros((feat_np.shape[0], 3))
        for j in range(feat_np.shape[0]):
            feat_pca[j] = [0.9 - j*0.1, 0.5 + j*0.05, 0.3]
    # Normalize to [0, 1]
    feat_min = feat_pca.min(axis=0)
    feat_max = feat_pca.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1
    colors = (feat_pca - feat_min) / feat_range

    # Compute softmax attention weights for comparison
    # softmax over dot-product scores (same query, mu as keys)
    scores = torch.bmm(mu, query.unsqueeze(-1)).squeeze(-1)  # [1, n]
    softmax_weights = torch.softmax(scores / (mu.shape[-1] ** 0.5), dim=-1)

    return {
        "sentence": sentence,
        "words": words,
        "token_ids": tokens,
        "gaussians": {
            "mu_3d": mu_3d.tolist(),
            "scale_3d": (np.sqrt(var_3d) * 0.3).tolist(),
            "alpha": alpha.squeeze(0).detach().cpu().tolist(),
            "colors": colors.tolist(),
        },
        "rendering": {
            "kernel": K.squeeze(0).detach().cpu().tolist(),
            "weights": weights.squeeze(0).detach().cpu().tolist(),
            "transmittance": transmittance.detach().cpu().tolist(),
            "eff_opacity": eff_opacity.detach().cpu().tolist(),
            "meaning_3d": pca_3d.transform(
                meaning.detach().cpu().numpy().reshape(1, -1)[:, :mu_np.shape[1]]
            ).tolist()[0] if mu_np.shape[1] >= 3 else [0, 0, 0],
        },
        "softmax": {
            "weights": softmax_weights.squeeze(0).detach().cpu().tolist(),
        },
        "stats": {
            "total_weight": float(weights.sum().item()),
            "residual_transmittance": float(transmittance[-1].item() * (1 - eff_opacity[-1].item())),
            "top_word": words[int(weights.squeeze(0).argmax().item())],
            "softmax_top_word": words[int(softmax_weights.squeeze(0).argmax().item())],
            "n_tokens": len(tokens),
        },
    }


def main(args):
    device = torch.device("cpu")  # CPU is fine for export
    d_s = 64

    # Load GloVe
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)

    # Build vocab (zero-shot — no training)
    vocab = SemanticGaussianVocab(vocab_size=len(words), d_s=d_s, d_f=300)
    vocab.init_from_glove(vectors, freqs)
    vocab = vocab.to(device)

    tau = 64.0

    # PCA for 3D projection of splatting space
    mu_all = vocab.mu[1:].detach().cpu().numpy()  # skip padding
    pca_3d = PCA(n_components=3)
    pca_3d.fit(mu_all)
    print(f"3D PCA explained variance: {pca_3d.explained_variance_ratio_.sum():.3f}")

    # Export vocab summary (top words with positions)
    print("Exporting vocabulary...")
    vocab_data = {
        "d_s": d_s,
        "tau": tau,
        "n_words": len(words),
        "pca_3d_variance": pca_3d.explained_variance_ratio_.tolist(),
    }

    # Export top 500 words with their 3D positions (for background cloud)
    top_n = 500
    mu_3d_all = pca_3d.transform(mu_all[:top_n])
    alpha_all = torch.sigmoid(vocab.raw_alpha[1:top_n+1]).detach().cpu().numpy()
    vocab_cloud = []
    for i in range(top_n):
        vocab_cloud.append({
            "word": words[i+1],
            "pos": mu_3d_all[i].tolist(),
            "alpha": float(alpha_all[i]),
        })
    vocab_data["cloud"] = vocab_cloud

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab_data, f)
    print(f"  Saved vocab.json ({top_n} words)")

    # Export example sentences
    print(f"\nExporting {len(EXAMPLES)} example sentences...")
    examples = []
    for i, sentence in enumerate(EXAMPLES):
        data = compute_sentence_data(sentence, word2idx, vocab, pca_3d, tau, device)
        if data:
            examples.append(data)
            print(f"  [{i+1:2d}/{len(EXAMPLES)}] {sentence[:50]}... → {data['stats']['n_tokens']} tokens, "
                  f"top: '{data['stats']['top_word']}', residual: {data['stats']['residual_transmittance']:.3f}")
        else:
            print(f"  [{i+1:2d}/{len(EXAMPLES)}] SKIPPED (too few tokens): {sentence}")

    with open(os.path.join(args.out, "examples.json"), "w") as f:
        json.dump(examples, f)
    print(f"\n  Saved examples.json ({len(examples)} sentences)")

    # Also export individual files for lazy loading
    examples_dir = os.path.join(args.out, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    for i, ex in enumerate(examples):
        with open(os.path.join(examples_dir, f"{i:03d}.json"), "w") as f:
            json.dump(ex, f)

    print(f"\nDone! Data exported to {args.out}/")
    print(f"  vocab.json: {os.path.getsize(os.path.join(args.out, 'vocab.json')) / 1024:.0f} KB")
    print(f"  examples.json: {os.path.getsize(os.path.join(args.out, 'examples.json')) / 1024:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Prisma visualization data")
    parser.add_argument("--glove", type=str, default="data/glove.6B.300d.txt")
    parser.add_argument("--out", type=str, default="prisma/data")
    args = parser.parse_args()
    main(args)
