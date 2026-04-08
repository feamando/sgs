#!/usr/bin/env python3
"""
Phase 0: Numerical Feasibility Check

Tests whether the Gaussian kernel produces usable values at d_s=64.
Run this FIRST before building the full training pipeline.

Usage:
    python scripts/phase0_feasibility.py --glove data/glove.6B.300d.txt
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data import load_glove
from src.kernel import gaussian_kernel_diag


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load GloVe
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=10000)

    # 2. PCA to d_s dimensions
    print(f"\nPCA: 300d → {args.d_s}d")
    pca = PCA(n_components=args.d_s)
    mu_np = pca.fit_transform(vectors[1:])  # Skip padding
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    mu = torch.from_numpy(mu_np).float().to(device)  # [9999, d_s]

    # 3. Initialize covariances from frequency
    freq_tensor = torch.from_numpy(freqs[1:]).float().to(device)
    log_freq = torch.log(freq_tensor.clamp(min=1e-10))
    log_freq_norm = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)
    log_var = (-log_freq_norm).unsqueeze(1).expand(-1, args.d_s)  # [9999, d_s]

    # 4. Temperature sweep
    print("\n" + "=" * 70)
    print("TEMPERATURE SWEEP")
    print("=" * 70)

    n_queries = 1000
    query_indices = torch.randint(0, mu.shape[0], (n_queries,))
    queries = mu[query_indices]  # [1000, d_s] — use existing word positions as queries

    # Reshape for batch evaluation
    queries_batch = queries.unsqueeze(0)  # [1, 1000, d_s]
    mu_batch = mu.unsqueeze(0).expand(1, -1, -1)  # [1, 9999, d_s]
    log_var_batch = log_var.unsqueeze(0).expand(1, -1, -1)  # [1, 9999, d_s]

    for tau_val in [8, 16, 32, 64, 128, 256, 512]:
        tau = torch.tensor(float(tau_val), device=device)

        # Evaluate kernel for each query against all Gaussians
        K_values = []
        for i in range(0, n_queries, 100):  # Process in chunks
            batch_q = queries[i:i+100]  # [chunk, d_s]
            chunk_K = []
            for q in batch_q:
                q_exp = q.unsqueeze(0).unsqueeze(0)  # [1, 1, d_s]
                K = gaussian_kernel_diag(
                    q_exp.squeeze(1),  # [1, d_s]
                    mu.unsqueeze(0),   # [1, 9999, d_s]
                    log_var.unsqueeze(0),  # [1, 9999, d_s]
                    tau,
                )  # [1, 9999]
                chunk_K.append(K.squeeze(0))
            K_values.append(torch.stack(chunk_K))

        K_all = torch.cat(K_values, dim=0)  # [1000, 9999]

        # Statistics
        K_flat = K_all.flatten()
        sparse_frac = (K_flat < 1e-3).float().mean().item()
        nonzero_K = K_flat[K_flat >= 1e-3]

        print(f"\n  τ = {tau_val:>4d}:")
        print(f"    Kernel range: [{K_flat.min().item():.2e}, {K_flat.max().item():.2e}]")
        print(f"    Mean (all):    {K_flat.mean().item():.4e}")
        print(f"    Sparsity:      {sparse_frac:.1%} below 1e-3")
        if nonzero_K.numel() > 0:
            print(f"    Mean (>1e-3):  {nonzero_K.mean().item():.4f}")
            print(f"    Std  (>1e-3):  {nonzero_K.std().item():.4f}")
            # Top-K contributing Gaussians per query
            top_k = (K_all > 1e-3).sum(dim=1).float()
            print(f"    Avg contributors/query: {top_k.mean().item():.1f} "
                  f"(min={top_k.min().item():.0f}, max={top_k.max().item():.0f})")

        # Check gradients
        q_test = queries[0:1].requires_grad_(True)  # [1, d_s]
        K_test = gaussian_kernel_diag(
            q_test, mu[:100].unsqueeze(0), log_var[:100].unsqueeze(0), tau,
        )
        K_test.sum().backward()
        grad_norm = q_test.grad.norm().item()
        print(f"    Gradient norm (∂K/∂q): {grad_norm:.4e}")

    # 5. Semantic sanity check
    print("\n" + "=" * 70)
    print("SEMANTIC SANITY CHECK")
    print("=" * 70)
    print("Finding nearest Gaussians by kernel value (τ=64)...")

    tau = torch.tensor(64.0, device=device)
    test_words = ["king", "computer", "happy", "bank", "science"]

    for word in test_words:
        if word not in word2idx:
            continue
        idx = word2idx[word] - 1  # -1 because we skipped padding
        q = mu[idx].unsqueeze(0)  # [1, d_s]
        K = gaussian_kernel_diag(
            q, mu.unsqueeze(0), log_var.unsqueeze(0), tau,
        ).squeeze(0)  # [9999]

        # Top 10 nearest
        topk_vals, topk_idx = K.topk(11)  # 11 to exclude self
        print(f"\n  '{word}' — nearest by kernel:")
        for v, j in zip(topk_vals[1:6], topk_idx[1:6]):  # Skip self
            print(f"    {words[j.item()+1]:>15s}  K={v.item():.4f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    # Use tau=64 for verdict
    tau = torch.tensor(64.0, device=device)
    K_sample = []
    for q in queries[:100]:
        K = gaussian_kernel_diag(
            q.unsqueeze(0), mu.unsqueeze(0), log_var.unsqueeze(0), tau,
        ).squeeze(0)
        K_sample.append(K)
    K_sample = torch.stack(K_sample)
    sparsity = (K_sample < 1e-3).float().mean().item()
    avg_contrib = (K_sample > 1e-3).sum(dim=1).float().mean().item()

    print(f"  At τ={64}, d_s={args.d_s}:")
    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Avg contributors/query: {avg_contrib:.1f}")

    # Check discriminativeness: do top-K values differ meaningfully from bottom-K?
    top5_mean = K_sample.topk(5, dim=1).values.mean().item()
    bot5_mean = K_sample.topk(5, dim=1, largest=False).values.mean().item()
    discrimination = top5_mean - bot5_mean

    print(f"  Discrimination (top5 - bot5): {discrimination:.4f}")

    if sparsity > 0.80:
        print("  ✓ PASS — kernel is sparse and discriminative")
        print("  → Proceed to Phase 1 (with efficiency advantage)")
    elif discrimination > 0.05:
        print("  ✓ PASS — kernel is discriminative (not sparse, but ranks correctly)")
        print("  → Proceed to Phase 1. Sparsity is an efficiency concern for later.")
        print("  → For short sentences (n<50), O(n²) is fine on a 4090.")
        print("  → Sparsity optimization can come in Phase 4 (lower d_s, learned spread).")
    elif discrimination > 0.01:
        print("  ~ MARGINAL — kernel is weakly discriminative")
        print("  → Proceed with caution; consider lower τ for sharper discrimination")
    else:
        print("  ✗ FAIL — kernel cannot distinguish nearby from far Gaussians")
        print("  → Try d_s=32 or inverse-quadratic kernel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", type=str, required=True,
                        help="Path to glove.6B.300d.txt")
    parser.add_argument("--d_s", type=int, default=64,
                        help="Splatting space dimensionality")
    args = parser.parse_args()
    main(args)
