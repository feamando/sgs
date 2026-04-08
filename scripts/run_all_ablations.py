#!/usr/bin/env python3
"""
Phase 1c: Run all ablations IN-PROCESS and produce comparison table.

Usage:
    python scripts/run_all_ablations.py --glove data/glove.6B.300d.txt
    python scripts/run_all_ablations.py --glove data/glove.6B.300d.txt --epochs 10
"""

import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from src.data import load_glove, get_dataloaders
from src.gaussian import SemanticGaussianVocab
from src.rendering import render_mean_pool
from src.model import (
    SGSEncoder, SGSSimilarityModel,
    MeanPoolModel, MeanPoolMuModel,
    SoftmaxAttentionModel, NoTransmittanceModel,
)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids_a, mask_a, ids_b, mask_b, scores in loader:
            ids_a, mask_a = ids_a.to(device), mask_a.to(device)
            ids_b, mask_b = ids_b.to(device), mask_b.to(device)
            preds = model(ids_a, mask_a, ids_b, mask_b)
            all_preds.append(preds.cpu())
            all_labels.append(scores)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    spearman, _ = spearmanr(preds, labels)
    return spearman


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, args):
    """Train a model and return test spearman."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    best_val = -1
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids_a, mask_a, ids_b, mask_b, scores in train_loader:
            ids_a, mask_a = ids_a.to(device), mask_a.to(device)
            ids_b, mask_b = ids_b.to(device), mask_b.to(device)
            scores = scores.to(device)
            preds = model(ids_a, mask_a, ids_b, mask_b)
            loss = nn.functional.mse_loss(preds, scores)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_sp = evaluate(model, val_loader, device)
        if val_sp > best_val:
            best_val = val_sp
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= 15:
            break

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_sp = evaluate(model, test_loader, device)
    return test_sp, best_val


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data once
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    train_loader, val_loader, test_loader = get_dataloaders(
        word2idx, batch_size=64, max_len=50,
    )

    results = []

    # ═══════════════════════════════════════════════════════
    # Define all ablations
    # ═══════════════════════════════════════════════════════

    ablations = [
        ("Mean-pool (no train)", "no_train"),
        ("Mean-pool features", "mean_pool"),
        ("Mean-pool means", "mean_pool_mu"),
        ("Softmax attention", "softmax_attn"),
        ("No transmittance", "no_transmittance"),
        ("SGS-1pass", "sgs_1"),
        ("SGS-2pass", "sgs_2"),
        ("SGS-full (P=4)", "sgs_4"),
        ("SGS-8pass", "sgs_8"),
    ]

    for name, ablation_type in ablations:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Reset seed for fair comparison
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Build fresh vocab for each run
        vocab = SemanticGaussianVocab(vocab_size=len(words), d_s=64, d_f=300)
        vocab.init_from_glove(vectors, freqs)
        vocab = vocab.to(device)

        try:
            if ablation_type == "no_train":
                # Just evaluate mean-pooling with GloVe init, no training
                model = MeanPoolModel(vocab).to(device)
                test_sp = evaluate(model, test_loader, device)
                val_sp = evaluate(model, val_loader, device)
                print(f"  Val: {val_sp:.4f} | Test: {test_sp:.4f}")
                results.append((name, test_sp, val_sp))
                continue

            elif ablation_type == "mean_pool":
                model = MeanPoolModel(vocab).to(device)

            elif ablation_type == "mean_pool_mu":
                model = MeanPoolMuModel(vocab).to(device)

            elif ablation_type == "softmax_attn":
                model = SoftmaxAttentionModel(vocab, d_s=64).to(device)

            elif ablation_type == "no_transmittance":
                encoder = SGSEncoder(len(words), d_s=64, d_f=300, n_passes=1, tau_init=64.0)
                encoder.vocab = vocab
                model = NoTransmittanceModel(encoder).to(device)

            elif ablation_type.startswith("sgs_"):
                n_passes = int(ablation_type.split("_")[1])
                encoder = SGSEncoder(len(words), d_s=64, d_f=300, n_passes=n_passes, tau_init=64.0)
                encoder.vocab = vocab
                model = SGSSimilarityModel(encoder).to(device)

            else:
                raise ValueError(f"Unknown ablation: {ablation_type}")

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {n_params:,}")

            # Zero-shot
            zs = evaluate(model, val_loader, device)
            print(f"  Zero-shot val: {zs:.4f}")

            # Train
            t0 = time.time()
            test_sp, val_sp = train_and_evaluate(
                model, train_loader, val_loader, test_loader, device, args,
            )
            dt = time.time() - t0
            print(f"  Val: {val_sp:.4f} | Test: {test_sp:.4f} | Time: {dt:.0f}s")
            results.append((name, test_sp, val_sp))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, None, None))

    # ═══════════════════════════════════════════════════════
    # Results table
    # ═══════════════════════════════════════════════════════

    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS — PHASE 1c")
    print(f"{'='*70}\n")
    print(f"{'Model':<30s} {'Val Spearman':>14s} {'Test Spearman':>15s} {'Status':>8s}")
    print("-" * 70)

    for name, test_sp, val_sp in sorted(results, key=lambda x: x[1] or 0, reverse=True):
        if test_sp is None:
            print(f"{name:<30s} {'—':>14s} {'FAILED':>15s} {'—':>8s}")
        else:
            status = "PASS" if test_sp >= 0.78 else ("CHECK" if test_sp >= 0.58 else "FAIL")
            print(f"{name:<30s} {val_sp:>14.4f} {test_sp:>15.4f} {status:>8s}")

    # Save
    with open("ablation_results.json", "w") as f:
        json.dump(
            {name: {"test": t, "val": v} for name, t, v in results},
            f, indent=2,
        )

    # Key comparisons
    rd = {name: t for name, t, v in results if t is not None}
    sgs = rd.get("SGS-full (P=4)")
    mp = rd.get("Mean-pool features")
    mp_nt = rd.get("Mean-pool (no train)")
    sa = rd.get("Softmax attention")

    print(f"\n{'='*70}")
    print("KEY COMPARISONS")
    print(f"{'='*70}")
    if sgs and mp_nt:
        print(f"  SGS vs Mean-pool (no train): {sgs - mp_nt:+.4f}")
    if sgs and mp:
        print(f"  SGS vs Mean-pool (trained):  {sgs - mp:+.4f}")
    if sgs and sa:
        diff = sgs - sa
        print(f"  SGS vs Softmax attention:    {diff:+.4f}")
        if diff > 0:
            print("  -> Alpha-compositing beats softmax!")
        else:
            print("  -> Softmax wins -> consider Gaussian Transformer hybrid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
