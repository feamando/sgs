#!/usr/bin/env python3
"""
Phase 1b: Train SGS on STS-B and compare to baselines.

Usage:
    python scripts/train_stsb.py --glove data/glove.6B.300d.txt
    python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model mean_pool
    python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model softmax_attn
    python scripts/train_stsb.py --glove data/glove.6B.300d.txt --n_passes 1
"""

import argparse
import os
import sys
import time
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

from src.data import load_glove, get_dataloaders
from src.gaussian import SemanticGaussianVocab
from src.model import (
    SGSEncoder, SGSSimilarityModel,
    MeanPoolModel, MeanPoolMuModel,
    SoftmaxAttentionModel, NoTransmittanceModel,
)


def evaluate(model, loader, device):
    """Compute Spearman correlation on a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

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
    mse = ((preds - labels) ** 2).mean()
    return spearman, mse


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for ids_a, mask_a, ids_b, mask_b, scores in loader:
        ids_a, mask_a = ids_a.to(device), mask_a.to(device)
        ids_b, mask_b = ids_b.to(device), mask_b.to(device)
        scores = scores.to(device)

        preds = model(ids_a, mask_a, ids_b, mask_b)
        loss = nn.functional.mse_loss(preds, scores)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def build_model(
    args, vocab: SemanticGaussianVocab, device: torch.device
) -> nn.Module:
    """Build model based on --model flag."""
    if args.model == "sgs":
        encoder = SGSEncoder(
            vocab_size=vocab.vocab_size,
            d_s=args.d_s,
            d_f=args.d_f,
            n_passes=args.n_passes,
            tau_init=float(args.d_s),
        )
        # Share vocabulary with encoder
        encoder.vocab = vocab
        model = SGSSimilarityModel(encoder)

    elif args.model == "mean_pool":
        model = MeanPoolModel(vocab)

    elif args.model == "mean_pool_mu":
        model = MeanPoolMuModel(vocab)

    elif args.model == "softmax_attn":
        model = SoftmaxAttentionModel(vocab, d_s=args.d_s)

    elif args.model == "no_transmittance":
        encoder = SGSEncoder(
            vocab_size=vocab.vocab_size,
            d_s=args.d_s,
            d_f=args.d_f,
            n_passes=1,
            tau_init=float(args.d_s),
        )
        encoder.vocab = vocab
        model = NoTransmittanceModel(encoder)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model.to(device)


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=args.vocab_size)
    train_loader, val_loader, test_loader = get_dataloaders(
        word2idx, batch_size=args.batch_size, max_len=args.max_len,
    )

    # Build vocabulary (shared across all models for fair comparison)
    vocab = SemanticGaussianVocab(
        vocab_size=len(words), d_s=args.d_s, d_f=args.d_f,
    )
    vocab.init_from_glove(vectors, freqs)
    vocab = vocab.to(device)

    # Build model
    model = build_model(args, vocab, device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model} (n_passes={args.n_passes})")
    print(f"Parameters: {n_params:,}")

    # Evaluate before training (zero-shot with GloVe init)
    val_spearman_0, val_mse_0 = evaluate(model, val_loader, device)
    print(f"\nZero-shot (GloVe init): val_spearman={val_spearman_0:.4f}, val_mse={val_mse_0:.4f}")

    # Check if model needs training (baselines don't learn beyond GloVe)
    trainable_beyond_vocab = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and 'vocab' not in name
    )

    if trainable_beyond_vocab == 0 and args.model in ("mean_pool", "mean_pool_mu"):
        # Pure baseline — just evaluate, optionally train vocab embeddings
        print(f"\n{'='*60}")
        print(f"Baseline model (no extra parameters)")

        if args.train_vocab:
            print("Training vocabulary embeddings only...")
            optimizer = torch.optim.AdamW(
                vocab.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
        else:
            print("No training — using GloVe initialization directly")
            test_spearman, test_mse = evaluate(model, test_loader, device)
            print(f"Test: spearman={test_spearman:.4f}, mse={test_mse:.4f}")
            return

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
    else:
        scheduler = None

    # Training loop
    best_val_spearman = val_spearman_0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_spearman, val_mse = evaluate(model, val_loader, device)
        dt = time.time() - t0

        if scheduler:
            scheduler.step()

        # Track best
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{args.model}.pt")
        else:
            patience_counter += 1

        # Log
        extra = ""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'tau'):
            extra = f" | τ={model.encoder.tau.item():.1f}"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_spearman={val_spearman:.4f} | "
            f"val_mse={val_mse:.4f} | "
            f"best={best_val_spearman:.4f}@{best_epoch} | "
            f"{dt:.1f}s{extra}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    # Load best model
    if os.path.exists(f"best_{args.model}.pt"):
        model.load_state_dict(torch.load(f"best_{args.model}.pt", weights_only=True))

    test_spearman, test_mse = evaluate(model, test_loader, device)
    val_spearman, val_mse = evaluate(model, val_loader, device)

    print(f"  Model:         {args.model} (n_passes={args.n_passes})")
    print(f"  Val Spearman:  {val_spearman:.4f}")
    print(f"  Test Spearman: {test_spearman:.4f}")
    print(f"  Test MSE:      {test_mse:.4f}")
    print(f"  Best epoch:    {best_epoch}")

    # Kill gate check
    print(f"\n{'='*60}")
    print("KILL GATE CHECK")
    print(f"{'='*60}")
    if test_spearman >= 0.78:
        print(f"  ✓ PASS (>= 0.78) — SGS composition works!")
        print(f"  → Proceed to Phase 2")
    elif test_spearman >= 0.58:
        print(f"  ~ INVESTIGATE ({test_spearman:.4f}) — better than random, weaker than SIF")
        print(f"  → Run ablations to understand what's contributing")
    else:
        print(f"  ✗ FAIL (< 0.58) — SGS composition doesn't work")
        print(f"  → Pivot to Gaussian Transformer hybrid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGS Phase 1: STS-B Training")

    # Data
    parser.add_argument("--glove", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=50)

    # Model
    parser.add_argument("--model", type=str, default="sgs",
                        choices=["sgs", "mean_pool", "mean_pool_mu",
                                 "softmax_attn", "no_transmittance"])
    parser.add_argument("--d_s", type=int, default=64)
    parser.add_argument("--d_f", type=int, default=300)
    parser.add_argument("--n_passes", type=int, default=4)
    parser.add_argument("--train_vocab", action="store_true",
                        help="Train vocabulary embeddings for baselines")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
