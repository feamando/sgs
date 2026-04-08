#!/usr/bin/env python3
"""
Phase 2: Two experiments to push SGS toward 0.72+

Experiment 1: Train on AllNLI (~500K pairs), evaluate on STS-B
Experiment 2: d_s sweep (32, 64, 128, 300) on STS-B

Usage:
    python scripts/run_phase2.py --glove data/glove.6B.300d.txt --exp nli
    python scripts/run_phase2.py --glove data/glove.6B.300d.txt --exp ds_sweep
    python scripts/run_phase2.py --glove data/glove.6B.300d.txt --exp all
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
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.data import (
    load_glove, get_dataloaders, get_nli_dataloader,
    collate_fn,
)
from src.gaussian import SemanticGaussianVocab
from src.model import (
    SGSEncoder, SGSSimilarityModel,
    MeanPoolModel, FairSoftmaxModel,
)


def evaluate_stsb(model_or_encoder, loader, device, is_encoder=False):
    """Evaluate on STS-B using cosine similarity."""
    if is_encoder:
        encoder = model_or_encoder
    else:
        encoder = model_or_encoder.encoder if hasattr(model_or_encoder, 'encoder') else None

    if encoder is not None:
        encoder.eval()
    else:
        model_or_encoder.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids_a, mask_a, ids_b, mask_b, scores in loader:
            ids_a, mask_a = ids_a.to(device), mask_a.to(device)
            ids_b, mask_b = ids_b.to(device), mask_b.to(device)

            if is_encoder:
                mean_a = encoder(ids_a, mask_a)
                mean_b = encoder(ids_b, mask_b)
                cos = F.cosine_similarity(mean_a, mean_b, dim=-1)
                preds = cos * 5.0
            else:
                preds = model_or_encoder(ids_a, mask_a, ids_b, mask_b)

            all_preds.append(preds.cpu())
            all_labels.append(scores)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    sp, _ = spearmanr(preds, labels)
    return sp


# ═══════════════════════════════════════════════════════════
# Experiment 1: NLI Contrastive Training
# ═══════════════════════════════════════════════════════════

class MultipleNegativesRankingLoss(nn.Module):
    """
    Contrastive loss for NLI training (same as sentence-transformers).

    Given (anchor, positive, negative) triplets, uses in-batch negatives:
    for each anchor, all other positives in the batch serve as negatives.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_emb, positive_emb):
        """
        Args:
            anchor_emb: [batch, d_f]
            positive_emb: [batch, d_f]
        """
        # Cosine similarity matrix: [batch, batch]
        anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
        positive_norm = F.normalize(positive_emb, p=2, dim=1)
        sim_matrix = anchor_norm @ positive_norm.T / self.temperature

        # Labels: diagonal (each anchor matches its own positive)
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

        # Cross-entropy: correct positive should have highest similarity
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


def train_nli(encoder, nli_loader, stsb_val_loader, stsb_test_loader,
              device, epochs, lr):
    """Train encoder on NLI with contrastive loss, evaluate on STS-B."""
    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )
    loss_fn = MultipleNegativesRankingLoss(temperature=0.05)

    best_val, best_state = -1, None

    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss = 0
        n_batches = 0

        for a_ids, a_mask, p_ids, p_mask, n_ids, n_mask in nli_loader:
            a_ids, a_mask = a_ids.to(device), a_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)

            a_emb = encoder(a_ids, a_mask)
            p_emb = encoder(p_ids, p_mask)

            loss = loss_fn(a_emb, p_emb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate on STS-B
        val_sp = evaluate_stsb(encoder, stsb_val_loader, device, is_encoder=True)

        if val_sp > best_val:
            best_val = val_sp
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

        avg_loss = total_loss / max(n_batches, 1)

        extra = ""
        if hasattr(encoder, 'tau'):
            extra = f" | τ={encoder.tau.item():.1f}"

        print(f"  Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} | "
              f"stsb_val={val_sp:.4f} | best={best_val:.4f}{extra}")

    # Load best and evaluate test
    if best_state:
        encoder.load_state_dict(best_state)
    test_sp = evaluate_stsb(encoder, stsb_test_loader, device, is_encoder=True)
    return test_sp, best_val


def run_nli_experiment(args, word2idx, vectors, freqs, words, device):
    """Experiment 1: NLI training."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: AllNLI Contrastive Training → STS-B Evaluation")
    print(f"{'='*70}\n")

    # STS-B for evaluation only
    _, stsb_val, stsb_test = get_dataloaders(word2idx, batch_size=64, max_len=50)

    # AllNLI for training
    nli_loader = get_nli_dataloader(
        word2idx, batch_size=args.nli_batch_size, max_len=50,
        max_samples=args.nli_max_samples,
    )

    results = []

    models_to_test = [
        ("SGS-2pass (NLI)", "sgs"),
        ("Mean-pool (NLI)", "mean_pool"),
        ("Fair Softmax (NLI)", "fair_softmax"),
    ]

    for name, model_type in models_to_test:
        print(f"\n--- {name} ---")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        d_s = args.d_s
        d_f = 300
        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab = vocab.to(device)

        if model_type == "sgs":
            encoder = SGSEncoder(len(words), d_s=d_s, d_f=d_f, n_passes=2,
                                 tau_init=float(d_s))
            encoder.vocab = vocab
            encoder = encoder.to(device)
        elif model_type == "mean_pool":
            # For mean-pool, create a simple encoder wrapper
            encoder = MeanPoolEncoder(vocab).to(device)
        elif model_type == "fair_softmax":
            encoder = FairSoftmaxEncoder(vocab, d_s=d_s, d_f=d_f).to(device)

        # Zero-shot STS-B before NLI training
        zs = evaluate_stsb(encoder, stsb_val, device, is_encoder=True)
        print(f"  Zero-shot STS-B val: {zs:.4f}")

        n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        test_sp, val_sp = train_nli(
            encoder, nli_loader, stsb_val, stsb_test, device,
            epochs=args.nli_epochs, lr=args.nli_lr,
        )
        dt = time.time() - t0

        print(f"  Final: STS-B val={val_sp:.4f} test={test_sp:.4f} ({dt:.0f}s)")
        results.append((name, test_sp, val_sp, zs, n_params))

    return results


class MeanPoolEncoder(nn.Module):
    """Wraps vocab as a mean-pool encoder for NLI training."""
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, token_ids, mask=None):
        _, _, _, features = self.vocab.get_params(token_ids)
        if mask is not None:
            features = features * mask.float().unsqueeze(-1)
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            return features.sum(dim=1) / lengths
        return features.mean(dim=1)


class FairSoftmaxEncoder(nn.Module):
    """Wraps FairSoftmaxModel as an encoder for NLI training."""
    def __init__(self, vocab, d_s=64, d_f=300):
        super().__init__()
        self.inner = __import__('src.model', fromlist=['FairSoftmaxModel']).FairSoftmaxModel(
            vocab, d_s=d_s, d_f=d_f, n_layers=2,
        )

    def forward(self, token_ids, mask=None):
        return self.inner._encode(token_ids, mask)


# ═══════════════════════════════════════════════════════════
# Experiment 2: d_s sweep
# ═══════════════════════════════════════════════════════════

def run_ds_sweep(args, word2idx, vectors, freqs, words, device):
    """Experiment 2: Vary splatting space dimension."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Splatting Space Dimension Sweep")
    print(f"{'='*70}\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        word2idx, batch_size=64, max_len=50,
    )

    results = []

    for d_s in [32, 64, 128, 300]:
        print(f"\n--- d_s = {d_s} ---")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        d_f = 300
        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab = vocab.to(device)

        encoder = SGSEncoder(len(words), d_s=d_s, d_f=d_f, n_passes=2,
                             tau_init=float(d_s))
        encoder.vocab = vocab
        model = SGSSimilarityModel(encoder).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        zs = evaluate_stsb(model, val_loader, device)
        print(f"  Params: {n_params:,} | Zero-shot val: {zs:.4f}")

        # Train
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        best_val, best_state, patience = -1, None, 0
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            for ids_a, mask_a, ids_b, mask_b, scores in train_loader:
                ids_a, mask_a = ids_a.to(device), mask_a.to(device)
                ids_b, mask_b = ids_b.to(device), mask_b.to(device)
                scores = scores.to(device)
                preds = model(ids_a, mask_a, ids_b, mask_b)
                loss = F.mse_loss(preds, scores)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            val_sp = evaluate_stsb(model, val_loader, device)
            if val_sp > best_val:
                best_val = val_sp
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 15:
                break

        dt = time.time() - t0
        if best_state:
            model.load_state_dict(best_state)
        test_sp = evaluate_stsb(model, test_loader, device)

        print(f"  Val: {best_val:.4f} | Test: {test_sp:.4f} | τ={encoder.tau.item():.1f} | {dt:.0f}s")
        results.append((f"SGS-2pass d_s={d_s}", test_sp, best_val, zs, n_params))

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)

    all_results = []

    if args.exp in ("nli", "all"):
        results = run_nli_experiment(args, word2idx, vectors, freqs, words, device)
        all_results.extend(results)

    if args.exp in ("ds_sweep", "all"):
        results = run_ds_sweep(args, word2idx, vectors, freqs, words, device)
        all_results.extend(results)

    # Summary
    print(f"\n\n{'='*70}")
    print("PHASE 2 SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Model':<35s} {'Test':>8s} {'Val':>8s} {'ZS Val':>8s} {'Params':>10s}")
    print("-" * 75)
    for name, test_sp, val_sp, zs, n_params in sorted(all_results, key=lambda x: x[1], reverse=True):
        print(f"{name:<35s} {test_sp:>8.4f} {val_sp:>8.4f} {zs:>8.4f} {n_params:>10,}")

    best = max(all_results, key=lambda x: x[1])
    print(f"\nBest: {best[0]} → test={best[1]:.4f}")
    if best[1] >= 0.72:
        print("PASS → Proceed to downstream tasks and compositionality benchmarks")
    elif best[1] >= 0.68:
        print("CLOSE → Architecture works, may need different task/larger data")
    else:
        print("INVESTIGATE → Check if architecture ceiling is real")

    with open("phase2_results.json", "w") as f:
        json.dump({name: {"test": t, "val": v, "zs": z} for name, t, v, z, _ in all_results}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGS Phase 2")
    parser.add_argument("--glove", type=str, required=True)
    parser.add_argument("--exp", type=str, default="all", choices=["nli", "ds_sweep", "all"])
    parser.add_argument("--d_s", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for STS-B training (d_s sweep)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    # NLI-specific
    parser.add_argument("--nli_epochs", type=int, default=3, help="NLI training epochs (large dataset)")
    parser.add_argument("--nli_lr", type=float, default=2e-4, help="NLI learning rate (lower for contrastive)")
    parser.add_argument("--nli_batch_size", type=int, default=128, help="NLI batch size")
    parser.add_argument("--nli_max_samples", type=int, default=0, help="Max NLI triplets (0=all)")
    args = parser.parse_args()
    main(args)
