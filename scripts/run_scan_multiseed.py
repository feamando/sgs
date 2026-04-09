#!/usr/bin/env python3
"""
SCAN multi-seed + RPE baseline (fixes C1 and C2 from paper challenge).

Runs SGS, vanilla Transformer, and Transformer+RPE on SCAN length split
across 5 seeds to get mean ± std.

Usage:
    python scripts/run_scan_multiseed.py
    python scripts/run_scan_multiseed.py --seeds 42 123 456 789 1337
    python scripts/run_scan_multiseed.py --split addprim_jump
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

from src.scan import get_scan_dataloaders
from src.seq2seq import SGSSeq2Seq, TransformerSeq2Seq, TransformerRPESeq2Seq


def evaluate_scan(model, test_loader, out_vocab, device):
    """Compute exact sequence-level accuracy on SCAN."""
    model.eval()
    correct, total = 0, 0
    eos_id = out_vocab["<EOS>"]

    with torch.no_grad():
        for src, src_mask, tgt, tgt_mask in test_loader:
            src, src_mask = src.to(device), src_mask.to(device)
            tgt = tgt.to(device)

            preds = model.greedy_decode(
                src, src_mask, bos_id=out_vocab["<BOS>"], eos_id=eos_id,
            )

            target_seqs = tgt[:, 1:]  # remove BOS
            for i in range(preds.shape[0]):
                pred_list = preds[i].tolist()
                tgt_list = target_seqs[i].tolist()

                if eos_id in pred_list:
                    pred_list = pred_list[:pred_list.index(eos_id)]
                if eos_id in tgt_list:
                    tgt_list = tgt_list[:tgt_list.index(eos_id)]
                tgt_list = [t for t in tgt_list if t != 0]

                if pred_list == tgt_list:
                    correct += 1
                total += 1

    return correct / max(total, 1), correct, total


def train_scan(model, train_loader, test_loader, out_vocab, device,
               epochs=20, lr=1e-3, eval_every=5):
    """Train on SCAN, return best test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0, 0
        for src, src_mask, tgt, tgt_mask in train_loader:
            src, src_mask = src.to(device), src_mask.to(device)
            tgt, tgt_mask = tgt.to(device), tgt_mask.to(device)

            logits = model(src, src_mask, tgt, tgt_mask)
            target = tgt[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        if epoch % eval_every == 0 or epoch == epochs:
            acc, c, t = evaluate_scan(model, test_loader, out_vocab, device)
            best_acc = max(best_acc, acc)
            print(f"      Epoch {epoch:2d}/{epochs} | loss={total_loss/n:.4f} | "
                  f"acc={acc:.4f} ({c}/{t}) | best={best_acc:.4f}")

    return best_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds = [int(s) for s in args.seeds]
    split = args.split

    # Load data once (same across seeds)
    train_loader, test_loader, in_vocab, out_vocab, out_idx2word = get_scan_dataloaders(
        split=split, batch_size=64,
    )

    models_config = [
        ("SGS Seq2Seq", SGSSeq2Seq, {"d_model": 128, "n_passes": 2}),
        ("Transformer", TransformerSeq2Seq, {"d_model": 128, "nhead": 4}),
        ("Transformer+RPE", TransformerRPESeq2Seq, {"d_model": 128, "nhead": 4}),
    ]

    all_results = {name: [] for name, _, _ in models_config}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"  SEED {seed} — SCAN {split}")
        print(f"{'#'*60}")

        for name, model_cls, kwargs in models_config:
            print(f"\n    --- {name} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_cls(len(in_vocab), len(out_vocab), **kwargs).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Params: {n_params:,}")

            t0 = time.time()
            best_acc = train_scan(
                model, train_loader, test_loader, out_vocab, device,
                epochs=args.epochs, lr=args.lr, eval_every=args.eval_every,
            )
            dt = time.time() - t0
            print(f"    Best: {best_acc:.4f} ({dt:.0f}s)")
            all_results[name].append(best_acc)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════

    print(f"\n\n{'='*70}")
    print(f"SCAN {split} — {len(seeds)} seeds")
    print(f"{'='*70}\n")
    print(f"{'Model':<25s} {'Mean':>8s} {'± Std':>8s} {'Min':>8s} {'Max':>8s} {'Seeds':>6s}")
    print("-" * 60)

    for name, _, _ in models_config:
        accs = all_results[name]
        mean = np.mean(accs)
        std = np.std(accs)
        print(f"{name:<25s} {mean:>8.4f} {std:>8.4f} {min(accs):>8.4f} {max(accs):>8.4f} {len(accs):>6d}")

    # Key comparison
    sgs_accs = all_results["SGS Seq2Seq"]
    tfm_accs = all_results["Transformer"]
    rpe_accs = all_results["Transformer+RPE"]

    print(f"\n  SGS vs Transformer:     {np.mean(sgs_accs):.4f} vs {np.mean(tfm_accs):.4f}")
    print(f"  SGS vs Transformer+RPE: {np.mean(sgs_accs):.4f} vs {np.mean(rpe_accs):.4f}")

    if np.mean(sgs_accs) > np.mean(rpe_accs):
        print(f"  SGS beats Transformer+RPE by {np.mean(sgs_accs) - np.mean(rpe_accs):+.4f}")
    else:
        print(f"  Transformer+RPE beats SGS by {np.mean(rpe_accs) - np.mean(sgs_accs):+.4f}")

    # Save
    results_file = f"scan_{split}_multiseed.json"
    with open(results_file, "w") as f:
        json.dump({
            "split": split,
            "seeds": seeds,
            "results": {name: accs for name, accs in all_results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCAN multi-seed + RPE baseline")
    parser.add_argument("--split", type=str, default="length", choices=["length", "addprim_jump"])
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "456", "789", "1337"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=5)
    args = parser.parse_args()
    main(args)
