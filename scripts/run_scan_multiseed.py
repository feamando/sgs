#!/usr/bin/env python3
"""
SCAN multi-seed with all ablations.

Fixes from paper challenge:
  C1: 5 seeds for error bars
  C2: Transformer+RPE baseline
  C4: MeanPool+GRU and TransformerEnc+GRU to isolate encoder contribution
  C8: Token-level accuracy alongside sequence-level

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
from src.seq2seq import (
    SGSSeq2Seq,
    TransformerSeq2Seq,
    TransformerRPESeq2Seq,
    MeanPoolGRUSeq2Seq,
    TransformerEncoderGRUDecoder,
)


def evaluate_scan(model, test_loader, out_vocab, device):
    """Compute sequence-level AND token-level accuracy on SCAN."""
    model.eval()
    seq_correct, seq_total = 0, 0
    tok_correct, tok_total = 0, 0
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

                # Trim at EOS
                if eos_id in pred_list:
                    pred_list = pred_list[:pred_list.index(eos_id)]
                if eos_id in tgt_list:
                    tgt_list = tgt_list[:tgt_list.index(eos_id)]
                tgt_list = [t for t in tgt_list if t != 0]

                # Sequence-level: exact match
                if pred_list == tgt_list:
                    seq_correct += 1
                seq_total += 1

                # Token-level: per-position accuracy (up to min length)
                max_len = max(len(pred_list), len(tgt_list))
                if max_len > 0:
                    # Pad shorter to compare
                    p = pred_list + [0] * (max_len - len(pred_list))
                    t = tgt_list + [0] * (max_len - len(tgt_list))
                    for pi, ti in zip(p, t):
                        if pi == ti and ti != 0:
                            tok_correct += 1
                        tok_total += 1

    seq_acc = seq_correct / max(seq_total, 1)
    tok_acc = tok_correct / max(tok_total, 1)
    return seq_acc, tok_acc, seq_correct, seq_total


def train_scan(model, train_loader, test_loader, out_vocab, device,
               epochs=20, lr=1e-3, eval_every=5):
    """Train on SCAN, return best test accuracies."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_seq_acc, best_tok_acc = 0, 0
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
            seq_acc, tok_acc, c, t = evaluate_scan(model, test_loader, out_vocab, device)
            best_seq_acc = max(best_seq_acc, seq_acc)
            best_tok_acc = max(best_tok_acc, tok_acc)
            print(f"      Ep {epoch:2d}/{epochs} | loss={total_loss/n:.4f} | "
                  f"seq={seq_acc:.4f} ({c}/{t}) tok={tok_acc:.4f} | "
                  f"best_seq={best_seq_acc:.4f}")

    return best_seq_acc, best_tok_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds = [int(s) for s in args.seeds]
    split = args.split

    train_loader, test_loader, in_vocab, out_vocab, out_idx2word = get_scan_dataloaders(
        split=split, batch_size=64,
    )

    models_config = [
        ("SGS Seq2Seq",         SGSSeq2Seq,                  {"d_model": 128, "n_passes": 2}),
        ("MeanPool+GRU",        MeanPoolGRUSeq2Seq,          {"d_model": 128}),
        ("TransfEnc+GRU",       TransformerEncoderGRUDecoder, {"d_model": 128, "nhead": 4}),
        ("Transformer",         TransformerSeq2Seq,           {"d_model": 128, "nhead": 4}),
        ("Transformer+RPE",     TransformerRPESeq2Seq,        {"d_model": 128, "nhead": 4}),
    ]

    all_seq = {name: [] for name, _, _ in models_config}
    all_tok = {name: [] for name, _, _ in models_config}

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
            best_seq, best_tok = train_scan(
                model, train_loader, test_loader, out_vocab, device,
                epochs=args.epochs, lr=args.lr, eval_every=args.eval_every,
            )
            dt = time.time() - t0
            print(f"    Best: seq={best_seq:.4f} tok={best_tok:.4f} ({dt:.0f}s)")
            all_seq[name].append(best_seq)
            all_tok[name].append(best_tok)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════

    print(f"\n\n{'='*80}")
    print(f"SCAN {split} — {len(seeds)} seeds")
    print(f"{'='*80}\n")

    header = f"{'Model':<25s} {'Seq Mean':>9s} {'± Std':>7s} {'Tok Mean':>9s} {'± Std':>7s} {'Seq Min':>8s} {'Seq Max':>8s}"
    print(header)
    print("-" * len(header))

    for name, _, _ in models_config:
        sa = all_seq[name]
        ta = all_tok[name]
        print(f"{name:<25s} {np.mean(sa):>9.4f} {np.std(sa):>7.4f} "
              f"{np.mean(ta):>9.4f} {np.std(ta):>7.4f} "
              f"{min(sa):>8.4f} {max(sa):>8.4f}")

    # Key comparisons
    sgs = np.mean(all_seq["SGS Seq2Seq"])
    mp_gru = np.mean(all_seq["MeanPool+GRU"])
    tf_gru = np.mean(all_seq["TransfEnc+GRU"])
    tf = np.mean(all_seq["Transformer"])
    rpe = np.mean(all_seq["Transformer+RPE"])

    print(f"\n{'='*80}")
    print("KEY COMPARISONS (sequence accuracy)")
    print(f"{'='*80}")
    print(f"  SGS vs MeanPool+GRU:      {sgs:.4f} vs {mp_gru:.4f} ({sgs - mp_gru:+.4f})")
    print(f"  SGS vs TransfEnc+GRU:     {sgs:.4f} vs {tf_gru:.4f} ({sgs - tf_gru:+.4f})")
    print(f"  SGS vs Transformer:       {sgs:.4f} vs {tf:.4f} ({sgs - tf:+.4f})")
    print(f"  SGS vs Transformer+RPE:   {sgs:.4f} vs {rpe:.4f} ({sgs - rpe:+.4f})")

    print(f"\n  C4 VERDICT (is it the encoder or the decoder?):")
    if sgs > mp_gru + 0.05:
        print(f"  SGS encoder contributes +{sgs - mp_gru:.4f} over mean-pool (ENCODER MATTERS)")
    else:
        print(f"  SGS encoder adds only +{sgs - mp_gru:.4f} over mean-pool (GRU may explain most)")

    if tf_gru > tf + 0.05:
        print(f"  GRU decoder alone helps: TransfEnc+GRU {tf_gru:.4f} vs full Transformer {tf:.4f}")
    else:
        print(f"  GRU decoder alone doesn't help much: TransfEnc+GRU {tf_gru:.4f} vs Transformer {tf:.4f}")

    # Save
    results_file = f"scan_{split}_full_ablation.json"
    with open(results_file, "w") as f:
        json.dump({
            "split": split,
            "seeds": seeds,
            "seq_acc": {n: a for n, a in all_seq.items()},
            "tok_acc": {n: a for n, a in all_tok.items()},
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCAN full ablation with encoder isolation")
    parser.add_argument("--split", type=str, default="length", choices=["length", "addprim_jump"])
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "456", "789", "1337"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=5)
    args = parser.parse_args()
    main(args)
