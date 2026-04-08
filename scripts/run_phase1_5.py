#!/usr/bin/env python3
"""
Phase 1.5: Improved SGS with multi-head queries, IDF init, fair baselines.

Usage:
    python scripts/run_phase1_5.py --glove data/glove.6B.300d.txt
    python scripts/run_phase1_5.py --glove data/glove.6B.300d.txt --seeds 42 123 456
    python scripts/run_phase1_5.py --glove data/glove.6B.300d.txt --quick  # fewer epochs
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
from src.model import (
    SGSEncoder, SGSSimilarityModel,
    MeanPoolModel, SoftmaxAttentionModel,
    FairSoftmaxModel, NoTransmittanceModel,
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
    sp, _ = spearmanr(preds, labels)
    return sp


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs, lr):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val, best_state, patience = -1, None, 0

    for epoch in range(1, epochs + 1):
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

    if best_state:
        model.load_state_dict(best_state)
    test_sp = evaluate(model, test_loader, device)
    return test_sp, best_val


def build_vocab(words, vectors, freqs, d_s, d_f, device, idf_init=False, remove_pc1=False):
    """Build a fresh vocabulary with given settings."""
    vocab = SemanticGaussianVocab(vocab_size=len(words), d_s=d_s, d_f=d_f)
    vocab.init_from_glove(vectors, freqs, idf_init=idf_init, remove_pc1=remove_pc1)
    return vocab.to(device)


def run_single(name, model, train_loader, val_loader, test_loader, device, epochs, lr):
    """Run a single ablation and return results."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    zs = evaluate(model, val_loader, device)
    print(f"  Params: {n_params:,} | Zero-shot val: {zs:.4f}", end="")

    t0 = time.time()
    test_sp, val_sp = train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs, lr)
    dt = time.time() - t0
    print(f" | Val: {val_sp:.4f} | Test: {test_sp:.4f} | {dt:.0f}s")
    return test_sp, val_sp, zs, n_params


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    train_loader, val_loader, test_loader = get_dataloaders(word2idx, batch_size=64, max_len=50)

    seeds = [int(s) for s in args.seeds]
    d_f = 300
    epochs = args.epochs
    lr = args.lr

    # ═══════════════════════════════════════════════════════
    # Define Phase 1.5 ablations
    # ═══════════════════════════════════════════════════════

    def make_ablations(seed, vocab_base, vocab_idf, vocab_idf_pc1):
        """Return list of (name, model) tuples for one seed."""
        ablations = []

        # --- Baselines ---
        ablations.append(("Mean-pool (no train)", MeanPoolModel(vocab_base).to(device), False))

        ablations.append(("Mean-pool (trained)", MeanPoolModel(vocab_base).to(device), True))

        # --- Phase 1 best: SGS-2pass (original) ---
        enc = SGSEncoder(len(words), d_s=args.d_s, d_f=d_f, n_passes=2, tau_init=float(args.d_s))
        enc.vocab = vocab_base
        ablations.append(("SGS-2pass", SGSSimilarityModel(enc).to(device), True))

        # --- Phase 1.5: SGS-2pass + IDF init ---
        enc = SGSEncoder(len(words), d_s=args.d_s, d_f=d_f, n_passes=2, tau_init=float(args.d_s))
        enc.vocab = vocab_idf
        ablations.append(("SGS-2pass + IDF", SGSSimilarityModel(enc).to(device), True))

        # --- Phase 1.5: SGS-2pass + IDF + PC1 removal ---
        enc = SGSEncoder(len(words), d_s=args.d_s, d_f=d_f, n_passes=2, tau_init=float(args.d_s))
        enc.vocab = vocab_idf_pc1
        ablations.append(("SGS-2pass + IDF + PC1", SGSSimilarityModel(enc).to(device), True))

        # --- Phase 1.5: Multi-head (4 heads) + IDF + PC1 ---
        enc = SGSEncoder(len(words), d_s=args.d_s, d_f=d_f, n_passes=2, tau_init=float(args.d_s), n_heads=4)
        enc.vocab = vocab_idf_pc1
        ablations.append(("SGS-2pass 4head + IDF + PC1", SGSSimilarityModel(enc).to(device), True))

        # --- Phase 1.5: Multi-head (8 heads) + IDF + PC1 ---
        enc = SGSEncoder(len(words), d_s=args.d_s, d_f=d_f, n_passes=2, tau_init=float(args.d_s), n_heads=8)
        enc.vocab = vocab_idf_pc1
        ablations.append(("SGS-2pass 8head + IDF + PC1", SGSSimilarityModel(enc).to(device), True))

        # --- Fair softmax baseline (matched arch) ---
        ablations.append(("Fair Softmax (2-layer)", FairSoftmaxModel(vocab_idf_pc1, d_s=args.d_s, d_f=d_f, n_layers=2).to(device), True))

        # --- Bare softmax (Phase 1 baseline) ---
        ablations.append(("Softmax (bare)", SoftmaxAttentionModel(vocab_base, d_s=args.d_s).to(device), True))

        return ablations

    # ═══════════════════════════════════════════════════════
    # Run across seeds
    # ═══════════════════════════════════════════════════════

    all_results = {}  # name -> list of (test, val, zs) across seeds

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build vocabs (one per init strategy)
        vocab_base = build_vocab(words, vectors, freqs, args.d_s, d_f, device,
                                 idf_init=False, remove_pc1=False)
        vocab_idf = build_vocab(words, vectors, freqs, args.d_s, d_f, device,
                                idf_init=True, remove_pc1=False)
        vocab_idf_pc1 = build_vocab(words, vectors, freqs, args.d_s, d_f, device,
                                    idf_init=True, remove_pc1=True)

        ablations = make_ablations(seed, vocab_base, vocab_idf, vocab_idf_pc1)

        for name, model, needs_training in ablations:
            print(f"\n  --- {name} ---")
            torch.manual_seed(seed)

            if not needs_training:
                test_sp = evaluate(model, test_loader, device)
                val_sp = evaluate(model, val_loader, device)
                print(f"  Val: {val_sp:.4f} | Test: {test_sp:.4f}")
                zs = val_sp
                n_params = 0
            else:
                test_sp, val_sp, zs, n_params = run_single(
                    name, model, train_loader, val_loader, test_loader, device, epochs, lr,
                )

            if name not in all_results:
                all_results[name] = []
            all_results[name].append({
                'seed': seed, 'test': test_sp, 'val': val_sp, 'zs': zs, 'params': n_params,
            })

    # ═══════════════════════════════════════════════════════
    # Aggregate results
    # ═══════════════════════════════════════════════════════

    print(f"\n\n{'='*80}")
    print(f"PHASE 1.5 RESULTS ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    print(f"{'='*80}\n")

    header = f"{'Model':<35s} {'Test (mean)':>11s} {'± std':>8s} {'Val (mean)':>11s} {'ZS Val':>8s} {'Params':>10s}"
    print(header)
    print("-" * len(header))

    summary = []
    for name in dict.fromkeys(n for abl in [list(all_results.keys())] for n in abl):
        runs = all_results.get(name, [])
        if not runs:
            continue
        tests = [r['test'] for r in runs]
        vals = [r['val'] for r in runs]
        zss = [r['zs'] for r in runs]
        mean_t = np.mean(tests)
        std_t = np.std(tests) if len(tests) > 1 else 0
        mean_v = np.mean(vals)
        mean_zs = np.mean(zss)
        params = runs[0]['params']
        summary.append((name, mean_t, std_t, mean_v, mean_zs, params))

    for name, mean_t, std_t, mean_v, mean_zs, params in sorted(summary, key=lambda x: x[1], reverse=True):
        std_str = f"±{std_t:.4f}" if std_t > 0 else "—"
        param_str = f"{params:,}" if params > 0 else "—"
        print(f"{name:<35s} {mean_t:>11.4f} {std_str:>8s} {mean_v:>11.4f} {mean_zs:>8.4f} {param_str:>10s}")

    # Key comparisons
    rd = {name: mt for name, mt, _, _, _, _ in summary}
    sgs_idf_pc1 = rd.get("SGS-2pass + IDF + PC1")
    sgs_mh = rd.get("SGS-2pass 4head + IDF + PC1")
    sgs_orig = rd.get("SGS-2pass")
    fair_sfm = rd.get("Fair Softmax (2-layer)")
    bare_sfm = rd.get("Softmax (bare)")
    mp = rd.get("Mean-pool (trained)")
    mp_nt = rd.get("Mean-pool (no train)")

    print(f"\n{'='*80}")
    print("KEY COMPARISONS")
    print(f"{'='*80}")

    def cmp(a_name, b_name, a_val, b_val):
        if a_val is not None and b_val is not None:
            d = a_val - b_val
            print(f"  {a_name} vs {b_name}: {d:+.4f}")

    cmp("SGS-2pass + IDF + PC1", "Mean-pool (no train)", sgs_idf_pc1, mp_nt)
    cmp("SGS-2pass + IDF + PC1", "Mean-pool (trained)", sgs_idf_pc1, mp)
    cmp("SGS-2pass + IDF + PC1", "SGS-2pass (original)", sgs_idf_pc1, sgs_orig)
    cmp("SGS 4-head", "SGS-2pass + IDF + PC1", sgs_mh, sgs_idf_pc1)
    cmp("SGS 4-head", "Fair Softmax (2-layer)", sgs_mh, fair_sfm)
    cmp("Fair Softmax", "Softmax (bare)", fair_sfm, bare_sfm)

    # Kill gate
    best_sgs = max(v for k, v in rd.items() if 'SGS' in k) if any('SGS' in k for k in rd) else 0
    print(f"\n  Best SGS test Spearman: {best_sgs:.4f}")
    if best_sgs >= 0.72:
        print("  PASS -> Proceed to Phase 2")
    elif best_sgs >= 0.65:
        print("  INVESTIGATE -> Check if improvements help significantly")
    else:
        print("  KILL -> Rendering doesn't compose language well enough")

    # Save
    with open("phase1_5_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to phase1_5_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGS Phase 1.5")
    parser.add_argument("--glove", type=str, required=True)
    parser.add_argument("--d_s", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "456"])
    parser.add_argument("--quick", action="store_true", help="Quick run: 1 seed, 20 epochs")
    args = parser.parse_args()

    if args.quick:
        args.seeds = ["42"]
        args.epochs = 20

    main(args)
