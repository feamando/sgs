#!/usr/bin/env python3
"""
Paper fixes: M2 (kernel isolation), M6 (hybrid), M4 (positioning).

M2: Is the zero-shot advantage from the Gaussian kernel or the rendering equation?
    → GaussianKernelSoftmax: softmax over kernel values (same kernel, no transmittance)
M6: Does SGS + softmax hybrid combine the best of both?
    → HybridSGSSoftmax: SGS rendering pass 1, softmax attention pass 2

Runs on STS-B with 3 seeds. Includes zero-shot evaluation.

Usage:
    python scripts/run_paper_fixes.py --glove data/glove.6B.300d.txt
    python scripts/run_paper_fixes.py --glove data/glove.6B.300d.txt --quick
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
    MeanPoolModel, SoftmaxAttentionModel, FairSoftmaxModel,
    GaussianKernelSoftmaxModel, HybridSGSSoftmaxModel,
)


def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for ids_a, mask_a, ids_b, mask_b, scores in loader:
            ids_a, mask_a = ids_a.to(device), mask_a.to(device)
            ids_b, mask_b = ids_b.to(device), mask_b.to(device)
            preds = model(ids_a, mask_a, ids_b, mask_b)
            preds_all.append(preds.cpu())
            labels_all.append(scores)
    p = torch.cat(preds_all).numpy()
    l = torch.cat(labels_all).numpy()
    sp, _ = spearmanr(p, l)
    return sp


def train_eval(model, train_loader, val_loader, test_loader, device, epochs, lr):
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
            loss = nn.functional.mse_loss(model(ids_a, mask_a, ids_b, mask_b), scores)
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    train_loader, val_loader, test_loader = get_dataloaders(word2idx, batch_size=64, max_len=50)

    seeds = [int(s) for s in args.seeds]
    d_s, d_f = 64, 300

    all_results = {}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"  SEED {seed}")
        print(f"{'#'*60}")

        # Fresh vocab per seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab = vocab.to(device)

        models = [
            ("Mean-pool (no train)", MeanPoolModel(vocab).to(device), False),
            ("SGS-2pass", None, True),  # built below with shared vocab
            ("GaussKernel+Softmax (M2)", GaussianKernelSoftmaxModel(vocab, d_s=d_s).to(device), True),
            ("Hybrid SGS+Softmax (M6)", HybridSGSSoftmaxModel(vocab, d_s=d_s, d_f=d_f).to(device), True),
            ("Fair Softmax", FairSoftmaxModel(vocab, d_s=d_s, d_f=d_f, n_layers=2).to(device), True),
            ("Softmax (bare)", SoftmaxAttentionModel(vocab, d_s=d_s).to(device), True),
        ]

        # Build SGS with shared vocab
        torch.manual_seed(seed)
        enc = SGSEncoder(len(words), d_s=d_s, d_f=d_f, n_passes=2, tau_init=float(d_s))
        enc.vocab = vocab
        sgs_model = SGSSimilarityModel(enc).to(device)
        models[1] = ("SGS-2pass", sgs_model, True)

        for name, model, needs_training in models:
            print(f"\n  --- {name} ---")
            torch.manual_seed(seed)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            zs = evaluate(model, val_loader, device)
            print(f"  Params: {n_params:,} | Zero-shot val: {zs:.4f}", end="")

            if not needs_training:
                test_sp = evaluate(model, test_loader, device)
                print(f" | Test: {test_sp:.4f}")
                result = {'test': test_sp, 'val': zs, 'zs': zs, 'params': n_params}
            else:
                t0 = time.time()
                test_sp, val_sp = train_eval(
                    model, train_loader, val_loader, test_loader, device, args.epochs, args.lr,
                )
                dt = time.time() - t0
                print(f" | Val: {val_sp:.4f} | Test: {test_sp:.4f} | {dt:.0f}s")
                result = {'test': test_sp, 'val': val_sp, 'zs': zs, 'params': n_params}

            if name not in all_results:
                all_results[name] = []
            all_results[name].append(result)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════

    print(f"\n\n{'='*80}")
    print(f"PAPER FIXES — {len(seeds)} seeds")
    print(f"{'='*80}\n")

    header = f"{'Model':<35s} {'Test':>8s} {'± Std':>7s} {'ZS Val':>8s} {'± Std':>7s}"
    print(header)
    print("-" * len(header))

    summary = []
    for name in all_results:
        runs = all_results[name]
        tests = [r['test'] for r in runs]
        zss = [r['zs'] for r in runs]
        summary.append((name, np.mean(tests), np.std(tests), np.mean(zss), np.std(zss)))

    for name, mt, st, mz, sz in sorted(summary, key=lambda x: x[1], reverse=True):
        std_t = f"±{st:.4f}" if st > 0 else "—"
        std_z = f"±{sz:.4f}" if sz > 0 else "—"
        print(f"{name:<35s} {mt:>8.4f} {std_t:>7s} {mz:>8.4f} {std_z:>7s}")

    # M2 verdict
    sgs_zs = np.mean([r['zs'] for r in all_results.get("SGS-2pass", [])])
    gk_zs = np.mean([r['zs'] for r in all_results.get("GaussKernel+Softmax (M2)", [])])
    sgs_test = np.mean([r['test'] for r in all_results.get("SGS-2pass", [])])
    gk_test = np.mean([r['test'] for r in all_results.get("GaussKernel+Softmax (M2)", [])])

    print(f"\n{'='*80}")
    print("M2 VERDICT: Is the advantage from the kernel or the rendering equation?")
    print(f"{'='*80}")
    print(f"  Zero-shot: SGS {sgs_zs:.4f} vs GaussKernel+Softmax {gk_zs:.4f} (Δ={sgs_zs - gk_zs:+.4f})")
    print(f"  Trained:   SGS {sgs_test:.4f} vs GaussKernel+Softmax {gk_test:.4f} (Δ={sgs_test - gk_test:+.4f})")
    if abs(sgs_zs - gk_zs) < 0.01:
        print("  → Zero-shot advantage is from the KERNEL, not the rendering equation")
    elif sgs_zs > gk_zs + 0.02:
        print("  → Rendering equation (transmittance + ordering) adds value beyond the kernel")
    else:
        print("  → Mixed: kernel provides most of the zero-shot advantage")

    # M6 verdict
    hybrid_test = np.mean([r['test'] for r in all_results.get("Hybrid SGS+Softmax (M6)", [])])
    fair_test = np.mean([r['test'] for r in all_results.get("Fair Softmax", [])])

    print(f"\n{'='*80}")
    print("M6 VERDICT: Does the SGS+Softmax hybrid beat both?")
    print(f"{'='*80}")
    print(f"  SGS:           {sgs_test:.4f}")
    print(f"  Hybrid (M6):   {hybrid_test:.4f}")
    print(f"  Fair Softmax:  {fair_test:.4f}")
    if hybrid_test > max(sgs_test, fair_test) + 0.005:
        print("  → HYBRID WINS — best of both worlds!")
    elif hybrid_test > min(sgs_test, fair_test):
        print("  → Hybrid is between SGS and Softmax — not clearly better")
    else:
        print("  → Hybrid doesn't help — combination isn't additive")

    # Save
    with open("paper_fixes_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to paper_fixes_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper fixes: M2, M6")
    parser.add_argument("--glove", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "456"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = ["42"]
        args.epochs = 20
    main(args)
