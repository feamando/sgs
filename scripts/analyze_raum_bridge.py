"""
Post-training analysis for Raum PoC-C: what did the space transform learn?

Usage:
    python scripts/analyze_raum_bridge.py --checkpoint checkpoints/raum_c/best.pt --glove data/glove.6B.300d.txt
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_glove
from src.gaussian import SemanticGaussianVocab
from src.raum.bridge import RaumBridge
from src.raum.analyze import (
    analyze_mu_proj, print_word_mapping,
    analyze_sentence_positions, interpolation_analysis,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Raum bridge space transform")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--glove", required=True)
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--save-dir", default="results/raum_c_analysis")
    return p.parse_args()


def main():
    args = parse_args()

    # Load GloVe
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    d_f = vectors.shape[1]

    # Build vocab
    vocab = SemanticGaussianVocab(len(words), d_s=args.d_s, d_f=d_f)
    vocab.init_from_glove(vectors, freqs)
    vocab.eval()

    # Load model
    model = RaumBridge(d_s=args.d_s, d_f=d_f, K=args.K)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Analysis 1: Word → xyz mapping ──
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Individual word → xyz mapping")
    print("=" * 60)
    results = analyze_mu_proj(model, word2idx, vocab.mu)
    print_word_mapping(results)

    # Save raw data
    with open(save_dir / "word_mapping.txt", "w") as f:
        for group, coords in results.items():
            f.write(f"\n{group.upper()}:\n")
            for word, xyz in coords.items():
                f.write(f"  {word:>10s} → {xyz[0]:+.4f} {xyz[1]:+.4f} {xyz[2]:+.4f}\n")

    # ── Analysis 2: Sentence composition ──
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Sentence word positions in 3D")
    print("=" * 60)
    test_sentences = [
        "a red sphere above a blue cube",
        "a green cone below a yellow cylinder",
        "a white torus left a purple sphere",
        "a large red cube on a small blue plane",
    ]
    sent_results = analyze_sentence_positions(model, word2idx, vocab.mu, test_sentences)
    for sr in sent_results:
        print(f"\n  \"{sr['sentence']}\"")
        for w, xyz in sr["word_positions"].items():
            print(f"    {w:>10s} → x={xyz[0]:+.3f} y={xyz[1]:+.3f} z={xyz[2]:+.3f}")

    # ── Analysis 3: Interpolation ──
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Interpolation in projected space")
    print("=" * 60)
    pairs = [("sphere", "cube"), ("red", "blue"), ("above", "below"), ("left", "right")]
    for w_a, w_b in pairs:
        path = interpolation_analysis(model, word2idx, vocab.mu, vocab.features, w_a, w_b, n_steps=5)
        if path:
            print(f"\n  {w_a} → {w_b}:")
            for t, xyz in path:
                print(f"    t={t:.1f} → x={xyz[0]:+.3f} y={xyz[1]:+.3f} z={xyz[2]:+.3f}")

    # ── Analysis 4: mu_proj weight matrix (if linear) ──
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Space transform weight norms")
    print("=" * 60)
    for name, param in model.mu_proj.named_parameters():
        if 'weight' in name:
            print(f"  {name}: shape={list(param.shape)}, norm={param.norm():.3f}")
            if param.shape[0] == 3:
                # This is the final projection d→3
                # Check which input dimensions contribute most to each xyz axis
                for axis, axis_name in enumerate(["x", "y", "z"]):
                    top_dims = param[axis].abs().topk(5)
                    dims_str = ", ".join(f"d{i}({v:.3f})" for i, v in zip(top_dims.indices.tolist(), top_dims.values.tolist()))
                    print(f"    {axis_name}-axis top input dims: {dims_str}")

    print(f"\nResults saved to {save_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
