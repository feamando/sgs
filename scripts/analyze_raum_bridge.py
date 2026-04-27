"""
Post-training analysis for Raum 1.0 routing bridge.

Reports:
  - Comp-gen test metrics (held-out object pairs)
  - Per-sample sentence probes showing predicted position, template,
    colour, scale, role for each token

Usage:
    python scripts/analyze_raum_bridge.py \
        --checkpoint checkpoints/raum_10/best.pt \
        --glove data/glove.6B.300d.txt
"""

import argparse
import sys
from pathlib import Path

import torch

# Windows consoles default to cp1252 and choke on non-ASCII glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_glove
from src.gaussian import SemanticGaussianVocab
from src.raum.bridge import RaumBridge
from src.raum.data import generate_comp_gen_split, RaumDataset, collate_raum
from src.raum.analyze import (
    probe_sentence, evaluate_routing,
    print_sentence_probe, print_eval,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Raum routing bridge")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--glove", required=True)
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--save-dir", default="results/raum_10_analysis")
    return p.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    d_f = vectors.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = SemanticGaussianVocab(len(words), d_s=args.d_s, d_f=d_f)
    vocab.init_from_glove(vectors, freqs)
    vocab.to(device).eval()

    model = RaumBridge(
        d_s=args.d_s, d_f=d_f,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        K=args.K,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── Comp-gen evaluation ──
    print("\n" + "=" * 60)
    print("COMP-GEN TEST (held-out object pairs)")
    print("=" * 60)
    _, _, test_scenes = generate_comp_gen_split(
        n_train=args.n_test, n_val=args.n_test, n_test=args.n_test, seed=42,
    )
    test_ds = RaumDataset(test_scenes, word2idx)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raum,
    )
    test_metrics = evaluate_routing(model, vocab, test_loader, device)
    print_eval(test_metrics, "test set (unseen pairs)")

    # ── Sentence probes ──
    print("\n" + "=" * 60)
    print("SENTENCE PROBES")
    print("=" * 60)
    test_sentences = [
        "a red sphere above a blue cube",
        "a green cone below a yellow cylinder",
        "a white torus left a purple sphere",
        "a huge red cube on a small blue plane",
        "a tiny orange sphere behind a large black cube",
        "a red sphere",
        "a blue cube",
    ]
    probes = []
    for s in test_sentences:
        p = probe_sentence(model, vocab, word2idx, s)
        probes.append(p)
        print_sentence_probe(p)

    # Save text transcript.
    out_path = save_dir / "report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        print_eval(test_metrics, "test set (unseen pairs)", file=f)
        for p in probes:
            print_sentence_probe(p, file=f)
    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
