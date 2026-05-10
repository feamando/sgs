"""
Post-training analysis for Raum bridge (1.0 and 1.1).

Reports:
  - Comp-gen test metrics (held-out object pairs)
  - Per-sample sentence probes showing predicted position, template,
    colour, scale, role for each token

Usage (1.0):
    python scripts/analyze_raum_bridge.py ^
        --checkpoint checkpoints/raum_10/best.pt ^
        --glove data/glove.6B.300d.txt

Usage (1.1):
    python scripts/analyze_raum_bridge.py ^
        --checkpoint checkpoints/raum_11_n3/best.pt ^
        --glove data/glove.6B.300d.txt ^
        --encoder-checkpoint checkpoints/planck13/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

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


def _infer_bridge_config(state: dict) -> dict:
    """Infer RaumBridge constructor args from a state_dict."""
    d_model = state["pos_emb"].shape[1]
    max_len = state["pos_emb"].shape[0]
    n_blobs = state["template_head.weight"].shape[0]

    # input_proj.weight shape is [d_model, d_s + d_f]
    d_in = state["input_proj.weight"].shape[1]

    # Count encoder layers
    n_layers = 0
    while f"encoder.layers.{n_layers}.self_attn.in_proj_weight" in state:
        n_layers += 1

    # n_heads: in_proj_weight is [3 * d_model, d_model] for MHA,
    # but we can infer n_heads from the out_proj or from the d_model
    # Best heuristic: d_model // head_dim where head_dim is typically 32 or 64
    # Actually we can just check if n_heads divides d_model evenly
    # Default: try common values
    for nh in [8, 6, 4, 2, 1]:
        if d_model % nh == 0:
            n_heads = nh
            break

    return {
        "d_model": d_model,
        "max_len": max_len,
        "n_blobs": n_blobs,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_in": d_in,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Raum bridge (1.0 / 1.1)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--glove", required=True)
    p.add_argument("--encoder-checkpoint", type=str, default=None,
                   help="Planck checkpoint (1.1 mode)")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="SP tokenizer model (auto-detected)")
    p.add_argument("--blobs-dir", type=str, default=None,
                   help="Path to blob library (for word2idx + scene objects)")
    p.add_argument("--n-objects-max", type=int, default=2)
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--save-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Default save-dir based on checkpoint location
    if args.save_dir is None:
        ckpt_dir = Path(args.checkpoint).parent
        args.save_dir = str(ckpt_dir.parent / "results" / (ckpt_dir.name + "_analysis"))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint and infer architecture ──
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = _infer_bridge_config(state)
    print(f"Inferred bridge config: d_model={config['d_model']}, "
          f"n_layers={config['n_layers']}, n_heads={config['n_heads']}, "
          f"n_blobs={config['n_blobs']}, d_in={config['d_in']}")

    # ── Load GloVe ──
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)

    # ── Infer d_s, d_f from checkpoint ──
    # input_proj.weight shape is [d_model, d_s + d_f]
    d_in = config["d_in"]

    # ── Setup encoder/vocab ──
    use_encoder = args.encoder_checkpoint is not None

    # If checkpoint was trained with encoder dims (d_in > 364), auto-enable encoder
    if d_in > 400 and not use_encoder:
        print(f"  NOTE: checkpoint input dim={d_in} suggests encoder mode.")
        print(f"  Pass --encoder-checkpoint to use the Planck encoder for analysis.")
        print(f"  Falling back to zero-padded GloVe (metrics will be meaningless).")

    if use_encoder:
        from src.raum.encoder import FrozenPlanckEncoder, build_sp_word2idx
        encoder = FrozenPlanckEncoder(args.encoder_checkpoint, device=device)
        encoder.to(device).eval()
        d_s = encoder.d_s
        d_f = encoder.d_f

        tokenizer_path = args.tokenizer
        if tokenizer_path is None:
            candidates = [
                Path("data/wikipedia/tokenizer.model"),
                Path("data/tinystories/tokenizer.model"),
            ]
            for c in candidates:
                if c.exists():
                    tokenizer_path = str(c)
                    break
        if tokenizer_path and Path(tokenizer_path).exists():
            extra = None
            if args.blobs_dir:
                import json as _json
                _idx_path = Path(args.blobs_dir) / "index.json"
                if _idx_path.exists():
                    with open(_idx_path) as _f:
                        extra = _json.load(_f)
            word2idx = build_sp_word2idx(tokenizer_path, extra_words=extra)
            print(f"Using SP tokenizer: {tokenizer_path} ({len(word2idx)} words)")

        vocab = None

        def get_features(token_ids):
            return encoder.encode(token_ids)
    else:
        # Infer d_s and d_f: for GloVe path d_f=300 always, d_s = d_in - 300
        d_f_glove = vectors.shape[1]  # 300
        d_s = d_in - d_f_glove
        d_f = d_f_glove

        if d_s <= 0 or d_s > 512:
            # Checkpoint was likely trained with encoder; can't analyze without it
            print(f"  ERROR: cannot infer valid d_s from d_in={d_in} - d_f={d_f_glove}")
            print(f"  This checkpoint needs --encoder-checkpoint.")
            sys.exit(1)

        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab.to(device).eval()

        def get_features(token_ids):
            mu_s, _, _, features = vocab.get_params(token_ids)
            return mu_s, features

    # ── Create model matching checkpoint ──
    has_relation = "relation_head.0.weight" in state
    model = RaumBridge(
        d_s=d_s, d_f=d_f,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_blobs=config["n_blobs"],
        with_relation_head=has_relation,
        K=args.K,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded bridge: {model.count_parameters():,} params")

    # ── Comp-gen evaluation ──
    print("\n" + "=" * 60)
    print("COMP-GEN TEST (held-out object pairs)")
    print("=" * 60)
    _, _, test_scenes = generate_comp_gen_split(
        n_train=args.n_test, n_val=args.n_test, n_test=args.n_test,
        n_objects_max=args.n_objects_max, seed=42,
    )
    max_objects = args.n_objects_max
    test_ds = RaumDataset(test_scenes, word2idx, max_objects=max_objects)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raum,
    )

    if use_encoder:
        test_metrics = evaluate_routing(model, get_features, test_loader, device,
                                        use_encoder=True)
    else:
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
    if args.n_objects_max >= 3:
        test_sentences += [
            "a red sphere above a blue cube and a green cone left the cube",
            "a yellow cylinder below a white torus and a purple plane right the torus",
        ]

    probes = []
    for s in test_sentences:
        if use_encoder:
            p = probe_sentence(model, get_features, word2idx, s, use_encoder=True)
        else:
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
