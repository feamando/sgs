"""
Raum diagnostic test suite.

Runs a series of targeted tests to identify where the bridge pipeline
is breaking down. Reports which components are working vs. failing.

Usage:
    python scripts/diagnose_raum.py ^
      --checkpoint checkpoints/raum_12/best.pt ^
      --glove data/glove.6B.300d.txt ^
      --encoder-checkpoint checkpoints/planck13/best.pt ^
      --blobs-dir data/blobs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Raum diagnostic suite")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--glove", required=True)
    p.add_argument("--encoder-checkpoint", type=str, default=None)
    p.add_argument("--blobs-dir", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    print("=" * 70)
    print("RAUM DIAGNOSTIC SUITE")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Tokenizer coverage
    # ══════════════════════════════════════════════════════════════════
    print("\n[TEST 1] Tokenizer coverage")
    print("-" * 40)

    from src.data import load_glove
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)

    blob_names = []
    if args.blobs_dir:
        with open(Path(args.blobs_dir) / "index.json") as f:
            blob_names = json.load(f)

    if args.encoder_checkpoint:
        from src.raum.encoder import build_sp_word2idx
        tokenizer_path = args.tokenizer
        if not tokenizer_path:
            for c in [Path("data/wikipedia/tokenizer.model"), Path("data/tinystories/tokenizer.model")]:
                if c.exists():
                    tokenizer_path = str(c)
                    break
        if tokenizer_path:
            sp_w2i = build_sp_word2idx(tokenizer_path, extra_words=blob_names)
            print(f"  SP word2idx: {len(sp_w2i)} words")
            # Check blob coverage
            unk_id = sp_w2i.get("<unk>", 0)
            blob_unk = [n for n in blob_names if sp_w2i.get(n, unk_id) == unk_id]
            blob_ok = [n for n in blob_names if sp_w2i.get(n, unk_id) != unk_id]
            print(f"  Blobs mapped to unique token: {len(blob_ok)}/{len(blob_names)}")
            if blob_unk:
                print(f"  Blobs mapped to UNK: {blob_unk[:10]}{'...' if len(blob_unk) > 10 else ''}")

            # Check for token collisions (multiple words -> same token_id)
            id_to_words: dict[int, list[str]] = {}
            for w, tid in sp_w2i.items():
                id_to_words.setdefault(tid, []).append(w)
            collisions = {tid: ws for tid, ws in id_to_words.items() if len(ws) > 1 and tid != unk_id}
            if collisions:
                print(f"  TOKEN COLLISIONS ({len(collisions)} ids shared by multiple words):")
                for tid, ws in list(collisions.items())[:5]:
                    print(f"    token_id={tid}: {ws}")
            else:
                print(f"  No token collisions (good)")

            active_w2i = sp_w2i
        else:
            print("  WARN: no SP tokenizer found")
            active_w2i = word2idx
    else:
        active_w2i = word2idx
        print(f"  GloVe word2idx: {len(active_w2i)} words")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Encoder embedding distinctness
    # ══════════════════════════════════════════════════════════════════
    print("\n[TEST 2] Encoder embedding distinctness")
    print("-" * 40)

    if args.encoder_checkpoint:
        from src.raum.encoder import FrozenPlanckEncoder
        encoder = FrozenPlanckEncoder(args.encoder_checkpoint, device=device)

        # Encode a sample of blob names and measure pairwise cosine similarity
        test_words = blob_names[:50] if blob_names else ["sphere", "cube", "cone", "car", "chair", "table"]
        unk_id = active_w2i.get("<unk>", 0)
        token_ids = torch.tensor([[active_w2i.get(w, unk_id) for w in test_words]])
        mu_s, features = encoder.encode(token_ids)

        # Check if all tokens map to the same embedding (the unk bug)
        feat_np = features[0].numpy()
        unique_rows = np.unique(feat_np, axis=0).shape[0]
        print(f"  {len(test_words)} words -> {unique_rows} unique feature vectors")
        if unique_rows < len(test_words) * 0.5:
            print(f"  ** PROBLEM: many words share embeddings (likely UNK collapse)")
            # Find which are duplicates
            unk_feat = feat_np[0]  # first word's embedding
            same_as_first = sum(1 for i in range(len(test_words)) if np.allclose(feat_np[i], unk_feat, atol=1e-5))
            print(f"  ** {same_as_first}/{len(test_words)} words have SAME embedding as '{test_words[0]}'")
        else:
            print(f"  OK: embeddings are distinct")

        # Cosine similarity matrix (should be moderate, not all 1.0)
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = feat_np / norms
        cos_matrix = normed @ normed.T
        avg_cos = (cos_matrix.sum() - len(test_words)) / (len(test_words) * (len(test_words) - 1))
        print(f"  Avg pairwise cosine: {avg_cos:.3f} (ideal: 0.1-0.5, bad: >0.9)")

        # mu_s distinctness
        mu_np = mu_s[0].numpy()
        mu_unique = np.unique(mu_np, axis=0).shape[0]
        print(f"  mu_s: {mu_unique}/{len(test_words)} unique positions")
    else:
        print("  (skipped, no encoder)")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Bridge checkpoint health
    # ══════════════════════════════════════════════════════════════════
    print("\n[TEST 3] Bridge checkpoint health")
    print("-" * 40)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    d_model = state["pos_emb"].shape[1]
    n_blobs = state["template_head.weight"].shape[0]
    print(f"  d_model={d_model}, n_blobs={n_blobs}")

    # Check template_head weight distribution (if collapsed, all rows similar)
    tpl_weight = state["template_head.weight"].numpy()
    tpl_norms = np.linalg.norm(tpl_weight, axis=1)
    print(f"  template_head weight norms: min={tpl_norms.min():.3f}, max={tpl_norms.max():.3f}, std={tpl_norms.std():.3f}")

    # Check if template_head has collapsed (all rows similar)
    tpl_normed = tpl_weight / np.clip(tpl_norms[:, None], 1e-8, None)
    sample_cos = tpl_normed[:20] @ tpl_normed[:20].T
    off_diag = sample_cos[np.triu_indices(20, k=1)]
    avg_tpl_cos = off_diag.mean()
    print(f"  template_head avg pairwise cosine (first 20 classes): {avg_tpl_cos:.3f}")
    if avg_tpl_cos > 0.8:
        print(f"  ** PROBLEM: template head has COLLAPSED (rows are nearly identical)")
    else:
        print(f"  OK: template head rows are distinct")

    # Check position_head bias (should be near 0 for healthy training)
    pos_bias = state["position_head.2.bias"].numpy()
    print(f"  position_head final bias: {pos_bias}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Training data sanity
    # ══════════════════════════════════════════════════════════════════
    print("\n[TEST 4] Training data sanity (small sample)")
    print("-" * 40)

    from src.raum.data import generate_comp_gen_split, load_blob_object_names, RaumDataset, collate_raum

    scene_objects = load_blob_object_names(args.blobs_dir) if args.blobs_dir else None
    train, _, _ = generate_comp_gen_split(50, 10, 10, n_objects_max=5,
                                          objects=scene_objects, seed=42)

    # Check that generated scenes use real blob names
    obj_names_seen = set()
    for scene in train:
        for obj in scene.objects:
            for name, idx in (scene_objects or {}).items():
                if obj.obj_type == idx:
                    obj_names_seen.add(name)
    print(f"  50 scenes use {len(obj_names_seen)} distinct object classes")
    print(f"  Sample: {list(obj_names_seen)[:10]}")

    # Check token IDs generated by RaumDataset
    ds = RaumDataset(train, active_w2i, max_objects=5)
    sample = ds[0]
    token_ids = sample["token_ids"].tolist()
    unk_id_val = active_w2i.get("<unk>", 0)
    n_unk = token_ids.count(unk_id_val)
    print(f"  Sample scene: '{train[0].sentence}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  UNK tokens: {n_unk}/{len(token_ids)}")
    if n_unk > len(token_ids) * 0.3:
        print(f"  ** PROBLEM: >30% tokens are UNK. Blob names not in word2idx!")

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Forward pass sanity
    # ══════════════════════════════════════════════════════════════════
    print("\n[TEST 5] Forward pass with known-good input")
    print("-" * 40)

    from src.raum.bridge import RaumBridge

    n_layers = 0
    while f"encoder.layers.{n_layers}.self_attn.in_proj_weight" in state:
        n_layers += 1
    for nh in [8, 6, 4, 2, 1]:
        if d_model % nh == 0:
            n_heads = nh
            break
    has_relation = "relation_head.0.weight" in state
    d_in = state["input_proj.weight"].shape[1]

    if args.encoder_checkpoint:
        d_s = encoder.d_s
        d_f = encoder.d_f
    else:
        d_s = d_in - 300
        d_f = 300

    model = RaumBridge(d_s=d_s, d_f=d_f, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, n_blobs=n_blobs,
                       with_relation_head=has_relation)
    model.load_state_dict(state)
    model.eval()

    # Feed a blob-name sentence through
    if blob_names and args.encoder_checkpoint:
        test_sent = f"a red {blob_names[0]} above a blue {blob_names[1]}"
        test_words = test_sent.lower().split()
        ids = [active_w2i.get(w, unk_id_val) for w in test_words]
        token_ids_t = torch.tensor([ids])
        mask = torch.ones_like(token_ids_t, dtype=torch.float32)
        mu_s, features = encoder.encode(token_ids_t)
        out = model(mu_s, features, mask)

        positions = out["positions"][0].detach().numpy()
        tpl_preds = out["template_logits"][0].detach().argmax(dim=-1).tolist()
        print(f"  Input: '{test_sent}'")
        print(f"  Token IDs: {ids}")
        for i, w in enumerate(test_words):
            pos = positions[i]
            pred_id = tpl_preds[i]
            pred_name = blob_names[pred_id] if pred_id < len(blob_names) else f"id{pred_id}"
            pos_ok = all(abs(p) < 5 for p in pos)
            print(f"    {w:12s} -> pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                  f"blob={pred_name:12s} {'OK' if pos_ok else '** BAD POS'}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Check the output above for ** PROBLEM markers. Common failure modes:

1. UNK COLLAPSE: blob names not in word2idx -> all tokens get same
   embedding -> bridge can't discriminate objects. Fix: ensure
   build_sp_word2idx gets blob names.

2. TEMPLATE HEAD COLLAPSE: all 300 rows converge to the same vector.
   Caused by (1) or by learning rate too high / too few epochs.

3. POSITION EXPLOSION: color/relation words at pos=(+90, +4, +200).
   Caused by the model learning to "push away" non-object tokens
   to reduce the position loss on object tokens only.

4. LOW EMBEDDING DISTINCTNESS: Planck embeddings for common nouns
   are too similar (all are "Wikipedia article" context vectors).
   Fix: use GloVe features for object discrimination instead of
   or alongside Planck embeddings.
""")


if __name__ == "__main__":
    main()
