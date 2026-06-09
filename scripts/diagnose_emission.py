"""
Raum 1.7 Stage 3 diagnostic: recall vs capacity vs positioning.

The snap-off renders (images 96/97) show a hill + one tower-ish cluster, not the
full castle. But a RENDER conflates three very different failures:
  (a) RECALL     -- the model emits FEWER parts than it should (1 tower, no walls)
  (b) POSITIONING-- it emits ALL parts but at bad coords, so they hide/overlap
  (c) STRUCTURE  -- it emits parts but malformed/unparseable

This reads the RAW emitted tree (BEFORE snap_layout) and counts what the model
actually output per prompt, so we know which failure we're fighting before
deciding 1.7's fate. No rendering, no snapping.

Usage (4090, .venv-sds):
  python scripts/diagnose_emission.py `
    --checkpoint checkpoints/planck_decomposer_stage3/best.pt `
    --tokenizer data/wikipedia/tokenizer.model
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PROMPTS = [
    # castle paraphrases (should emit 4 towers + 4 walls + keep)
    "a castle on a hill",
    "a stone castle on a green hill",
    "a medieval fortress on a hill surrounded by trees",
    # non-castle (catastrophic-forgetting check -- should still emit their parts)
    "a tower with a window",
    "a wall with an arrow slit",
    "a stone gatehouse with a gate",
]

# what a full castle should contain
CASTLE_EXPECT = {"tower": 4, "wall": 4, "keep": 1}


def count_parts(tree):
    """Count emitted parts by kind from the RAW tree (pre-snap)."""
    from scripts.castle_grammar import _part_kind
    counts = collections.Counter()
    names = []

    def walk(node):
        nm = node.get("name", "")
        k = _part_kind(nm)
        if k:
            counts[k] += 1
            names.append(nm)
        for c in node.get("children", []) or []:
            walk(c)

    walk(tree)
    return counts, names


def main():
    p = argparse.ArgumentParser(description="Diagnose Stage 3 emission: recall vs capacity")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts", nargs="*", default=None, help="override the prompt set")
    p.add_argument("--out", default="output/emission_diag.json")
    args = p.parse_args()

    import torch
    from scripts.infer_decomposer import Decomposer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dec = Decomposer(args.checkpoint, args.tokenizer, device)

    prompts = args.prompts or PROMPTS
    results = []
    print(f"\n{'prompt':<48} {'parts emitted':<32} verdict")
    print("-" * 100)
    for prompt in prompts:
        tree = dec.generate_tree(prompt)
        if tree is None:
            print(f"{prompt:<48} {'<PARSE FAILED>':<32} STRUCTURE failure")
            results.append({"prompt": prompt, "parsed": False})
            continue
        counts, names = count_parts(tree)
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "(none)"

        is_castle = "castle" in prompt.lower() or "fortress" in prompt.lower()
        if is_castle:
            t, w, k = counts.get("tower", 0), counts.get("wall", 0), counts.get("keep", 0)
            total = t + w + k
            if total <= 1:
                verdict = "RECALL fail (1 part)"
            elif t >= 3 and w >= 2 and k >= 1:
                verdict = "FULL layout (-> positioning?)"
            else:
                verdict = f"PARTIAL ({total}/9 parts)"
        else:
            verdict = "emitted parts" if counts else "RECALL fail (none)"

        print(f"{prompt:<48} {summary:<32} {verdict}")
        results.append({"prompt": prompt, "parsed": True,
                        "counts": dict(counts), "names": names, "verdict": verdict})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nREAD: 'RECALL fail' across castle prompts = model can't emit many parts")
    print("      (capacity wall -> bigger base or hybrid). 'FULL layout' = parts")
    print("      are there but mispositioned (fixable -> Stage 2-style placement).")


if __name__ == "__main__":
    main()
