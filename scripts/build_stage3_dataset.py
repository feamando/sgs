"""
Raum 1.7 Stage 3: build the decomposer fine-tune dataset from Stage-2 layouts.

Stage 3 makes the model EMIT render-good proportions directly, so inference
needs no per-scene optimization and the snap_layout() scaffolding can come off.
The decomposer (train_decomposer.py) learns (prompt -> tree_json) pairs. Today
those trees carry the GRAMMAR's hand-tuned proportions; this swaps in the
Stage-2 OPTIMIZED proportions (from optimize_layout.py / layout_opt.params.json),
which are non-circular (render-scored, not grammar constants) -- exactly the
"D needs a non-circular signal" the roadmap calls for.

Key efficiency point: the castle prompt set is all PARAPHRASES of one scene
("a castle on a hill" == "a fortress atop a hill" == ...). So we do NOT run the
expensive SDS search per prompt -- we run it ONCE for the canonical castle and
pair the single optimized tree with every paraphrase. N searches -> 1.

The emitted tree is SHALLOW (skeleton: named parts with position+scale, no
stones) -- that is what the model is trained to produce and what
_fill_gaussians expands at inference. We strip the gaussians and keep only the
structure + transforms, so the target teaches WHERE/HOW-BIG, not 3000 stones.

Usage:
  # uses output/layout_opt.params.json (the Stage-2 result) by default:
  python scripts/build_stage3_dataset.py --out data/decomposition_trees/stage3_train.json
  # or pass explicit params:
  python scripts/build_stage3_dataset.py --params output/layout_opt.params.json --out ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.optimize_layout import params_to_tree, PARAM_NAMES, initial_params


# the castle paraphrases (mirror scripts/castle_grammar.py PROMPTS["castle_on_hill"])
CASTLE_PARAPHRASES = [
    "a castle on a hill",
    "a fortress atop a hill",
    "a hilltop castle",
    "a castle on a hill with trees",
    "a stone castle on a green hill",
    "a medieval fortress on a hill surrounded by trees",
]


def shallow_skeleton(tree):
    """Emit the SHALLOW skeleton in the exact 1.6 training format (castle_16):
    named PARTS as empty-leaf nodes (name + position + scale, NO children, NO
    gaussians); _fill_gaussians expands each part at inference. Recursion stops
    at the part level -- we do NOT descend into a part's stone sub-structure
    (tower_body/roof/crenellation), which the 1.6 model never emits.

    Depth contract (matches data/decomposition_trees/castle_16):
      scene
        hill                 (empty leaf)
        castle  s=,pos=       -> children: tower_0..3, wall_0..3, keep (empty leaves)
        tree_i                (empty leaf)
    """
    PART_PREFIXES = ("tower", "wall", "keep", "gate", "hill", "tree", "ground",
                     "arrow_slit", "window", "door", "arch", "cliff", "rock")

    def is_part(name):
        n = name.lower()
        return any(n == pfx or n.startswith(pfx + "_") for pfx in PART_PREFIXES)

    def emit(node, stop_at_part):
        out = {"name": node.name, "position": [round(x, 3) for x in node.position],
               "scale": round(node.scale, 3)}
        if getattr(node, "rotation", None) and node.rotation != [1.0, 0, 0, 0]:
            out["rotation"] = [round(x, 4) for x in node.rotation]
        # a recognized PART is a leaf in the skeleton -- do not descend into its
        # stones. The container ("castle") and root ("scene") keep their children.
        if stop_at_part and is_part(node.name):
            return out
        kids = [emit(c, stop_at_part=True) for c in node.children]
        if kids:
            out["children"] = kids
        return out

    return emit(tree, stop_at_part=False)


def load_params(path):
    if path and Path(path).exists():
        d = json.load(open(path))
        return np.array([d[k] for k in PARAM_NAMES])
    print(f"[stage3] no params at {path}, using grammar defaults", file=sys.stderr)
    return initial_params()


def main():
    p = argparse.ArgumentParser(description="Build Stage 3 decomposer dataset from Stage-2 layout")
    p.add_argument("--params", default="output/layout_opt.params.json",
                   help="Stage-2 optimized layout params")
    p.add_argument("--out", default="data/decomposition_trees/stage3_train.json")
    p.add_argument("--repeat", type=int, default=8,
                   help="duplicate each paraphrase N times (small dataset augmentation)")
    p.add_argument("--mix", default="data/decomposition_trees/castle_16/train.json",
                   help="blend NON-castle records from this 1.6 dataset to prevent "
                        "catastrophic forgetting (empty string to disable)")
    p.add_argument("--mix-n", type=int, default=400,
                   help="how many non-castle records to mix in")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    params = load_params(args.params)
    print(f"[stage3] layout params: {dict(zip(PARAM_NAMES, params.round(3).tolist()))}")

    tree = params_to_tree(params)
    skeleton = shallow_skeleton(tree)

    records = []
    for prompt in CASTLE_PARAPHRASES:
        for _ in range(args.repeat):
            records.append({"prompt": prompt, "tree": skeleton})
    n_castle = len(records)
    print(f"[stage3] {len(CASTLE_PARAPHRASES)} paraphrases x{args.repeat} = {n_castle} castle records")
    print(f"[stage3] ALL map to the ONE Stage-2 optimized layout (paraphrases of one scene)")

    # Mix in NON-castle records from the 1.6 dataset so the fine-tune SHIFTS
    # castle proportions without catastrophically forgetting the other scenes
    # (walls/arches/gates/lighthouses/...) the 1.6 model already knows. We keep
    # ONLY non-castle prompts so we don't reintroduce the OLD castle proportions
    # that would fight the new Stage-2 targets.
    if args.mix and Path(args.mix).exists():
        import random
        rng = random.Random(args.seed)
        pool = json.load(open(args.mix))
        non_castle = [r for r in pool
                      if "castle" not in r["prompt"].lower()
                      and "fortress" not in r["prompt"].lower()]
        rng.shuffle(non_castle)
        mix = non_castle[:args.mix_n]
        records.extend(mix)
        print(f"[stage3] mixed in {len(mix)} non-castle records from {args.mix} "
              f"(anti-forgetting); total {len(records)}")
    elif args.mix:
        print(f"[stage3] WARN: --mix path {args.mix} not found; NO anti-forgetting "
              f"mix. Fine-tuning on castle-only risks collapsing all prompts to "
              f"castles.", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print(f"[stage3] saved -> {args.out} ({n_castle} castle + {len(records)-n_castle} mix)")
    print(f"[stage3] next: fine-tune with train_decomposer.py --data {args.out}")


if __name__ == "__main__":
    main()
