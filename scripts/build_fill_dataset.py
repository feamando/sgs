"""
Raum Path A: build the learned-FILL training dataset.

The fill stage today is hand-built grammar: expand_part(name) -> a compound
CompositionNode of stones. Path A replaces it with a LEARNED conditional
generator (train_fill.py): part-token + pose -> N Gaussians. This script makes
its training data the cheap way -- the SAME 0.5->1.5 trick, the grammar IS the
data generator.

For each known part kind (tower/wall/keep/gate/...), we expand it via
expand_part at varied courses + stone colours + seeds, flatten to renderer
tensors, and record one example:

    {
      "part": "tower",                 # the part-token the fill model conditions on
      "params": {"courses": 7, "color": [r,g,b], "scale": 1.1},  # pose/style
      "gaussians": {                   # the TARGET cloud (local frame)
         "means": [[x,y,z],...],       # [N,3]
         "scales_log": [[sx,sy,sz],...],
         "rotations": [[w,x,y,z],...],
         "opacities": [o,...],
         "colors": [[r,g,b],...]
      }
    }

The fill model learns to reproduce these clouds, then (train_fill.py) is
fine-tuned with render-score (SDS) supervision so parts LOOK right, not just
match the template. Local frame only -- the decomposer's tree supplies the
world pose; fill only needs to generate the part's own geometry.

Usage:
  python scripts/build_fill_dataset.py --out data/fill/path1_fill.json
  python scripts/build_fill_dataset.py --out - --stats        # dry-run, counts only
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import tree_to_tensors
from scripts.castle_grammar import expand_part


# Part kinds the grammar can expand, with a representative name per kind so
# expand_part's name-based routing (is_gate via _s suffix, square via "square")
# triggers the variants we want. We sample courses + colour + a few name
# variants to get within-kind diversity.
PART_SPECS = [
    ("tower",      ["tower", "tower_ne", "square_tower"]),
    ("wall",       ["wall", "wall_n", "wall_s"]),          # _s -> gate variant
    ("keep",       ["keep"]),
    ("gatehouse",  ["gatehouse"]),
    ("gate",       ["gate"]),
    ("tree",       ["tree", "tree_3"]),
    ("door",       ["door"]),
    ("window",     ["window", "arch_window"]),
    ("arrow_slit", ["arrow_slit"]),
    ("arch",       ["arch"]),
    ("cliff",      ["cliff"]),
    ("rock",       ["rock"]),
]


def _rand_stone(rng):
    return [rng.uniform(0.55, 0.70), rng.uniform(0.52, 0.63), rng.uniform(0.48, 0.60)]


def make_examples(per_variant, rng):
    examples = []
    skipped = []
    for kind, names in PART_SPECS:
        for name in names:
            for _ in range(per_variant):
                courses = rng.randint(4, 9)
                color = _rand_stone(rng)
                node = expand_part(name, color=color, courses=courses, rng=rng)
                if node is None:
                    skipped.append(name)
                    continue
                t = tree_to_tensors(node)
                n = t["means"].shape[0]
                if n == 0:
                    skipped.append(name)
                    continue
                examples.append({
                    "part": kind,
                    "name": name,
                    "params": {"courses": courses, "color": [round(c, 4) for c in color]},
                    "n_gaussians": n,
                    "gaussians": {
                        "means": [[round(x, 4) for x in p] for p in t["means"].tolist()],
                        "scales_log": [[round(x, 4) for x in s] for s in t["scales_log"].tolist()],
                        "rotations": [[round(x, 4) for x in r] for r in t["rotations"].tolist()],
                        "opacities": [round(o, 4) for o in t["opacities"].tolist()],
                        "colors": [[round(x, 4) for x in c] for c in t["colors"].tolist()],
                    },
                })
    return examples, skipped


def main():
    p = argparse.ArgumentParser(description="Build learned-fill (part+pose -> gaussians) dataset")
    p.add_argument("--out", default="data/fill/path1_fill.json", help="'-' = stats only")
    p.add_argument("--per-variant", type=int, default=40,
                   help="examples per (kind,name) variant -> dataset size")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    examples, skipped = make_examples(args.per_variant, rng)

    import collections
    by_kind = collections.Counter(e["part"] for e in examples)
    sizes = [e["n_gaussians"] for e in examples]
    print(f"[fill-data] {len(examples)} examples across {len(by_kind)} part kinds")
    for k, c in sorted(by_kind.items()):
        print(f"   {k:12s} {c:4d} examples")
    if sizes:
        print(f"   gaussians/example: min {min(sizes)} max {max(sizes)} "
              f"mean {sum(sizes)//len(sizes)}")
    if skipped:
        print(f"   WARN skipped (expand_part returned empty): {set(skipped)}")

    if args.out == "-":
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(examples, open(args.out, "w"))
    print(f"[fill-data] saved -> {args.out}")
    print(f"[fill-data] next: train_fill.py --data {args.out}")


if __name__ == "__main__":
    main()
