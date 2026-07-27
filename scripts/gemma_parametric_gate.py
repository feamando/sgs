"""
Phase 1 gate (SETUP_072026_gemma.md §7): can Gemma compose recognizable objects
from PARAMETRIC PRIMITIVES instead of naming the 14 fixed grammar parts?

The bet: the bottleneck was never the model or the layout task -- it's the FILL
(hand-authored geometry per named part, path1 "fill richness caps usable data
richness"). If Gemma emits shape+size+color primitives, the vocabulary ceiling
(14 castle parts) AND the geometry ceiling (one hard-coded shape per name) both
lift: a ship/pagoda/lighthouse become expressible from box/cylinder/cone/... with
NO new hand-authored parts and NO learned fill (path1's FillModel blobbed).

Gate = NON-CASTLE prompts (the whole point is breadth). Reports JSON-valid rate,
primitive count, and SHAPE DIVERSITY per prompt (a ship that's all boxes is a
fail; a ship with hull+mast+sail using box+cylinder+plane is the win). Dumps each
filled tree so you can render it model-free:
  python scripts/infer_decomposer.py --scene-file <dump>.json --no-snap --serve

Usage (4090 box):
  python scripts/gemma_parametric_gate.py --model models/gemma-4-e4b-it \
    --prompts scripts/assets/parametric_prompts.txt \
    --out results/gemma_parametric.json --dump-dir output/parametric
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_trees_gemma import parse_tree
from scripts.gemma_decomposer import validate_parametric


def _shapes(tree):
    return [str(c.get("shape", "")).lower() for c in (tree or {}).get("children", [])]


def main():
    p = argparse.ArgumentParser(description="Phase-1 gate: parametric-primitive composition")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True, help="one non-castle object per line")
    p.add_argument("--out", default="results/gemma_parametric.json")
    p.add_argument("--dump-dir", default=None, help="write each filled tree here for --scene-file")
    p.add_argument("--max-new", type=int, default=1400)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    from scripts.gemma_decomposer import GemmaMMDecomposer
    dec = GemmaMMDecomposer(args.model, n_shot=0, max_new=args.max_new,
                            temperature=args.temperature)

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    print(f"[param-gate] {len(prompts)} prompts")

    rows, n_valid, tot_prims = [], 0, 0
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        tree = dec.generate_parametric(prompt=prompt)
        # measure the RAW emitted tree too (pre-fill) for shape diversity
        raw = parse_tree(dec.last_raw)
        clean, dropped = validate_parametric(raw) if raw else (None, 0)
        shapes = _shapes(clean)
        n_prim = len(shapes)
        n_distinct = len(set(shapes))
        ok = tree is not None and n_prim >= 2 and n_distinct >= 2
        n_valid += int(ok)
        tot_prims += n_prim
        rows.append({"prompt": prompt, "ok": ok, "n_primitives": n_prim,
                     "n_distinct_shapes": n_distinct, "shapes": shapes,
                     "dropped": dropped, "tree": tree})
        flag = "ok  " if ok else "WEAK"
        print(f"  [{i+1}/{len(prompts)}] {flag} prims={n_prim} distinct={n_distinct} "
              f"{sorted(set(shapes))}  {prompt[:40]}")
        if args.dump_dir and tree is not None:
            slug = "".join(ch if ch.isalnum() else "_" for ch in prompt)[:40]
            json.dump(tree, open(Path(args.dump_dir) / f"{i:02d}_{slug}.json", "w"), indent=2)

    n = len(prompts)
    summary = {"model": args.model, "n_prompts": n, "n_ok": n_valid,
               "ok_rate": round(n_valid / n, 3) if n else 0.0,
               "mean_primitives": round(tot_prims / n, 1) if n else 0.0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)

    print("\n[param-gate] PHASE-1 GATE ==================================")
    print(f"  ok rate (>=2 prims, >=2 distinct shapes): {summary['ok_rate']:.1%} ({n_valid}/{n})")
    print(f"  mean primitives/object: {summary['mean_primitives']}")
    print(f"  saved -> {args.out}" + (f"; trees -> {args.dump_dir}" if args.dump_dir else ""))
    print("  PASS = objects render RECOGNIZABLE (eyeball the dumps via --scene-file).")
    print("  ok-rate is necessary not sufficient: a valid all-box tree is WEAK. Look.")
    print("  FAIL -> parametric composition incoherent; fall back to vocab-expansion.")


if __name__ == "__main__":
    main()
