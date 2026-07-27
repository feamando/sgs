"""
Gate 2: quantify the DECOMPOSER difference between two backends.

The render can't show a model difference: every splat is grammar (fill_gaussians
/ expand_part), so two backends that emit the same part LIST render identically.
That's why Gemma looks like Hertz on a castle -- castle decomposition is a
saturated ~14-part task (project_sgs_path1_outcome), so the part lists converge.

This compares the actual model OUTPUT: the structure-only tree (part names +
positions + scales), NOT the rendered gaussians. Runs the SAME prompts through
both backends and reports, per prompt:
  - part-name set overlap (Jaccard) -- do they pick the same parts?
  - pose delta (mean L2 on position, |dscale|) for parts both emit
  - leaf count each
Aggregated, plus full raw trees dumped for inspection.

The interesting signal is NOT the castle (both saturate) but BREADTH prompts
(lighthouse, pagoda, windmill, bridge...) where a big pretrained base should
diverge from a narrowly-fine-tuned custom model. Divergence there is the payoff
of bringing in Gemma; convergence on castles is the expected floor.

Usage (4090 box, both checkpoints present):
  python scripts/compare_decomposers.py `
    --gemma models/gemma-4-e4b-it `
    --hertz checkpoints/hertz_decomposer/best.pt `
    --hertz-tokenizer data/hertz12_data/tokenizer.model `
    --prompts scripts/assets/breadth_prompts.txt `
    --out results/decomposer_compare.json

  # Gemma-only (no hertz checkpoint): just dumps Gemma trees per prompt.
  python scripts/compare_decomposers.py --gemma models/gemma-4-e4b-it `
    --prompts scripts/assets/breadth_prompts.txt --out results/gemma_trees.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def skeleton(tree):
    """Reduce a (possibly filled) tree to its top-level part signature:
    [(name, [x,y,z] rounded, scale rounded)]. fill_gaussians preserves each
    top-level node's name/position/scale, so this is the pure model output even
    if gaussians were added underneath."""
    if not isinstance(tree, dict):
        return []
    out = []
    for c in tree.get("children", []) or []:
        if not isinstance(c, dict) or "name" not in c:
            continue
        pos = c.get("position", [0, 0, 0])
        pos = [round(float(p), 3) for p in (pos + [0, 0, 0])[:3]]
        out.append((c["name"].lower(), pos, round(float(c.get("scale", 1.0)), 3)))
    return out


def compare(sk_a, sk_b):
    """Two skeletons -> overlap + pose deltas. Matches parts by name (first
    unmatched occurrence), so repeated names (tower_0..3) pair up in order."""
    names_a = [p[0] for p in sk_a]
    names_b = [p[0] for p in sk_b]
    set_a, set_b = set(names_a), set(names_b)
    inter = set_a & set_b
    union = set_a | set_b
    jaccard = len(inter) / len(union) if union else 1.0

    # pose delta on parts present in both, matched positionally by name
    b_by_name = {}
    for p in sk_b:
        b_by_name.setdefault(p[0], []).append(p)
    pos_deltas, scale_deltas, matched = [], [], 0
    for p in sk_a:
        bucket = b_by_name.get(p[0])
        if bucket:
            q = bucket.pop(0)
            pos_deltas.append(math.dist(p[1], q[1]))
            scale_deltas.append(abs(p[2] - q[2]))
            matched += 1
    return {
        "jaccard_names": round(jaccard, 3),
        "only_a": sorted(set_a - set_b),
        "only_b": sorted(set_b - set_a),
        "n_leaves_a": len(sk_a),
        "n_leaves_b": len(sk_b),
        "matched_parts": matched,
        "mean_pos_delta": round(sum(pos_deltas) / matched, 4) if matched else None,
        "mean_scale_delta": round(sum(scale_deltas) / matched, 4) if matched else None,
    }


def main():
    p = argparse.ArgumentParser(description="Gate 2: diff decomposer output across backends")
    p.add_argument("--gemma", required=True, help="Gemma model folder, e.g. models/gemma-4-e4b-it")
    p.add_argument("--hertz", default=None, help="SGS decomposer checkpoint (omit for Gemma-only dump)")
    p.add_argument("--hertz-tokenizer", default="data/hertz12_data/tokenizer.model")
    p.add_argument("--prompts", required=True, help="one prompt per line")
    p.add_argument("--out", default="results/decomposer_compare.json")
    p.add_argument("--exemplars", default="data/decomposition_trees/path1_train.json")
    p.add_argument("--n-shot", type=int, default=3)
    args = p.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scripts.gemma_decomposer import GemmaDecomposer
    gemma = GemmaDecomposer(args.gemma, exemplars_path=args.exemplars,
                            n_shot=args.n_shot, temperature=0.1)
    gemma.scan_library = None

    hertz = None
    if args.hertz:
        from scripts.infer_decomposer import Decomposer
        print(f"[compare] loading SGS decomposer {args.hertz} ...")
        hertz = Decomposer(args.hertz, args.hertz_tokenizer, device)
        hertz.scan_library = None
        print("[compare] SGS decomposer ready.")
    else:
        print("[compare] no --hertz: Gemma-only dump (no diff).")

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    print(f"[compare] {len(prompts)} prompts")

    rows = []
    for i, prompt in enumerate(prompts):
        g_tree = gemma.generate_tree(prompt)
        g_sk = skeleton(g_tree)
        row = {"prompt": prompt, "gemma_skeleton": g_sk, "gemma_tree": g_tree}
        if hertz is not None:
            h_tree = hertz.generate_tree(prompt)
            h_sk = skeleton(h_tree)
            row["hertz_skeleton"] = h_sk
            row["hertz_tree"] = h_tree
            row["diff"] = compare(g_sk, h_sk) if (g_sk and h_sk) else None
        rows.append(row)
        if hertz is not None and row["diff"]:
            d = row["diff"]
            print(f"  [{i+1}/{len(prompts)}] J={d['jaccard_names']:.2f} "
                  f"posΔ={d['mean_pos_delta']} leaves G={d['n_leaves_a']}/H={d['n_leaves_b']} "
                  f"| onlyG={d['only_a']} onlyH={d['only_b']}  {prompt[:40]}")
        else:
            print(f"  [{i+1}/{len(prompts)}] gemma leaves={len(g_sk)}  {prompt[:48]}")

    summary = {"gemma": args.gemma, "hertz": args.hertz, "n_prompts": len(prompts)}
    diffs = [r["diff"] for r in rows if r.get("diff")]
    if diffs:
        summary["mean_jaccard"] = round(sum(d["jaccard_names"] for d in diffs) / len(diffs), 3)
        pos = [d["mean_pos_delta"] for d in diffs if d["mean_pos_delta"] is not None]
        summary["mean_pos_delta"] = round(sum(pos) / len(pos), 4) if pos else None
        summary["prompts_with_diff_parts"] = sum(
            1 for d in diffs if d["only_a"] or d["only_b"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)

    print("\n[compare] ==================================")
    if diffs:
        print(f"  mean name-Jaccard : {summary['mean_jaccard']}  (1.0 = identical part sets)")
        print(f"  mean pose delta   : {summary['mean_pos_delta']}  (0.0 = identical placement)")
        print(f"  prompts where part sets DIFFER: {summary['prompts_with_diff_parts']}/{len(diffs)}")
        print("  Low Jaccard / high pose delta on BREADTH prompts = Gemma genuinely")
        print("  decomposes differently. Identical on castles = expected (saturated).")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
