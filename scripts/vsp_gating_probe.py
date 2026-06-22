"""
Raum Path B: VSP gating probe -- the kill-fast reachability test.

The cheapest possible test of the whole VSP thesis. NO training, ~minutes to
run. Question: does a Visual+Semantic+Physical token representation SEPARATE
polysemous senses that a Semantic-only representation COLLAPSES?

If yes, VSP is real -- "bank (river)" and "bank (institution)" become distinct
because their V (water vs building) and P (none vs concrete) differ even at
identical text. This is the documented fix for the Raum 1.2 failure (SentencePiece
merged related words, 35 collision groups / 300 classes) and the basis for
Planck 2.0 + the VSP-Models paper.

If no separation, STOP -- the binding does not buy disambiguation and Path B
should not be trained.

How it works:
  S (semantic): GloVe 300d vector for the word (data/glove.6B.300d.txt). The
    SAME word string -> the SAME S vector for both senses (that's the collapse).
  V (visual): a per-sense visual-category descriptor. Seeded here from a small
    hand table (water/building/...) as a one-hot over visual categories; the
    full version pulls the nearest Objaverse/ShapeNet blob centroid per sense.
  P (physical): a per-sense physical-property vector over the P6 properties
    (hardness, density, friction, ...). Hand-seeded here; full version reads the
    P6 material lookup.

Metric: for each polysemous WORD, measure similarity between its two SENSES
under (a) S-only and (b) full V|S|P concat. The thesis predicts S-only ~ HIGH
(senses collapse) while V|S|P is LOWER (senses separate). We report the gap.

Usage:
  python scripts/vsp_gating_probe.py --words scripts/assets/vsp_polysemous.json \
    --glove data/glove.6B.300d.txt --out results/vsp_gating.json
  python scripts/vsp_gating_probe.py --words ... --no-glove   # S from a stub, runs anywhere
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# P6 physical properties (order fixed; matches results/p6_correlation_*.json)
P6_PROPS = ["hardness", "elasticity", "friction", "density", "brittleness",
            "thermal_conductivity", "transparency", "deformability"]

# Visual category vocabulary for the hand-seeded V (one-hot). The full probe
# replaces this with nearest-blob centroids from the Objaverse/ShapeNet library.
VIS_CATEGORIES = ["water", "building", "aircraft", "flat_surface", "animal",
                  "tool", "plant", "vehicle", "furniture", "terrain", "none"]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def load_glove(path, words):
    """Load GloVe vectors for just the words we need (streamed, no full load)."""
    need = set(words)
    vecs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split(" ", 1)
            if tok[0] in need:
                vecs[tok[0]] = np.fromstring(tok[1], sep=" ", dtype=np.float32)
                if len(vecs) == len(need):
                    break
    return vecs


def vis_vector(category):
    v = np.zeros(len(VIS_CATEGORIES), dtype=np.float32)
    if category in VIS_CATEGORIES:
        v[VIS_CATEGORIES.index(category)] = 1.0
    return v


def phys_vector(props):
    """props: {prop_name: value in [0,1]} -> vector over P6_PROPS."""
    return np.array([props.get(p, 0.0) for p in P6_PROPS], dtype=np.float32)


def cos(a, b):
    a, b = _unit(a), _unit(b)
    return float(np.dot(a, b))


def main():
    p = argparse.ArgumentParser(description="VSP gating probe: do V/S/P separate senses S collapses?")
    p.add_argument("--words", default="scripts/assets/vsp_polysemous.json")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--no-glove", action="store_true",
                   help="skip GloVe (use a deterministic stub S); makes the probe run anywhere")
    p.add_argument("--out", default="results/vsp_gating.json")
    p.add_argument("--derived", default=None,
                   help="use AUTO-derived V/P from derive_vsp.py output (results/"
                        "vsp_derived.json) instead of the hand-seeded --words table")
    args = p.parse_args()

    if args.derived:
        # derive_vsp.py already produced per-sense visual_dist + physical; convert
        # to the {word, senses:[{label, visual, physical}]} shape, with visual as
        # the soft distribution (not a one-hot category).
        dd = json.load(open(args.derived))
        entries = []
        for e in dd["entries"]:
            senses = [{"label": s["label"], "visual_dist": s["visual_dist"],
                       "physical": s["physical"]} for s in e["senses"]]
            entries.append({"word": e["word"], "senses": senses})
        print(f"[vsp] using AUTO-derived V/P from {args.derived}")
    else:
        entries = json.load(open(args.words))  # hand-seeded table
    base_words = [e["word"] for e in entries]

    glove = {}
    if not args.no_glove and Path(args.glove).exists():
        glove = load_glove(args.glove, base_words)
        print(f"[vsp] GloVe: {len(glove)}/{len(base_words)} words found")
    else:
        print("[vsp] GloVe OFF -- using deterministic stub S (string hash)")

    def s_vec(word):
        if word in glove:
            return glove[word]
        # deterministic stub: same word -> same vector (preserves the collapse)
        rng = np.random.default_rng(abs(hash(word)) % (2**32))
        return rng.standard_normal(300).astype(np.float32)

    results = []
    s_only_sims, vsp_sims = [], []
    for e in entries:
        word = e["word"]
        senses = e["senses"]
        if len(senses) < 2:
            continue
        s = s_vec(word)
        # build S-only and V|S|P vectors per sense (S identical across senses)
        sense_vecs = []
        for sn in senses:
            # V: a soft distribution (auto-derived) or a one-hot (hand-seeded)
            if "visual_dist" in sn:
                v = np.array(sn["visual_dist"], dtype=np.float32)
            else:
                v = vis_vector(sn.get("visual", "none"))
            ph = phys_vector(sn.get("physical", {}))
            # scale-balance the three blocks to unit norm each before concat
            vsp = np.concatenate([_unit(v), _unit(s), _unit(ph)])
            sense_vecs.append((sn["label"], v, s, ph, vsp))
        # pairwise (first two senses) -- the canonical polysemy pair
        (l0, v0, s0, p0, vsp0), (l1, v1, s1, p1, vsp1) = sense_vecs[0], sense_vecs[1]
        s_sim = cos(s0, s1)          # identical -> 1.0 (the collapse)
        vsp_sim = cos(vsp0, vsp1)
        s_only_sims.append(s_sim); vsp_sims.append(vsp_sim)
        results.append({"word": word, "sense_a": l0, "sense_b": l1,
                        "s_only_sim": round(s_sim, 4), "vsp_sim": round(vsp_sim, 4),
                        "separation_gain": round(s_sim - vsp_sim, 4)})

    mean_s = float(np.mean(s_only_sims)) if s_only_sims else 0.0
    mean_vsp = float(np.mean(vsp_sims)) if vsp_sims else 0.0
    gain = mean_s - mean_vsp

    print(f"\n{'word':<14}{'sense A':<16}{'sense B':<16}{'S-only':>8}{'V|S|P':>8}{'gain':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['word']:<14}{r['sense_a']:<16}{r['sense_b']:<16}"
              f"{r['s_only_sim']:>8.3f}{r['vsp_sim']:>8.3f}{r['separation_gain']:>8.3f}")
    print("-" * 70)
    print(f"{'MEAN':<46}{mean_s:>8.3f}{mean_vsp:>8.3f}{gain:>8.3f}")

    # GATE: V/S/P must measurably separate senses S-only collapses.
    PASS = mean_s > 0.9 and gain > 0.3
    verdict = "PASS -- VSP separates senses S collapses; Path B is reachable" if PASS \
        else "FAIL -- no meaningful separation; reconsider Path B before training"
    print(f"\nGATE: {verdict}")
    print(f"  (criterion: S-only mean > 0.9 AND separation gain > 0.3)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "mean_s_only": mean_s, "mean_vsp": mean_vsp,
               "separation_gain": gain, "pass": PASS}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
