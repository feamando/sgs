"""
Raum Path B: automatic V and P derivation for a word SENSE.

The gating probe (vsp_gating_probe.py) first ran on HAND-SEEDED V and P, which
proves the representation CAN separate but not that AUTO-derived V/P will. This
module derives V and P automatically from a SENSE TERM (a disambiguating word/
phrase, e.g. "bank" -> sense terms "riverbank" and "financial institution"),
so the only human input is naming the senses, not authoring their vectors.

P (physical), automatic via the P6 finding:
  The P6 result is that GloVe -> physical properties IS learnable (hardness
  R^2=0.54). We train that exact MLP (PhysicsPredictionMLP) on the 80-material
  MATERIAL_TABLE, then PREDICT the 8-property vector for any sense term's GloVe
  vector. Polysemy is handled because each SENSE has its own disambiguating
  term -> its own GloVe vector -> its own predicted P. "bank" alone has one
  GloVe vector (the collapse); "riverbank" vs "bank_building" do not.

V (visual), automatic via category anchors:
  Instead of a hand one-hot, score the sense term's GloVe similarity against a
  small set of ANCHOR words per visual category (water: river/lake/ocean;
  building: house/tower/wall; ...). The category with the highest mean anchor
  similarity is the sense's visual class; V = the softmax distribution over
  categories (a soft, derived descriptor, not a hand label).

Reused from scripts/validate_p6_correlation.py: MATERIAL_TABLE,
PhysicsPredictionMLP, PROPERTY_NAMES, load_glove.

Usage:
  python scripts/derive_vsp.py --senses scripts/assets/vsp_sense_terms.json \
    --glove data/glove.6B.300d.txt --out results/vsp_derived.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_p6_correlation import (
    MATERIAL_TABLE, PROPERTY_NAMES, PhysicsPredictionMLP, load_glove,
)

# Visual category anchors (each category = a few unambiguous example words).
# V is DERIVED by GloVe similarity to these, not hand-assigned per sense.
VIS_ANCHORS = {
    "water":        ["river", "lake", "ocean", "water"],
    "building":     ["house", "tower", "wall", "building"],
    "aircraft":     ["airplane", "jet", "helicopter", "aircraft"],
    "flat_surface": ["plane", "sheet", "board", "surface"],
    "animal":       ["dog", "bird", "fish", "animal"],
    "tool":         ["hammer", "wrench", "device", "tool"],
    "plant":        ["tree", "leaf", "flower", "plant"],
    "vehicle":      ["car", "truck", "machine", "vehicle"],
    "furniture":    ["chair", "table", "desk", "furniture"],
    "terrain":      ["rock", "ground", "hill", "terrain"],
}
VIS_CATEGORIES = list(VIS_ANCHORS)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def train_p6_mlp(glove, device="cpu", epochs=300, seed=0):
    """Train GloVe -> 8 physical properties on MATERIAL_TABLE. Returns the MLP."""
    import torch
    torch.manual_seed(seed)
    words, props = [], []
    for w, p in MATERIAL_TABLE.items():
        if w in glove and len(p) == len(PROPERTY_NAMES):
            words.append(w); props.append(p)
    X = torch.tensor(np.stack([glove[w] for w in words]), dtype=torch.float32)
    Y = torch.tensor(np.array(props), dtype=torch.float32)
    mlp = PhysicsPredictionMLP()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(mlp(X), Y)
        loss.backward(); opt.step()
    print(f"[derive] P6 MLP trained on {len(words)} materials, final MSE {loss.item():.4f}")
    return mlp


def derive_P(term_vec, mlp):
    import torch
    with torch.no_grad():
        return mlp(torch.tensor(term_vec[None], dtype=torch.float32))[0].numpy()


def derive_V(term_vec, anchor_vecs):
    """Softmax distribution over visual categories by mean anchor similarity."""
    scores = []
    tv = _unit(term_vec)
    for cat in VIS_CATEGORIES:
        sims = [float(np.dot(tv, _unit(a))) for a in anchor_vecs[cat] if a is not None]
        scores.append(np.mean(sims) if sims else -1.0)
    scores = np.array(scores)
    e = np.exp((scores - scores.max()) / 0.1)   # temperature 0.1 -> peaky
    return e / e.sum(), VIS_CATEGORIES[int(scores.argmax())]


def main():
    p = argparse.ArgumentParser(description="Derive V and P automatically from sense terms")
    p.add_argument("--senses", default="scripts/assets/vsp_sense_terms.json")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--out", default="results/vsp_derived.json")
    args = p.parse_args()

    entries = json.load(open(args.senses))  # [{word, senses:[{label, term}]}]
    # vocab = all sense terms + material words + anchor words
    terms = {s["term"] for e in entries for s in e["senses"]}
    vocab = set(MATERIAL_TABLE) | terms
    for anchors in VIS_ANCHORS.values():
        vocab |= set(anchors)

    glove = load_glove(Path(args.glove), vocab)
    print(f"[derive] GloVe: {len(glove)}/{len(vocab)} vocab words found")
    anchor_vecs = {c: [glove.get(a) for a in ws] for c, ws in VIS_ANCHORS.items()}
    mlp = train_p6_mlp(glove)

    out = []
    for e in entries:
        senses_out = []
        for s in e["senses"]:
            tv = glove.get(s["term"])
            if tv is None:
                print(f"  WARN no GloVe for sense term '{s['term']}' ({e['word']})")
                continue
            P = derive_P(tv, mlp)
            Vdist, Vcat = derive_V(tv, anchor_vecs)
            senses_out.append({
                "label": s["label"], "term": s["term"],
                "visual_category": Vcat,
                "visual_dist": [round(float(x), 4) for x in Vdist],
                "physical": {n: round(float(v), 4) for n, v in zip(PROPERTY_NAMES, P)},
            })
        if len(senses_out) >= 2:
            out.append({"word": e["word"], "senses": senses_out})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"vis_categories": VIS_CATEGORIES, "entries": out}, open(args.out, "w"), indent=2)
    print(f"[derive] {len(out)} words with derived V/P -> {args.out}")
    # quick readout
    for e in out[:6]:
        cats = " vs ".join(s["visual_category"] for s in e["senses"][:2])
        print(f"   {e['word']:12s} {cats}")
    print(f"[derive] next: vsp_gating_probe.py --derived {args.out}")


if __name__ == "__main__":
    main()
