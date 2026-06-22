"""
Raum Path B: GROUNDED V/P -- the fix for the derived-gate failure.

derive_vsp.py took V/P from the sense TERM's GloVe vector, which collapses on
colliding surface forms (crane bird vs crane machine -> same text -> same V/P).
ground_vsp.py removes text from the V/P path entirely:

  V (visual): keyed to a curated SHAPENET SYNSET id per sense (the non-textual
    bridge -- "crane.machine" -> synset 02958343 car, "crane.bird" -> null/animal,
    DIFFERENT identities even though the word collides). Vector is the real blob
    centroid if data/blobs/<name>.pt exists (build_blobs_shapenet.py), else a
    one-hot over the synset vocabulary. Either way it does NOT come from the word.

  P (physical): the P6 MEASURED MATERIAL_TABLE value for a curated material name
    per sense (granite/steel/water/...). Real measurements, not the GloVe->P MLP.

The only human input is the curated sense -> {synset, material} map
(scripts/assets/vsp_grounded_map.json). That curation IS the legitimate
grounding -- a human disambiguates the sense ONCE; everything downstream is
asset/measurement, never the colliding text.

Output matches derive_vsp / the gating probe --derived schema:
  {"vis_categories": [...], "entries": [{word, senses:[{label, visual_dist,
   physical:{prop:val}}]}]}

Usage:
  python scripts/ground_vsp.py --map scripts/assets/vsp_grounded_map.json \
    --out results/vsp_grounded.json
  # then:
  python scripts/vsp_gating_probe.py --derived results/vsp_grounded.json \
    --glove data/glove.6B.300d.txt --out results/vsp_gating_grounded.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_p6_correlation import MATERIAL_TABLE, PROPERTY_NAMES
from scripts.build_blobs_shapenet import SYNSET_TO_NAME


def build_synset_vocab(entries):
    """Ordered list of every synset used + 'abstract' (null) bucket -> V index."""
    used = []
    for e in entries:
        for s in e["senses"]:
            syn = s.get("synset") or "abstract"
            if syn not in used:
                used.append(syn)
    if "abstract" not in used:
        used.append("abstract")
    return used


def visual_vector(synset, vocab, blob_dir):
    """Real blob centroid (mean over the synset's Gaussians) if available, else
    a one-hot over the synset vocab. Centroid path: data/blobs/<name>.pt."""
    syn = synset or "abstract"
    # try real blob centroid
    if synset and synset in SYNSET_TO_NAME and blob_dir:
        name = SYNSET_TO_NAME[synset]
        pt = Path(blob_dir) / f"{name}.pt"
        if pt.exists():
            try:
                import torch
                blob = torch.load(pt, map_location="cpu", weights_only=False)
                means = blob["means"] if isinstance(blob, dict) else blob
                c = means.mean(0).numpy().astype(np.float32)
                # pad/truncate to a fixed 16-d visual descriptor for concat stability
                v = np.zeros(16, dtype=np.float32)
                v[:min(3, c.shape[0])] = c[:3]
                # append a synset one-hot tail so identity still separates senses
                idx = vocab.index(syn)
                tail = np.zeros(len(vocab), dtype=np.float32); tail[idx] = 1.0
                return np.concatenate([v[:3], tail])
            except Exception:
                pass
    # fallback: pure synset one-hot (identity-only V)
    one = np.zeros(len(vocab), dtype=np.float32)
    one[vocab.index(syn)] = 1.0
    return np.concatenate([np.zeros(3, dtype=np.float32), one])


def phys_vector(material):
    if material and material in MATERIAL_TABLE:
        return np.array(MATERIAL_TABLE[material], dtype=np.float32)
    return np.zeros(len(PROPERTY_NAMES), dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Grounded V/P from curated synset+material map")
    p.add_argument("--map", default="scripts/assets/vsp_grounded_map.json")
    p.add_argument("--blob-dir", default="data/blobs",
                   help="ShapeNet blob dir (real centroids); falls back to one-hot if absent")
    p.add_argument("--out", default="results/vsp_grounded.json")
    args = p.parse_args()

    data = json.load(open(args.map))
    entries = data["entries"]
    vocab = build_synset_vocab(entries)
    blob_dir = args.blob_dir if Path(args.blob_dir).exists() else None
    print(f"[ground] {len(entries)} words, {len(vocab)} synset buckets, "
          f"blobs: {'REAL centroids from ' + args.blob_dir if blob_dir else 'one-hot (no blobs downloaded)'}")

    out = []
    for e in entries:
        senses_out = []
        for s in e["senses"]:
            V = visual_vector(s.get("synset"), vocab, blob_dir)
            P = phys_vector(s.get("material"))
            senses_out.append({
                "label": s["label"],
                "synset": s.get("synset"),
                "material": s.get("material"),
                "visual_dist": [round(float(x), 4) for x in V],
                "physical": {n: round(float(v), 4) for n, v in zip(PROPERTY_NAMES, P)},
            })
        if len(senses_out) >= 2:
            out.append({"word": e["word"], "senses": senses_out})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"vis_categories": vocab, "entries": out}, open(args.out, "w"), indent=2)
    print(f"[ground] {len(out)} words grounded -> {args.out}")
    print(f"[ground] next: vsp_gating_probe.py --derived {args.out}")


if __name__ == "__main__":
    main()
