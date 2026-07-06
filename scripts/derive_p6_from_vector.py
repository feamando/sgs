"""Raum Path B, phase 1(B): DERIVE P per sense, not look it up.

Phase 1(A) passed on auto-derived CLIP-image V, with P EXCLUDED because P was a
hand-authored material table (MATERIAL_TABLE) keyed by a hand-assigned material
tag per sense (two labeling layers = a curated prior, same tautology class as
the synset one-hot). To let the full V/S/P bundle count toward the gate -- and to
make the "encode everything into ONE vector, automatically" thesis real -- P must
be DERIVED from the sense's own semantics.

This trains the P6 physics MLP (GloVe 300d -> 8 physical properties) on the
MATERIAL_TABLE, then predicts P for each SENSE from the GloVe embedding of that
sense's descriptive phrase (e.g. "a crane bird with long legs"). No material tag
is read. The output is a drop-in replacement for the `physical` field in a
vsp_clip_image.json, so the gate can be re-run with --gate-blocks v,s,p.

Honest by construction: P now comes from a model applied to the sense phrase,
exactly like V comes from SD+CLIP applied to the sense phrase. Neither reads a
hand label.

Usage:
  python scripts/derive_p6_from_vector.py --in results/vsp_clip_image.json \
    --glove data/glove.6B.300d.txt --out results/vsp_clip_image_pderiv.json
  # then re-gate with P counting:
  python scripts/vsp_gating_probe.py --derived results/vsp_clip_image_pderiv.json \
    --glove data/glove.6B.300d.txt --aggregate max --gate-blocks v,s,p \
    --out results/vsp_gating_clip_image_pderiv.json
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


def phrase_vector(phrase, glove, dim=300):
    """Mean GloVe over the phrase's content words. Returns None if no word hits."""
    vecs = [glove[w] for w in phrase.lower().split() if w in glove]
    if not vecs:
        return None
    return np.mean(vecs, axis=0).astype(np.float32)


def train_p6_mlp(glove, epochs=400, seed=0):
    """Train GloVe(material word) -> 8 properties on MATERIAL_TABLE. Returns the
    fitted MLP. Single-word materials only (they have a clean GloVe vector)."""
    import torch
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    X, Y = [], []
    for mat, props in MATERIAL_TABLE.items():
        if mat in glove:                       # single-token material with a vector
            X.append(glove[mat]); Y.append(props)
    if len(X) < 5:
        raise SystemExit(f"only {len(X)} materials had GloVe vectors; need >=5")
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    model = PhysicsPredictionMLP(input_dim=X.shape[1], output_dim=len(PROPERTY_NAMES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), Y)
        loss.backward(); opt.step()
    model.eval()
    print(f"[p6] trained physics MLP on {len(X)} materials, final MSE {loss.item():.4f}")
    return model


def main():
    p = argparse.ArgumentParser(description="Derive P per sense from its phrase (P6 MLP)")
    p.add_argument("--in", dest="inp", default="results/vsp_clip_image.json",
                   help="a derive output with per-sense {term, visual_dist, physical}")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--out", default="results/vsp_clip_image_pderiv.json")
    p.add_argument("--epochs", type=int, default=400)
    args = p.parse_args()

    import torch
    data = json.load(open(args.inp))
    entries = data["entries"] if "entries" in data else data

    # vocab = material words (for training) + sense-phrase words (for prediction)
    vocab = set(MATERIAL_TABLE.keys())
    for e in entries:
        for s in e["senses"]:
            vocab.update((s.get("term") or "").lower().split())
    glove = load_glove(Path(args.glove), vocab)
    print(f"[p6] GloVe: {len(glove)} vectors for {len(vocab)} vocab words")

    mlp = train_p6_mlp(glove, epochs=args.epochs)

    n_derived, n_fallback = 0, 0
    for e in entries:
        for s in e["senses"]:
            pv = phrase_vector(s.get("term") or s.get("label", ""), glove)
            if pv is None:
                # no GloVe hit -> zero P (honest: we couldn't derive it)
                P = np.zeros(len(PROPERTY_NAMES), dtype=np.float32)
                n_fallback += 1
            else:
                with torch.no_grad():
                    P = mlp(torch.tensor(pv).unsqueeze(0))[0].numpy()
                n_derived += 1
            # overwrite the hand-authored P with the DERIVED P
            s["physical"] = {name: round(float(v), 4)
                             for name, v in zip(PROPERTY_NAMES, P)}
            s["p_source"] = "p6_mlp_derived" if pv is not None else "zero_no_glove"
            s.pop("material", None)   # drop the hand tag; it no longer feeds P

    out = dict(data) if isinstance(data, dict) else {"entries": entries}
    out["entries"] = entries
    out["p_source"] = "p6_mlp_from_phrase"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[p6] derived P for {n_derived} senses ({n_fallback} zero-fallback); "
          f"saved -> {args.out}")


if __name__ == "__main__":
    main()
