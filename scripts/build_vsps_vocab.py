"""VSP v1, phase 1: build the two-tier VSPS vocabulary.

A VSPS token is a word-SENSE, not a subword. Two tiers:
  - GROUNDED tokens: one per concrete sense, carrying (V, S, P):
      V = CLIP-image visual vector (auto, from derive_vsp_clip.py)
      P = derived physics vector    (auto, from derive_p6_from_vector.py)
      S = GloVe semantic vector      (word-level; identical across a word's senses)
    crane-bird and crane-machine are TWO grounded tokens. This is the whole point:
    the sense split SentencePiece can't make (it merges them into one subword).
  - ABSTRACT tokens: frequent corpus words with no grounded sense (the, justice,
    however). S-only; V and P are zero. That is CORRECT, not a gap.

Plus specials (<pad>, <unk>, <bos>, <eos>).

Blocks are stored at their native dims (V=512 CLIP, S=300 GloVe, P=8) and each is
UNIT-normalized so no block dominates by scale (mirrors the gating probe). The
trainer projects/concats them into the model embedding (phase 3).

Every V/S/P is auto-derived; no hand labels enter the vocab (the VSP gate lesson,
[[project_sgs_vsp_gate]]).

Usage:
  python scripts/build_vsps_vocab.py \
    --senses results/vsp_clip_image_pderiv.json \
    --glove data/glove.6B.300d.txt \
    --corpus-vocab data/vsps/tinystories_wordfreq.json \
    --out data/vsps/vocab.json
  # --corpus-vocab optional; without it, abstract tier = the grounded words'
  # surface forms only (enough to unit-test the builder off the box).
  python scripts/build_vsps_vocab.py --selftest
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_p6_correlation import load_glove, PROPERTY_NAMES

V_DIM = 512   # CLIP ViT-B/32 image embedding
S_DIM = 300   # GloVe
P_DIM = len(PROPERTY_NAMES)  # 8

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _round(v, nd=5):
    return [round(float(x), nd) for x in v]


def build_grounded(senses_path, glove):
    """One grounded token per sense from the derive output. Returns list of token
    dicts + the set of surface words that got grounded."""
    d = json.load(open(senses_path))
    entries = d.get("entries", d)
    tokens, grounded_words = [], set()
    for e in entries:
        word = e["word"]
        s_vec = _unit(glove[word]) if word in glove else np.zeros(S_DIM, np.float32)
        for sn in e["senses"]:
            V = _unit(sn["visual_dist"])
            P = _unit(list(sn["physical"].values())) if "physical" in sn \
                else np.zeros(P_DIM, np.float32)
            tokens.append({
                "surface": word,
                "sense": sn["label"],
                "term": sn.get("term"),
                "tier": "grounded",
                "V": _round(V), "S": _round(s_vec), "P": _round(P),
                "has_v": bool(np.any(V)), "has_p": bool(np.any(P)),
            })
        grounded_words.add(word)
    return tokens, grounded_words


def build_abstract(corpus_vocab, grounded_words, glove, top_n):
    """Abstract (S-only) tokens: frequent corpus words NOT already grounded.
    V and P are zero (correct: no physical/visual grounding for 'the', 'justice')."""
    tokens = []
    if not corpus_vocab:
        return tokens
    # corpus_vocab: {word: freq} or [word,...]; take the most frequent.
    if isinstance(corpus_vocab, dict):
        ranked = [w for w, _ in Counter(corpus_vocab).most_common(top_n)]
    else:
        ranked = list(corpus_vocab)[:top_n]
    zeroV, zeroP = [0.0] * V_DIM, [0.0] * P_DIM
    for w in ranked:
        if w in grounded_words:
            continue  # already has grounded sense token(s)
        s_vec = _unit(glove[w]) if w in glove else np.zeros(S_DIM, np.float32)
        tokens.append({
            "surface": w, "sense": None, "term": None, "tier": "abstract",
            "V": zeroV, "S": _round(s_vec), "P": zeroP,
            "has_v": False, "has_p": False,
        })
    return tokens


def assemble_vocab(grounded, abstract):
    """Specials first (stable ids), then grounded, then abstract."""
    vocab = []
    zeroV, zeroS, zeroP = [0.0] * V_DIM, [0.0] * S_DIM, [0.0] * P_DIM
    for sp in SPECIALS:
        vocab.append({"surface": sp, "sense": None, "term": None, "tier": "special",
                      "V": zeroV, "S": zeroS, "P": zeroP, "has_v": False, "has_p": False})
    vocab.extend(grounded)
    vocab.extend(abstract)
    for i, t in enumerate(vocab):
        t["id"] = i
    return vocab


def coverage_report(vocab):
    tiers = Counter(t["tier"] for t in vocab)
    grounded = [t for t in vocab if t["tier"] == "grounded"]
    words = Counter(t["surface"] for t in grounded)
    multi = {w: c for w, c in words.items() if c >= 2}
    blowup = (len(grounded) / len(words)) if words else 0.0
    return {
        "total_tokens": len(vocab),
        "specials": tiers["special"], "grounded": tiers["grounded"],
        "abstract": tiers["abstract"],
        "grounded_words": len(words),
        "polysemous_words": len(multi),          # words that got >=2 sense tokens
        "grounded_blowup_x": round(blowup, 3),   # sense tokens per grounded word
        "with_v": sum(1 for t in grounded if t["has_v"]),
        "with_p": sum(1 for t in grounded if t["has_p"]),
    }


def selftest():
    """Build a tiny vocab from a synthetic senses file; check tiers, dims, ids."""
    print("[selftest] build_vsps_vocab")
    rng = np.random.default_rng(0)
    glove = {"crane": rng.standard_normal(S_DIM).astype(np.float32),
             "the": rng.standard_normal(S_DIM).astype(np.float32)}
    senses = {"entries": [{"word": "crane", "senses": [
        {"label": "bird", "term": "a crane bird", "visual_dist": list(rng.standard_normal(V_DIM)),
         "physical": {n: float(rng.random()) for n in PROPERTY_NAMES}},
        {"label": "machine", "term": "a crane machine", "visual_dist": list(rng.standard_normal(V_DIM)),
         "physical": {n: float(rng.random()) for n in PROPERTY_NAMES}},
    ]}]}
    import tempfile
    p = Path(tempfile.mkdtemp()) / "s.json"
    json.dump(senses, open(p, "w"))
    g, gw = build_grounded(p, glove)
    a = build_abstract({"the": 100, "crane": 50}, gw, glove, top_n=10)
    vocab = assemble_vocab(g, a)
    rep = coverage_report(vocab)
    ok = (rep["grounded"] == 2 and rep["polysemous_words"] == 1
          and rep["abstract"] == 1  # 'the' abstract, 'crane' skipped (grounded)
          and len(vocab[4]["V"]) == V_DIM and len(vocab[4]["S"]) == S_DIM
          and vocab[0]["surface"] == "<pad>")
    print(f"[selftest] {rep}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Build the two-tier VSPS vocabulary")
    p.add_argument("--senses", default="results/vsp_clip_image_pderiv.json",
                   help="grounded senses (V + derived P) from derive_p6_from_vector.py")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--corpus-vocab", default=None,
                   help="{word: freq} json (or word list) for the abstract tier")
    p.add_argument("--abstract-top-n", type=int, default=8000)
    p.add_argument("--out", default="data/vsps/vocab.json")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    # vocab for GloVe: grounded words + corpus words
    senses = json.load(open(args.senses))
    entries = senses.get("entries", senses)
    need = {e["word"] for e in entries}
    corpus_vocab = None
    if args.corpus_vocab and Path(args.corpus_vocab).exists():
        corpus_vocab = json.load(open(args.corpus_vocab))
        need |= set(corpus_vocab if isinstance(corpus_vocab, list)
                    else corpus_vocab.keys())
    glove = load_glove(Path(args.glove), need)
    print(f"[vsps] GloVe: {len(glove)}/{len(need)} words")

    grounded, gw = build_grounded(args.senses, glove)
    abstract = build_abstract(corpus_vocab, gw, glove, args.abstract_top_n)
    vocab = assemble_vocab(grounded, abstract)
    rep = coverage_report(vocab)

    print("[vsps] coverage:")
    for k, v in rep.items():
        print(f"    {k}: {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dims": {"V": V_DIM, "S": S_DIM, "P": P_DIM},
               "specials": SPECIALS, "report": rep, "tokens": vocab},
              open(args.out, "w"))
    print(f"[vsps] saved {len(vocab)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
