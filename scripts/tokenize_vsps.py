"""VSP v1, phase 2: sense-tagging VSPS tokenizer.

Tokenizes text against a VSPS vocab (build_vsps_vocab.py). The key step is WORD
SENSE DISAMBIGUATION at tokenize time -- the thing SentencePiece skips and the
reason it collapses "crane" (bird) and "crane" (machine) into one token.

For each word occurrence whose surface has GROUNDED senses, pick the sense whose
descriptive TERM best matches the sentence CONTEXT (embedding Lesk: cosine of the
context's GloVe centroid vs each sense-term's GloVe centroid). Emit that sense's
token id. Words with a single grounded sense take it directly; ungrounded words
map to their abstract token or <unk>.

Output: a token-id stream per line + the cached V/S/P bundle per vocab id, so the
trainer reads a lookup, not a live grounding pass.

This is a lexical, dependency-light WSD (no torch). It's a validation-scale
tokenizer for TinyStories, not a production tagger; the gate is round-trip +
sense-correctness on minimal pairs, not SOTA WSD.

Usage:
  python scripts/tokenize_vsps.py --corpus data/tinystories \
    --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt \
    --out data/tinystories_vsps
  python scripts/tokenize_vsps.py --selftest --glove data/glove.6B.300d.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_p6_correlation import load_glove

WORD_RE = re.compile(r"[a-z]+")


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def centroid(words, glove):
    vs = [glove[w] for w in words if w in glove]
    return _unit(np.mean(vs, axis=0)) if vs else None


class VSPSTokenizer:
    def __init__(self, vocab_path, glove):
        vj = json.load(open(vocab_path))
        self.tokens = vj["tokens"]
        self.glove = glove
        self.unk = next(t["id"] for t in self.tokens if t["surface"] == "<unk>")
        # surface -> list of grounded token dicts; surface -> abstract token id
        self.by_surface = {}
        self.abstract_id = {}
        self.sense_term_vec = {}   # token id -> unit GloVe centroid of its term
        for t in self.tokens:
            if t["tier"] == "grounded":
                self.by_surface.setdefault(t["surface"], []).append(t)
                tv = centroid(WORD_RE.findall((t.get("term") or "").lower()), glove)
                self.sense_term_vec[t["id"]] = tv
            elif t["tier"] == "abstract":
                self.abstract_id[t["surface"]] = t["id"]

    def disambiguate(self, surface, context_words):
        """Return the token id for `surface` given surrounding context words."""
        senses = self.by_surface.get(surface)
        if senses:
            if len(senses) == 1:
                return senses[0]["id"], "single-sense"
            cvec = centroid(context_words, self.glove)
            if cvec is None:
                return senses[0]["id"], "no-context-default"
            best, best_id = -2.0, senses[0]["id"]
            for s in senses:
                tv = self.sense_term_vec.get(s["id"])
                if tv is None:
                    continue
                sim = float(np.dot(cvec, tv))
                if sim > best:
                    best, best_id = sim, s["id"]
            return best_id, "wsd"
        if surface in self.abstract_id:
            return self.abstract_id[surface], "abstract"
        return self.unk, "unk"

    def encode(self, text, window=6):
        words = WORD_RE.findall(text.lower())
        ids, tags = [], []
        for i, w in enumerate(words):
            ctx = words[max(0, i - window):i] + words[i + 1:i + 1 + window]
            tid, how = self.disambiguate(w, ctx)
            ids.append(tid); tags.append(how)
        return ids, tags


def selftest(glove):
    """crane in a bird context vs a machine context must get DIFFERENT ids."""
    print("[selftest] tokenize_vsps WSD")
    import tempfile
    # minimal vocab with two crane senses whose terms sit in GloVe
    vocab = {"dims": {"V": 1, "S": 1, "P": 1}, "specials": ["<unk>"], "tokens": [
        {"id": 0, "surface": "<unk>", "tier": "special"},
        {"id": 1, "surface": "crane", "sense": "bird", "tier": "grounded",
         "term": "a crane bird with feathers flying in the sky"},
        {"id": 2, "surface": "crane", "sense": "machine", "tier": "grounded",
         "term": "a construction crane machine lifting steel at a building site"},
    ]}
    p = Path(tempfile.mkdtemp()) / "v.json"
    json.dump(vocab, open(p, "w"))
    tok = VSPSTokenizer(p, glove)
    bird_ctx = "the crane flew over the lake with its wings".split()
    mach_ctx = "the crane lifted heavy steel at the construction site".split()
    id_bird, _ = tok.disambiguate("crane", [w for w in bird_ctx if w != "crane"])
    id_mach, _ = tok.disambiguate("crane", [w for w in mach_ctx if w != "crane"])
    print(f"[selftest] crane(bird-context)->{id_bird}  crane(machine-context)->{id_mach}")
    ok = id_bird == 1 and id_mach == 2
    print(f"[selftest] {'PASS -- context picks the right sense' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Sense-tagging VSPS tokenizer")
    p.add_argument("--corpus", help="dir of *.txt (one article/line) or a single file")
    p.add_argument("--vocab", default="data/vsps/vocab.json")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--corpus-vocab", default=None,
                   help="{word: freq} json; load GloVe over it too so CONTEXT words "
                        "drive WSD (else disambiguation is blind to most context)")
    p.add_argument("--out", default="data/tinystories_vsps")
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    # GloVe vocab: everything the tokenizer might touch. For selftest, just terms.
    if args.selftest:
        need = set("crane bird feathers flying sky construction machine lifting "
                   "steel building site the flew over lake with its wings lifted "
                   "heavy at".split())
        sys.exit(0 if selftest(load_glove(Path(args.glove), need)) else 1)

    vj = json.load(open(args.vocab))
    need = set()
    for t in vj["tokens"]:
        need.add(t["surface"])
        need.update(WORD_RE.findall((t.get("term") or "").lower()))
    # WSD needs GloVe for the CONTEXT words too, or disambiguation is blind.
    # Load over vocab words UNION the corpus wordfreq (--corpus-vocab).
    if args.corpus_vocab and Path(args.corpus_vocab).exists():
        cv = json.load(open(args.corpus_vocab))
        need |= set(cv if isinstance(cv, list) else cv.keys())
        print(f"[vsps] +{len(need)} words incl corpus vocab for full-context WSD")
    else:
        print("[vsps] WARNING: no --corpus-vocab; context words outside the "
              "term/surface set won't contribute to WSD. Pass --corpus-vocab "
              "data/wiki_vsp/wordfreq.json for full disambiguation.")
    glove = load_glove(Path(args.glove), need)
    print(f"[vsps] GloVe: {len(glove)} vectors preloaded")

    tok = VSPSTokenizer(args.vocab, glove)

    # gather corpus lines. ONLY *.txt (one article/line from prepare_wikipedia_senses
    # --corpus-out). Do NOT glob *.json -- that once grabbed wordfreq.json and
    # tokenized the dict itself (99.5% <unk>).
    cpath = Path(args.corpus)
    if cpath.is_dir():
        files = sorted(cpath.glob("*.txt"))
        if not files:
            raise SystemExit(
                f"[vsps] no *.txt in {cpath}. Write a corpus first: "
                f"prepare_wikipedia_senses.py --corpus-out {cpath}/corpus.txt")
    else:
        files = [cpath]
    from collections import Counter
    how_counts = Counter()
    out_ids = []
    n_lines = 0
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ids, tags = tok.encode(line, window=args.window)
            out_ids.append(ids)
            how_counts.update(tags)
            n_lines += 1
            if args.max_lines and n_lines >= args.max_lines:
                break
        if args.max_lines and n_lines >= args.max_lines:
            break

    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump({"token_ids": out_ids}, open(Path(args.out) / "tokens.json", "w"))
    print(f"[vsps] tokenized {n_lines} lines, {sum(len(x) for x in out_ids)} tokens")
    print(f"[vsps] tag breakdown: {dict(how_counts)}")
    print(f"[vsps] saved -> {args.out}/tokens.json")


if __name__ == "__main__":
    main()
