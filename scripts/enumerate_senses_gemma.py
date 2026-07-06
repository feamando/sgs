"""VSP v1, phase 0.3: enumerate word senses (Gemma) + tag captions to senses.

Turns the COCO caption pile (prepare_coco_vsp.py) into a grounded-sense inventory:

  1. Take the frequent concrete nouns from wordfreq.json.
  2. Gemma 4 enumerates each noun's DISTINCT senses, each with a concrete,
     unambiguous grounding phrase (the probe lesson: "a crane bird with long
     legs", never bare "crane", or SD/CLIP draws the dominant prior).
  3. Tag caption occurrences to a sense (embedding-Lesk over the caption context
     vs each sense phrase) so every sense collects the REAL caption images that
     depict it -> native-photo V (no SD needed) in the next step.

Output senses_coco.json is the shape derive_vsp_clip.py / build_vsps_vocab.py
consume: {entries: [{word, senses: [{label, term, image_refs:[...]}]}]}.

Gemma's role = the curation that was manual in the 20-word probe: sense recall
(a missed rare sense = a missing token) and caption->sense matching. Both are
real knobs; log coverage.

Usage:
  python scripts/enumerate_senses_gemma.py --corpus data/coco_vsp \
    --model models/gemma-4-e4b-it --top-nouns 300 --min-freq 50 \
    --glove data/glove.6B.300d.txt --out data/vsps/senses_coco.json
  python scripts/enumerate_senses_gemma.py --selftest --glove data/glove.6B.300d.txt
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

# words that are frequent in captions but not groundable concrete-noun senses.
# (Abstract/function/verb-ish; they stay S-only in the vocab's abstract tier.)
NON_NOUN = set(
    "the a an of to and in on at is are was were be been it its this that with "
    "for as by from into over under near next down some two three man woman "
    "people person standing sitting holding walking looking wearing white black "
    "red blue green large small young old there here his her their they front "
    "behind while during very just also then than more most".split())


def _unit(v):
    v = np.asarray(v, dtype=np.float32); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def centroid(words, glove):
    vs = [glove[w] for w in words if w in glove]
    return _unit(np.mean(vs, axis=0)) if vs else None


def pick_nouns(wordfreq, top_nouns, min_freq, glove):
    """Frequent, groundable candidate nouns: not in NON_NOUN, has a GloVe vector,
    length > 2, above min_freq. Returns list sorted by frequency."""
    ranked = sorted(wordfreq.items(), key=lambda kv: -kv[1])
    out = []
    for w, f in ranked:
        if f < min_freq or len(w) <= 2 or w in NON_NOUN or w not in glove:
            continue
        out.append(w)
        if len(out) >= top_nouns:
            break
    return out


SENSE_PROMPT = """List the DISTINCT common meanings of the noun "{word}".
For each meaning give a short label and a concrete visual phrase that would let
an image generator draw exactly that meaning (never just the bare word).
Reply ONLY as JSON: [{{"label": "...", "term": "..."}}]. 1-4 meanings.
Example for "crane": [{{"label":"bird","term":"a crane bird with long legs"}},{{"label":"machine","term":"a yellow construction crane machine"}}]"""


def parse_senses(text, word):
    """Extract the JSON sense list from Gemma output; tolerate fenced/prefixed."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for s in arr:
        if isinstance(s, dict) and s.get("label") and s.get("term"):
            term = s["term"].strip()
            # guard the probe lesson: reject a bare-word term
            if term.lower() == word.lower():
                term = f"a {word}"
            out.append({"label": str(s["label"]).strip(), "term": term})
    return out or None


def tag_captions_to_senses(word, senses, captions, glove, window=6, max_refs=64):
    """For each caption containing `word`, assign it to the best-matching sense
    (context centroid vs sense-term centroid). Attach image_refs per sense."""
    term_vecs = [(s, centroid(WORD_RE.findall(s["term"].lower()), glove)) for s in senses]
    for s in senses:
        s["image_refs"] = []
    for cap in captions:
        words = WORD_RE.findall(cap["caption"].lower())
        if word not in words:
            continue
        i = words.index(word)
        ctx = words[max(0, i - window):i] + words[i + 1:i + 1 + window]
        cvec = centroid(ctx, glove)
        best, best_s = -2.0, senses[0]
        if cvec is not None:
            for s, tv in term_vecs:
                if tv is None:
                    continue
                sim = float(np.dot(cvec, tv))
                if sim > best:
                    best, best_s = sim, s
        if len(best_s["image_refs"]) < max_refs:
            best_s["image_refs"].append(cap.get("image_ref"))
    return senses


def gemma_enumerate(words, model_path):
    """Return {word: raw_gemma_text}. Reuses the working generate(**inputs) path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[senses] loading {model_path} ...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None).eval()
    out = {}
    for w in words:
        msgs = [{"role": "user", "content": SENSE_PROMPT.format(word=w)}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        ilen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        out[w] = tok.decode(gen[0][ilen:], skip_special_tokens=True)
    return out


def selftest(glove):
    print("[selftest] enumerate_senses_gemma (parse + caption tagging, no Gemma)")
    senses = parse_senses(
        'here: [{"label":"bird","term":"a crane bird with long legs"},'
        '{"label":"machine","term":"a construction crane machine lifting steel"}]', "crane")
    assert senses and len(senses) == 2, senses
    caps = [{"caption": "a crane flew over the lake", "image_ref": "img_bird"},
            {"caption": "a crane lifted heavy steel beams", "image_ref": "img_mach"}]
    tag_captions_to_senses("crane", senses, caps, glove)
    bird = next(s for s in senses if s["label"] == "bird")
    mach = next(s for s in senses if s["label"] == "machine")
    ok = "img_bird" in bird["image_refs"] and "img_mach" in mach["image_refs"]
    print(f"[senses] bird refs={bird['image_refs']} machine refs={mach['image_refs']}")
    print(f"[selftest] {'PASS -- captions routed to the right sense' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Gemma sense enumeration + caption tagging")
    p.add_argument("--corpus", default="data/coco_vsp",
                   help="dir with captions.jsonl + wordfreq.json (prepare_coco_vsp.py)")
    p.add_argument("--model", help="local Gemma 4 path (required unless --selftest)")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--top-nouns", type=int, default=300)
    p.add_argument("--min-freq", type=int, default=50)
    p.add_argument("--out", default="data/vsps/senses_coco.json")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        need = set("crane bird long legs construction machine lifting steel flew "
                   "over lake heavy beams the a".split())
        sys.exit(0 if selftest(load_glove(Path(args.glove), need)) else 1)

    if not args.model:
        raise SystemExit("--model (local Gemma path) required unless --selftest")

    cdir = Path(args.corpus)
    wordfreq = json.load(open(cdir / "wordfreq.json"))
    captions = [json.loads(l) for l in open(cdir / "captions.jsonl", encoding="utf-8")]

    # GloVe over caption vocab (for noun picking + context tagging)
    vocab = set(wordfreq.keys())
    glove = load_glove(Path(args.glove), vocab)
    nouns = pick_nouns(wordfreq, args.top_nouns, args.min_freq, glove)
    print(f"[senses] {len(nouns)} candidate nouns (top {args.top_nouns}, freq>={args.min_freq})")

    raw = gemma_enumerate(nouns, args.model)

    entries, n_senses, n_multi = [], 0, 0
    for w in nouns:
        senses = parse_senses(raw.get(w, ""), w)
        if not senses:
            continue
        tag_captions_to_senses(w, senses, captions, glove)
        senses = [s for s in senses if s["image_refs"]]  # keep senses seen in corpus
        if not senses:
            continue
        entries.append({"word": w, "senses": senses})
        n_senses += len(senses)
        if len(senses) >= 2:
            n_multi += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "coco+gemma", "entries": entries}, open(args.out, "w"), indent=2)
    print(f"[senses] {len(entries)} words, {n_senses} senses ({n_multi} polysemous); "
          f"saved -> {args.out}")


if __name__ == "__main__":
    main()
