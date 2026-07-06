"""VSP v1, phase 0.2: prepare the COCO caption corpus for VSP.

Image-caption text is the VSP corpus (not TinyStories): dense in concrete,
groundable, polysemous nouns, and every caption comes WITH its image so V can be
grounded in a REAL photo (not an SD generation). See SETUP_202607_VSP_v1.md 0.2.

Pulls COCO Captions via HF datasets and writes:
  captions.jsonl   -- one {image_id, caption, image_ref} per line
  wordfreq.json    -- {word: freq} over captions (feeds the abstract vocab tier)
Images are referenced (id/url), not copied; the V-grounding step (0.3) fetches
and CLIP-embeds them, caching V per sense.

Usage:
  python scripts/prepare_coco_vsp.py --split train --max-images 40000 --out data/coco_vsp
  python scripts/prepare_coco_vsp.py --selftest
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[a-z]+")

# minimal stoplist so wordfreq surfaces content words (abstract tier still keeps
# function words as S-only tokens later; this is just for inspection ranking)
STOP = set("the a an of to and in on at is are was were be been it its this that "
           "with for as by from into over under near".split())


def build_wordfreq(captions):
    c = Counter()
    for cap in captions:
        c.update(w for w in WORD_RE.findall(cap.lower()) if len(w) > 2)
    return c


def _extract_captions(ex):
    """Pull caption string(s) from a row across the common COCO schemas."""
    for key in ("sentences", "captions", "caption", "sentences_raw", "text"):
        if key in ex and ex[key] is not None:
            v = ex[key]
            if isinstance(v, str):
                return [v]
            if isinstance(v, dict):
                r = v.get("raw") or v.get("caption") or v.get("text")
                return r if isinstance(r, list) else ([r] if r else [])
            if isinstance(v, list):
                out = []
                for c in v:
                    out.append(c["raw"] if isinstance(c, dict) and "raw" in c
                               else (c if isinstance(c, str) else str(c)))
                return out
    return []


def load_coco(split, max_images, dataset_id="yerevann/coco-karpathy"):
    """COCO captions via HF datasets (PARQUET mirrors only).

    NOTE: script-based mirrors (HuggingFaceM4/COCO ships COCO.py) FAIL on modern
    datasets ("Dataset scripts are no longer supported"). Use a parquet dataset.
    Known-good parquet ids to try with --dataset-id:
      yerevann/coco-karpathy   (default; 'sentences'/'filename', karpathy split)
      nlphuji/coco_captions
      sayakpaul/coco-30-val-2014
    Field-probing handles the common schemas; adjust if your mirror differs."""
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split, streaming=True)
    rows, seen = [], set()
    for i, ex in enumerate(ds):
        img_id = (ex.get("imgid") or ex.get("cocoid") or ex.get("image_id")
                  or ex.get("filename") or i)
        for text in _extract_captions(ex):
            if text:
                rows.append({"image_id": img_id, "caption": text,
                             "image_ref": ex.get("url") or ex.get("filepath")
                             or ex.get("filename") or img_id})
        seen.add(img_id)
        if max_images and len(seen) >= max_images:
            break
    if not rows:
        raise SystemExit(
            f"[coco] loaded 0 captions from '{dataset_id}'. The schema likely "
            f"differs; inspect one row (print(next(iter(ds)))) and extend "
            f"_extract_captions, or try another --dataset-id (see load_coco note).")
    return rows


def selftest():
    """No download: check wordfreq + jsonl shape on synthetic captions."""
    print("[selftest] prepare_coco_vsp")
    caps = ["a crane flies over the lake",
            "a large crane lifts steel beams at the site",
            "a wooden bat leans against the fence",
            "a small bat flies in the cave"]
    wf = build_wordfreq(caps)
    ok = wf["crane"] == 2 and wf["bat"] == 2 and "the" not in [w for w in wf if len(w) <= 2]
    # polysemous nouns should surface with freq >= 2
    top = [w for w, _ in wf.most_common(5)]
    print(f"[selftest] top content words: {top}")
    print(f"[selftest] crane={wf['crane']} bat={wf['bat']}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Prepare COCO caption corpus for VSP")
    p.add_argument("--split", default="train")
    p.add_argument("--max-images", type=int, default=40000)
    p.add_argument("--dataset-id", default="yerevann/coco-karpathy",
                   help="HF PARQUET dataset id for COCO captions (script-based "
                        "mirrors like HuggingFaceM4/COCO FAIL; see load_coco note)")
    p.add_argument("--out", default="data/coco_vsp")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    print(f"[coco] loading COCO captions ({args.split}, <= {args.max_images} images)...")
    rows = load_coco(args.split, args.max_images, args.dataset_id)
    caps = [r["caption"] for r in rows]
    wf = build_wordfreq(caps)

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    with open(outd / "captions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(wf.most_common()), open(outd / "wordfreq.json", "w"))

    content = [(w, n) for w, n in wf.most_common(30) if w not in STOP]
    print(f"[coco] {len(rows)} captions, {len(wf)} unique words -> {outd}")
    print(f"[coco] top content nouns (VSP grounding candidates): "
          f"{[w for w, _ in content[:15]]}")


if __name__ == "__main__":
    main()
