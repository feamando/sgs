"""VSP v1, phase 0.2/0.3 (Wikipedia): sense inventory from disambiguation pages.

Switched from COCO to filtered Wikipedia (2026-07-06): COCO captions are visually
dense but people-centric and LOW-polysemy (top nouns man/woman/street/table) and
off-domain for Raum. Wikipedia has genuine polysemy AND the sense inventory comes
FOR FREE from disambiguation pages -- "Crane" -> {Crane (bird), Crane (machine),
Crane (surname)} -- each a real article. No Gemma enumeration and no caption
guessing needed: the parenthetical qualifier IS the sense label, title-labeled.

Pipeline:
  1. Load wikimedia/wikipedia (title + text) via the existing loader path.
  2. Find disambiguation pages; parse their sense links "Word (qualifier)".
  3. Keep words with >= 2 senses (the polysemy VSP targets). Build a concrete
     grounding phrase per sense from the qualifier ("a crane bird").
  4. Optionally pull each sense-article's Wikimedia lead IMAGE for native V
     (--with-images, REST API); else SD-generate from the phrase downstream.
  5. Emit senses_wiki.json in the shape build_vsps_vocab/derive_vsp_clip consume.

Also writes wordfreq.json over a domain-filtered article subset (for the
abstract vocab tier), Raum-relevant: keep articles whose title/lead hits an
object/nature/architecture/animal/vehicle lexicon.

Usage:
  python scripts/prepare_wikipedia_senses.py --hf-cache data/wikipedia/hf \
    --out data/vsps/senses_wiki.json --wordfreq-out data/wiki_vsp/wordfreq.json
  python scripts/prepare_wikipedia_senses.py --selftest
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WORD_RE = re.compile(r"[a-z]+")
# "Crane (bird)" / "Mercury (element)" -> (word, qualifier)
QUALIFIED = re.compile(r"^([A-Z][A-Za-z\-]+)\s*\(([^)]+)\)$")

# Raum-relevant domain seed lexicon (for the domain filter / abstract tier)
DOMAIN_HINTS = set(
    "building tower bridge castle house church temple wall roof dome arch column "
    "tree forest mountain river lake rock stone hill cliff valley "
    "animal bird fish mammal reptile insect "
    "vehicle car ship boat plane train aircraft engine machine tool "
    "metal wood glass stone water sand material".split())

DISAMBIG_MARKERS = ("may refer to:", "can refer to:", "disambiguation")


def is_disambiguation(title, text):
    t = (title or "").lower()
    if "(disambiguation)" in t:
        return True
    head = (text or "")[:400].lower()
    return any(m in head for m in DISAMBIG_MARKERS)


def parse_disambig_senses(title, text):
    """From a disambiguation page, return base word + [{label, term, article}].

    Senses come from qualified links like 'Crane (bird)'. We read them from the
    parenthetical qualifiers present in the page body."""
    base = re.sub(r"\s*\(disambiguation\)\s*$", "", title or "").strip()
    if not base:
        return None
    baseword = base.lower()
    senses, seen = [], set()
    # find 'Base (qualifier)' occurrences in the text
    for m in re.finditer(rf"{re.escape(base)}\s*\(([^)]+)\)", text or ""):
        qual = m.group(1).strip().lower()
        if qual in ("disambiguation",) or qual in seen or len(qual) < 2:
            continue
        seen.add(qual)
        # concrete grounding phrase: "a crane bird", "a mercury element"
        term = f"a {baseword} {qual}" if qual not in baseword else f"a {baseword}"
        senses.append({"label": qual.replace(" ", "-"), "term": term,
                       "article": f"{base} ({m.group(1).strip()})"})
    if len(senses) >= 2:
        return {"word": baseword, "senses": senses[:6]}
    return None


def selftest():
    print("[selftest] prepare_wikipedia_senses (disambiguation parse)")
    title = "Crane"
    text = ("Crane may refer to:\n"
            "Crane (bird), a family of large birds\n"
            "Crane (machine), a machine for lifting\n"
            "Crane (surname), a surname\n")
    assert is_disambiguation(title, text)
    got = parse_disambig_senses(title, text)
    labels = [s["label"] for s in got["senses"]] if got else []
    terms = [s["term"] for s in got["senses"]] if got else []
    ok = got and got["word"] == "crane" and set(labels) == {"bird", "machine", "surname"} \
        and "a crane bird" in terms
    print(f"[senses] word={got and got['word']} senses={labels}")
    print(f"[senses] terms={terms}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Wikipedia disambiguation -> VSP senses")
    p.add_argument("--hf-cache", default="data/wikipedia/hf")
    p.add_argument("--revision", default="20231101.en")
    p.add_argument("--max-articles", type=int, default=300000,
                   help="scan cap (default 300k ~= all disambiguation pages appear "
                        "early-ish; set 0 for the FULL ~6.4M dump, slow)")
    p.add_argument("--log-every", type=int, default=20000,
                   help="print progress every N articles scanned")
    p.add_argument("--out", default="data/vsps/senses_wiki.json")
    p.add_argument("--wordfreq-out", default="data/wiki_vsp/wordfreq.json")
    p.add_argument("--with-images", action="store_true",
                   help="fetch each sense-article's Wikimedia lead image for native V "
                        "(serial REST calls; SLOW, off by default; SD fallback works)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    from datasets import load_dataset
    print(f"[wiki] loading wikimedia/wikipedia ({args.revision}) from {args.hf_cache}...")
    ds = load_dataset("wikimedia/wikipedia", args.revision,
                      cache_dir=args.hf_cache, split="train", streaming=True)

    entries, wf = [], Counter()
    n_seen, n_disambig = 0, 0
    cap = args.max_articles  # None = full dump (~6.4M, slow); default is a cap
    import time
    t0 = time.time()
    for ex in ds:
        title, text = ex.get("title"), ex.get("text")
        n_seen += 1
        if is_disambiguation(title, text):
            n_disambig += 1
            e = parse_disambig_senses(title, text)
            if e:
                entries.append(e)
        else:
            head = (text or "")[:600].lower()
            if any(h in head for h in DOMAIN_HINTS):
                wf.update(w for w in WORD_RE.findall(head) if len(w) > 2)
        if n_seen % args.log_every == 0:
            rate = n_seen / max(time.time() - t0, 1e-6)
            print(f"[wiki] {n_seen:,} scanned | {n_disambig:,} disambig | "
                  f"{len(entries):,} polysemous words | {rate:,.0f} art/s", flush=True)
        if cap and n_seen >= cap:
            print(f"[wiki] hit --max-articles cap ({cap:,}); stopping scan.", flush=True)
            break

    if args.with_images:
        print(f"[wiki] fetching lead images for {len(entries)} words "
              f"(serial REST calls; use --no-images to skip)...", flush=True)
        _attach_lead_images(entries, log_every=50)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "wikipedia-disambiguation", "entries": entries},
              open(args.out, "w"), indent=2)
    Path(args.wordfreq_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(wf.most_common()), open(args.wordfreq_out, "w"))
    n_multi = sum(1 for e in entries if len(e["senses"]) >= 2)
    print(f"[wiki] {n_seen} articles, {n_disambig} disambiguation pages -> "
          f"{len(entries)} polysemous words ({n_multi} with >=2 senses)")
    print(f"[wiki] abstract-tier wordfreq: {len(wf)} words -> {args.wordfreq_out}")
    print(f"[wiki] senses -> {args.out}")


def _attach_lead_images(entries, log_every=50):
    """Fetch each sense-article's lead image URL via the Wikimedia REST summary
    API. Best-effort; senses without an image fall back to SD generation later."""
    import urllib.request, urllib.parse
    base = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    done, hits = 0, 0
    for e in entries:
        for s in e["senses"]:
            art = s.get("article")
            if not art:
                continue
            done += 1
            try:
                url = base + urllib.parse.quote(art.replace(" ", "_"))
                req = urllib.request.Request(url, headers={"User-Agent": "sgs-vsp/1.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.load(r)
                img = (data.get("originalimage") or data.get("thumbnail") or {}).get("source")
                if img:
                    s["image_refs"] = [img]; hits += 1
            except Exception:
                pass  # SD fallback downstream
            if done % log_every == 0:
                print(f"[wiki]   images: {done} senses queried, {hits} with a photo",
                      flush=True)
    print(f"[wiki]   images done: {hits}/{done} senses got a lead photo", flush=True)


if __name__ == "__main__":
    main()
