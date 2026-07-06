"""
Raum Path B: CLIP/image-grounded V with sense discovery by dedup.

The synset-grounded probe (ground_vsp.py) PASSED but its V was a synset one-hot
(separates by construction) and capped at ShapeNet's ~30 categories. This module
tests the open-vocabulary alternative the user proposed:

  V = CLIP embedding of GENERATED VIEWS of a sense, NOT a synset id and NOT the
  colliding word's text vector. Open vocabulary -- any sense you can describe,
  no ShapeNet ceiling.

Sense discovery by DEDUP (the user's idea): instead of a human writing
"crane -> {bird, machine}", enumerate candidate senses, embed each, and keep
only senses that occupy a DISTINCT region of embedding space. Once "crane-bird"
is mapped, the next "crane" sample must be far enough away (cosine < --dedup-thr)
to count as a new sense; otherwise it's the same sense from another angle.
Curation moves from human to a distance threshold.

Pipeline on the box (4090):
  Gemma enumerates senses (text) -> Stable Diffusion renders N views per sense
  -> CLIP image-encodes the views -> mean = the sense's V -> dedup across senses.

V source is abstracted by --v-source:
  clip-image  : real pipeline above (needs SD + CLIP + GPU). The honest version.
  clip-text   : CPU stand-in -- CLIP TEXT-encode a disambiguating phrase per
                sense. Validates the plumbing + dedup logic anywhere; an upper
                bound on text, a lower bound on image (text still can collide).

P stays the P6 measured-material path (ground_vsp) -- orthogonal to V source.

Usage:
  # CPU plumbing/dedup test (CLIP text stand-in):
  python scripts/derive_vsp_clip.py --senses scripts/assets/vsp_sense_terms.json \
    --v-source clip-text --out results/vsp_clip.json
  # real image-grounded V (box): add --v-source clip-image --gen-model <sd path>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_p6_correlation import MATERIAL_TABLE, PROPERTY_NAMES


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _feat(out):
    """Normalize CLIP get_*_features return across transformers versions: some
    return a [B, d] tensor, others a BaseModelOutputWithPooling -> .pooler_output."""
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    if hasattr(out, "shape"):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    raise TypeError(f"unexpected CLIP feature output: {type(out)}")


# ── V sources ──────────────────────────────────────────────────────────

def clip_text_vectors(phrases, device="cpu"):
    """CPU stand-in: CLIP TEXT embedding per phrase. Same encoder family as the
    image path, so distances are comparable; but text can still collide, so this
    is a LOWER bound on what image-grounding achieves."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    out = {}
    with torch.no_grad():
        for ph in phrases:
            inp = proc(text=[ph], return_tensors="pt", padding=True).to(device)
            v = _feat(model.get_text_features(**inp))[0].cpu().numpy().astype(np.float32)
            out[ph] = _unit(v)
    return out


def clip_image_vectors(phrases, gen_model, n_views, device="cuda", save_views=None):
    """Real path (box): SD-generate n_views per phrase, CLIP image-encode, mean.
    Open-vocabulary V grounded in generated appearance, not text or a synset.

    save_views: if set, write every generated image to that dir so the SD output
    can be eyeballed before trusting the embeddings (garbage views -> garbage V).
    """
    import torch
    import re
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPModel, CLIPProcessor
    sd = StableDiffusionPipeline.from_pretrained(gen_model, torch_dtype=torch.float16).to(device)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    views = ["", " side view", " front view", " 3/4 view"][:n_views]
    if save_views:
        Path(save_views).mkdir(parents=True, exist_ok=True)
        print(f"[clip] saving generated views to {save_views}/")
    out = {}
    for ph in phrases:
        embs = []
        for vi, vsuffix in enumerate(views):
            img = sd(ph + vsuffix, num_inference_steps=25).images[0]
            if save_views:
                slug = re.sub(r"[^a-z0-9]+", "_", ph.lower()).strip("_")
                img.save(Path(save_views) / f"{slug}__view{vi}.png")
            with torch.no_grad():
                inp = proc(images=img, return_tensors="pt").to(device)
                e = _feat(clip.get_image_features(**inp))[0].cpu().numpy().astype(np.float32)
            embs.append(_unit(e))
        out[ph] = _unit(np.mean(embs, axis=0))
    return out


# ── sense discovery by dedup ───────────────────────────────────────────

def dedup_senses(sense_list, vecs, thr):
    """Keep senses whose V is far (cosine < thr) from all already-kept senses of
    the SAME word. Models the user's idea: once a sense's region is occupied,
    later samples must separate to count as new. Returns kept senses + a report."""
    kept, report = [], []
    for s in sense_list:
        v = vecs.get(s["term"]) if "term" in s else vecs.get(s["label"])
        if v is None:
            report.append((s["label"], "no-vector", None)); continue
        collide = None
        for k in kept:
            sim = float(np.dot(_unit(v), _unit(k["_v"])))
            if sim >= thr:
                collide = (k["label"], round(sim, 3)); break
        if collide:
            report.append((s["label"], f"merged-into:{collide[0]}", collide[1]))
        else:
            s = dict(s); s["_v"] = v; kept.append(s)
            report.append((s["label"], "kept", None))
    return kept, report


def phys_vector(material):
    if material and material in MATERIAL_TABLE:
        return np.array(MATERIAL_TABLE[material], dtype=np.float32)
    return np.zeros(len(PROPERTY_NAMES), dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="CLIP/image-grounded V + sense dedup")
    p.add_argument("--senses", default="scripts/assets/vsp_sense_terms.json")
    p.add_argument("--v-source", choices=["clip-text", "clip-image"], default="clip-text")
    p.add_argument("--gen-model", default="runwayml/stable-diffusion-v1-5",
                   help="SD model for clip-image view generation (box only)")
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--dedup-thr", type=float, default=0.85,
                   help="cosine >= thr => same sense (merge); the discovery knob")
    p.add_argument("--material-map", default="scripts/assets/vsp_grounded_map.json",
                   help="reuse the curated material per sense for P (V is the experiment)")
    p.add_argument("--save-views", default=None,
                   help="dir to dump generated SD images (clip-image only) so you "
                        "can eyeball view quality before trusting the embeddings")
    p.add_argument("--limit-words", type=int, default=None,
                   help="ground only the top-N most frequent polysemous words "
                        "(SCALE GUARD; the rest fall to S-only). Needs --wordfreq.")
    p.add_argument("--wordfreq", default=None,
                   help="{word: freq} json to rank words for --limit-words")
    p.add_argument("--out", default="results/vsp_clip.json")
    args = p.parse_args()

    entries = json.load(open(args.senses))
    if isinstance(entries, dict) and "entries" in entries:
        entries = entries["entries"]

    # SCALE GUARD: grounding = ~n_views SD generations PER SENSE. The full
    # Wikipedia dump has ~150k polysemous words (~450k senses ~= days of GPU).
    # Ground only the top-N most FREQUENT words (--limit-words), ranked by the
    # corpus wordfreq. The rest fall to S-only, which is correct for rare words.
    if args.limit_words and len(entries) > args.limit_words:
        rank = {}
        if args.wordfreq and Path(args.wordfreq).exists():
            rank = json.load(open(args.wordfreq))
        entries.sort(key=lambda e: -rank.get(e["word"], 0))
        dropped = len(entries) - args.limit_words
        entries = entries[:args.limit_words]
        n_sense = sum(len(e["senses"]) for e in entries)
        print(f"[clip] limit-words {args.limit_words}: grounding {len(entries)} words "
              f"/ {n_sense} senses (~{n_sense*args.n_views} SD images); "
              f"dropped {dropped} rarer words -> S-only.")

    # material lookup by (word,label) from the grounded map, for P
    matmap = {}
    if Path(args.material_map).exists():
        gm = json.load(open(args.material_map))
        for e in gm["entries"]:
            for s in e["senses"]:
                matmap[(e["word"], s["label"])] = s.get("material")

    # gather all disambiguating phrases (term = sense-specific phrase)
    phrases = sorted({s["term"] for e in entries for s in e["senses"]})
    print(f"[clip] V source: {args.v_source}, {len(phrases)} sense phrases, dedup_thr={args.dedup_thr}")
    device = "cuda" if args.v_source == "clip-image" else "cpu"
    if args.v_source == "clip-image":
        vecs = clip_image_vectors(phrases, args.gen_model, args.n_views, device,
                                  save_views=args.save_views)
    else:
        vecs = clip_text_vectors(phrases, device)

    out, total_kept, total_merged = [], 0, 0
    for e in entries:
        kept, report = dedup_senses(e["senses"], vecs, args.dedup_thr)
        for label, status, sim in report:
            if status.startswith("merged"):
                total_merged += 1
                print(f"   {e['word']}/{label}: {status} (sim {sim})")
        total_kept += len(kept)
        senses_out = []
        for s in kept:
            V = _unit(s["_v"])
            mat = matmap.get((e["word"], s["label"]))
            P = phys_vector(mat)
            senses_out.append({
                "label": s["label"], "term": s.get("term"), "material": mat,
                "visual_dist": [round(float(x), 4) for x in V],
                "physical": {n: round(float(v), 4) for n, v in zip(PROPERTY_NAMES, P)},
            })
        if len(senses_out) >= 2:
            out.append({"word": e["word"], "senses": senses_out})

    print(f"[clip] kept {total_kept} senses, {total_merged} merged by dedup; "
          f"{len(out)} words retain >=2 distinct senses")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"v_source": args.v_source, "dedup_thr": args.dedup_thr, "entries": out},
              open(args.out, "w"), indent=2)
    print(f"[clip] saved -> {args.out}")
    print(f"[clip] next: vsp_gating_probe.py --derived {args.out}")


if __name__ == "__main__":
    main()
