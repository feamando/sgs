# SGS VSP v1 — deliver Planck 2.0 (first VSP model), July 2026

*Windows + RTX 4090. Other platforms not validated.*

**Objective:** ship **Planck 2.0**, the first SGS language model whose token
embedding is a bundled **V/S/P** vector (Visual + Semantic + Physical) instead of
a text-only lookup row. This doc is the build plan from a PASSED grounding gate
through tokenizer → training → the disambiguation benchmark that proves the win.

Origin: the VSP grounding gate PASSED 2026-07-06 (see below, and
[[project_sgs_vsp_gate]]). This supersedes the exploratory §2-4 of
`SETUP_202607_path2.md` (which stays as the gate/probe record). Companion:
[[project_sgs_path1_outcome]] (path1 showed base-scaling saturates on a fixed
task; representation is the lever, hence VSP).

DISCIPLINE: gate-and-kill, same as the probe. Prove each step cheaply before the
next. Kill fast if a gate fails; don't build the trainer for a tokenizer that
doesn't round-trip.

## 0. What we KNOW from the VSP experiment (carry these forward)

These are settled findings; the build must respect them.

1. **The representation separates senses; text alone does not.** Gate result:
   text-only 0.13 (FAIL) → auto visual+physical 0.37 (PASS), max-agg over V,S,P,
   20 polysemous words, nothing hand-labeled.
2. **A token is a SENSE, not a word.** crane-bird and crane-machine are two
   tokens. Word-level tokens collapse polysemy (every text regime failed). Sense
   discovery = Gemma-enumerate candidate senses → SD-image per sense → CLIP-embed
   → DEDUP in embedding space (cosine ≥ thr = same sense). NOT a subword splitter.
3. **V = CLIP-image, auto-derived.** SD generates an image per sense PHRASE, CLIP
   image-encodes, mean over views. Open-vocabulary, no ShapeNet ceiling. NOT a
   synset one-hot (tautological), NOT a text vector (collapses).
4. **P = derived, not looked up.** P6 physics MLP (GloVe→8 material props, trained
   on MATERIAL_TABLE) predicts P from the sense phrase. `derive_p6_from_vector.py`.
   Do NOT reintroduce the hand material-tag table — it's a tautology and P must
   stay auto to keep the "one vector, automatically" claim honest.
5. **S = GloVe** (later: the model's own learned S). Identical across senses of
   the same surface word — that's the collapse the bundle fixes.
6. **Separation metric = MAX-aggregation over blocks**, not concat/mean. A sense
   pair is separable if ANY modality distinguishes it (grey fur ball vs grey
   boulder: V collapses, P/S separate). Concat-of-unit-blocks == mean-of-cosines,
   which lets an identical block dilute/veto an informative one. Use
   `vsp_gating_probe.py --aggregate max`.
7. **Separation is carried mostly by V; derived-P is a small honest bonus.** Don't
   overclaim P. If a sense has no groundable image, it falls back toward S-only.
8. **Prompt hygiene matters.** SD sense phrases MUST be concrete + unambiguous
   ("a crane bird with long legs", not "crane") or SD draws its dominant prior
   for every sense and dedup merges them. This is the #1 data bug.
9. **Abstract senses (justice, the, however) are correctly S-only.** No V/P is a
   feature, not a gap. Two-tier vocab (grounded vs abstract).

## 0.1 Environment

```powershell
cd sgs
# tokenizer + vocab build: main .venv (torch 2.6). Grounding (SD/CLIP): .venv-sds.
.venv\Scripts\Activate.ps1
python -c "import torch, sentencepiece; print('base ok')"
# for grounding at corpus scale (SD + CLIP), switch to .venv-sds as in path2 §0.
```

## 0.2 Corpus: filtered Wikipedia (disambiguation senses), NOT COCO/TinyStories

Decision history:
- TinyStories rejected: deliberately SIMPLE vocab = no polysemy to disambiguate.
- COCO captions TRIED then rejected (2026-07-06): visually dense with native
  images, BUT people-centric and LOW-polysemy (200k captions, only 16.9k unique
  words; top nouns man/woman/street/table). It doesn't CONTAIN both senses of the
  words VSP targets, and it's off-domain for Raum. (COCO scripts kept, reusable.)
- **Wikipedia CHOSEN**: genuine polysemy (an encyclopedia has crane-bird AND
  crane-machine), concrete/material/structural vocab (Raum's domain when
  filtered), and the sense inventory is FREE + title-labeled. Wikipedia's
  DISAMBIGUATION PAGES enumerate senses directly: "Crane" -> {Crane (bird),
  Crane (machine), Crane (surname)}, each its own article with its own lead
  image. No Gemma enumeration, no caption sense-guessing; native V is
  BETTER-labeled than COCO (the article title states the sense).

## 0.3 Sense inventory + V/P at corpus scale (Wikipedia pipeline)

1. **Sense inventory (FREE, no Gemma)**: `prepare_wikipedia_senses.py` scans the
   dump, finds disambiguation pages, parses "Word (qualifier)" links -> per-word
   senses with a concrete grounding phrase ("a crane bird"). Keeps words with
   >=2 senses. Also writes a domain-filtered wordfreq for the abstract tier.
2. **V**: native lead image per sense-article via Wikimedia REST (`--with-images`),
   CLIP-embed; SD-generate from the phrase as fallback (derive_vsp_clip.py, the
   proven probe path). Non-visual senses (crane=surname) get zero-V -> S-only,
   correct.
3. **P, derived**: derive_p6_from_vector.py -- P6 MLP from the sense phrase. NO
   hand material table (the tautology lesson).
4. **S**: GloVe (later: the model's own learned S).
5. **Cache** V/S/P per sense token so training reads a lookup, not a live pass.

```powershell
# 1. sense inventory + abstract-tier wordfreq (prepare_wikipedia_senses.py BUILT)
python scripts/prepare_wikipedia_senses.py --hf-cache data/wikipedia/hf `
  --with-images `                              # native Wikimedia lead images for V
  --out data/vsps/senses_wiki.json --wordfreq-out data/wiki_vsp/wordfreq.json
# 2. V (native images if present, else SD), then derived P:
python scripts/derive_vsp_clip.py --senses data/vsps/senses_wiki.json `
  --v-source clip-image --save-views results/vsp_views_wiki `
  --out data/vsps/wiki_clip.json               # (extend to embed image_refs when present)
python scripts/derive_p6_from_vector.py --in data/vsps/wiki_clip.json `
  --glove data/glove.6B.300d.txt --out data/vsps/wiki_pderiv.json
```

(Gemma is NOT needed for Wikipedia sense enumeration; enumerate_senses_gemma.py
and the COCO prep scripts stay for caption corpora if that path is revisited.)

## 1. VSPS vocabulary (TO BUILD: scripts/build_vsps_vocab.py)

Mint a two-tier vocabulary. Reuse the gate's grounding pipeline at corpus scale.

- **Grounded tokens** — one per concrete SENSE, carrying (CLIP-image V, derived P,
  GloVe S). Senses come from Wikipedia disambiguation pages (§0.3), NOT a splitter.
- **Abstract tokens** — function words / abstractions, S-only, no V/P.

```powershell
# corpus-scale senses (Wikipedia) for grounded tier; wordfreq for abstract tier
python scripts/build_vsps_vocab.py `
  --senses data/vsps/wiki_pderiv.json `            # CLIP-image V (native photos) + DERIVED P
  --glove data/glove.6B.300d.txt `                 # S
  --corpus-vocab data/wiki_vsp/wordfreq.json `     # abstract (S-only) tier
  --out data/vsps/vocab.json
# (build_vsps_vocab.py is BUILT + selftested; ran on the 20-word probe = 44 tokens)
```

GATE: grounded vocab covers the common concrete-noun space at a manageable blowup
(most words 1-3 senses → nouns expand ~1-3x; abstract vocab unchanged). Measure
actual coverage on the Wikipedia sense inventory before assuming. Open item:
groundable words with no lead image AND no SD asset → S-only fallback.

## 2. VSPS tokenize the corpus (tokenize_vsps.py — BUILT)

Sense-tag the Wikipedia corpus (genuine polysemy, Raum-domain) before Planck 2.0.

- **Sense-tag each grounded word occurrence**: embedding-Lesk — pick the sense
  whose descriptive term best matches the sentence context. WSD at tokenize time,
  the step SentencePiece skips. (BUILT + selftested: 5/5 on minimal pairs.)
- **Cache V/S/P per token** so training reads a lookup, not a live CLIP pass.

```powershell
python scripts/tokenize_vsps.py --corpus data/wiki_vsp `
  --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt `
  --out data/wiki_vsps
# NOTE: load GloVe over the FULL corpus vocab so context words contribute to WSD
# (the probe run only loaded term+surface words).
```

GATE: VSPS round-trips Wikipedia text, and polysemous words get SENSE-CORRECT
tokens (the disambiguation the thesis promises) where SentencePiece merged them.

## 3. Planck 2.0 training (TO BUILD: scripts/train_planck2.py)

Adapt `train_planck11.py` / the SGSLanguageModel scaffold so each token embedding
IS the V/S/P bundle, not S alone.

1. **Embedding = concat(V, S, P)** (or a learned projection of it), initialized
   from the cached bundles. V frozen (CLIP-image; don't backprop into generated
   appearance early). S/P fine-tunable.
2. **Frozen-V warm start** (mirrors Raum freezing the Planck encoder): learn to
   USE the grounded representation before it can corrupt it; unfreeze later.
3. **Plain AdamW, stdout logging** ([[project_sgs_accel_shelved]],
   [[feedback_sgs_wandb_default]] — accel shelved, wandb is paid).
4. **Hard opt-step budget + resume that doesn't restart the epoch**
   ([[project_hertz_resume_epoch_restart]] — reuse train_hertz.py's fix; don't
   rely on loader exhaustion as the stop).
5. **Arch loads via infer_arch** if resuming from any SGS checkpoint
   ([[project_sgs_path1_outcome]] — Planck/Hertz shapes differ; never hardcode).

```powershell
python scripts/train_planck2.py --data data/wiki_vsps `
  --vocab data/vsps/vocab.json --d-f 1000 --freeze-vp `
  --save-dir checkpoints/planck2 --tokens 1B   # TO BUILD; mirror train_planck11.py
```

## 4. The gate that proves the win (TO BUILD: disambiguation benchmark)

**Val loss will NOT show the win.** Build a polysemy benchmark: sense-correct
next-token / retrieval on minimal pairs ("the crane flew" vs "the crane lifted",
"sat on the river bank" vs "money in the bank").

GATE (the publishable result): Planck 2.0 beats a SentencePiece-baseline Planck
on the disambiguation benchmark **at matched params and tokens**. That delta is
the paper. If no delta at matched compute → the bundle didn't help the model
(distinct from "the representation separates", which is already proven); stop and
diagnose whether it's the embedding wiring or frozen-V starving the model.

## Sequencing (gate-and-kill)

| Phase | What | Status / kill-if |
|-------|------|------------------|
| 0 | grounding gate (V+S+P separate senses) | **PASS 2026-07-06, 0.37** (auto V + derived P) |
| 0.2 | Wikipedia corpus + disambiguation senses | **DONE (sample): 300k articles -> 7,678 polysemous words** at ~4k art/s. Full dump default now (~25-30 min). COCO tried+rejected (low polysemy). |
| 0.3 | native V from Wikimedia lead images | wire derive_vsp_clip.py to embed image_refs (SD fallback already works); KILL if disambiguation sense yield too low |
| 1 | VSPS vocab (two-tier) | **build_vsps_vocab.py BUILT + selftested** (probe: 44 tokens, 2x blowup). Run on Wikipedia senses. |
| 2 | VSPS tokenize Wikipedia | **tokenize_vsps.py BUILT** (5/5 minimal pairs). Run on Wikipedia; load GloVe over full corpus vocab. |
| 3 | Planck 2.0 training | TO BUILD train_planck2.py. plain AdamW, frozen-V warm start, hard step budget |
| 4 | disambiguation benchmark | TO BUILD. KILL claim if no win vs SentencePiece baseline at matched compute |

## Papers this feeds

- **VSPS Tokenization** — phases 1-2 (the sense-token + grounding pipeline).
- **VSP-based Models** — phase 4 (extends the Physical Gaussians paper).
- (Alpha-Compositing > Softmax already at JMLR, [[project_sgs_jmlr_submission]], not this.)

## Honest scope reminder

The gate proved the REPRESENTATION separates senses on a 20-word probe. It did
NOT prove a trained model benefits. Phases 1-4 are unbuilt. The real result is
phase 4's delta; everything before it is plumbing toward that measurement.
