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

## 1. VSPS vocabulary (TO BUILD: scripts/build_vsps_vocab.py)

Mint a two-tier vocabulary. Reuse the gate's grounding pipeline at corpus scale.

- **Grounded tokens** — one per concrete SENSE, carrying (CLIP-image V, derived P,
  GloVe S). Senses come from Gemma-enumerate + CLIP-image-dedup, NOT a splitter.
- **Abstract tokens** — function words / abstractions, S-only, no V/P.

```powershell
python scripts/build_vsps_vocab.py `
  --senses results/vsp_clip_image_pderiv.json `   # CLIP-image V + DERIVED P + dedup'd senses
  --glove data/glove.6B.300d.txt `                 # S
  --out data/vsps/vocab.json   # TO BUILD
```

GATE: grounded vocab covers the common concrete-noun space at a manageable blowup
(most words 1-3 senses → nouns expand ~1-3x; abstract vocab unchanged). Measure
actual coverage on TinyStories before assuming. Open item: groundable words with
no generated asset yet → S-only fallback or generate on demand.

## 2. VSPS tokenize a corpus (TO BUILD: scripts/tokenize_vsps.py)

Validate on TinyStories (small, clean, concrete nouns) before Planck 2.0.

- **Sense-tag each grounded word occurrence**: assign the dedup'd sense whose
  V/S/P best matches context. This is WSD at tokenize time — the step
  SentencePiece skips and why it collapses senses.
- **Cache V/S/P per token** so training reads a lookup, not a live SD+CLIP pass
  (generation is the expensive step; do it ONCE at vocab-build).

```powershell
python scripts/tokenize_vsps.py --corpus data/tinystories `
  --vocab data/vsps/vocab.json --out data/tinystories_vsps   # TO BUILD
```

GATE: VSPS round-trips TinyStories, and polysemous words get SENSE-CORRECT tokens
(the disambiguation the thesis promises) where SentencePiece merged them.

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
python scripts/train_planck2.py --data data/tinystories_vsps `
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
| 1 | VSPS vocab (two-tier, corpus-scale senses) | KILL if grounding coverage too thin or vocab blowup explodes |
| 2 | VSPS tokenize TinyStories | KILL if it doesn't round-trip or WSD is wrong |
| 3 | Planck 2.0 training | plain AdamW, frozen-V warm start, hard step budget |
| 4 | disambiguation benchmark | KILL claim if no win vs SentencePiece baseline at matched compute |

## Papers this feeds

- **VSPS Tokenization** — phases 1-2 (the sense-token + grounding pipeline).
- **VSP-based Models** — phase 4 (extends the Physical Gaussians paper).
- (Alpha-Compositing > Softmax already at JMLR, [[project_sgs_jmlr_submission]], not this.)

## Honest scope reminder

The gate proved the REPRESENTATION separates senses on a 20-word probe. It did
NOT prove a trained model benefits. Phases 1-4 are unbuilt. The real result is
phase 4's delta; everything before it is plumbing toward that measurement.
