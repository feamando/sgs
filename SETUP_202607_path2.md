# SGS — Path B (Planck 2.0, VSP architecture) Setup, July 2026

*Windows + RTX 4090. Other platforms not validated.*

Path B is the NEW-architecture experiment: a ~100M model (Planck 2.0) built on a
VSP object representation, where each SGS object bundles **Visual + Semantic +
Physical** components as ONE representation rather than three matrices bound
after the fact. Disambiguation falls out for free: "bank (river)" and "bank
(institution)" become DISTINCT tokens because their V (water vs building) and P
(none vs concrete) differ at identical text. This directly fixes the polysemy
collapse that FAILED Raum 1.2 (SentencePiece merged related words, 35 collision
groups / 300 classes).

Roadmap rows: `1-planck-2-0` (model), `12-vsps-0-1` (tokenization). Success ->
two publications (VSP-based Models, VSPS Tokenization). Runs parallel to Path A.
Origin: 2026-06-22 planning, see [[project_sgs_post17_directions]].

DISCIPLINE: this is a high-uncertainty bet. Gate each step like Raum 1.7 Stage 1
(prove reachability cheaply before committing training). Kill fast if a gate fails.

## 0. General setup

Reuse the main `.venv` (torch 2.6 / py3.12). Seed assets for V and P already
exist in the tree:

```powershell
cd sgs
.venv\Scripts\Activate.ps1
# V (visual): Objaverse/ShapeNet blob libraries
python -c "import os; print('shapenet blobs:', os.path.exists('data/blobs_shapenet'))"
# P (physical): Physical Gaussians P6 material work (hardness R^2=0.54)
ls docs/papers/physical_gaussians*.md
```

## 1. VSP gating experiment (the reachability probe -- DO THIS FIRST)

The cheapest test of the whole thesis. ~1 week, NO training. If V/S/P similarity
does not separate senses that S-only collapses, the VSP idea is not real and
Planck 2.0 should not be trained.

Procedure:
1. Pick ~20 polysemous words (bank, plane, bat, spring, crane, mouse, ...).
2. For each SENSE, hand-build a V/S/P vector from existing assets:
   - **S**: embedding from the current Planck/GloVe path
   - **V**: nearest Objaverse/ShapeNet blob for that sense (water vs building)
   - **P**: Physical-Gaussians P6 material signature (none vs concrete)
3. Compute pairwise similarity two ways: S-only, and full V/S/P concat.
4. GATE: V/S/P similarity SEPARATES the senses (bank-river far from
   bank-institution) where S-only COLLAPSES them (close/identical).

```powershell
python scripts/vsp_gating_probe.py --words data/vsp/polysemous_20.json `
  --out results/vsp_gating.json   # TO BUILD
```

PASS = senses separate under V/S/P, collapse under S-only -> VSP is real, proceed.
FAIL = no separation -> stop; the binding does not buy disambiguation, rethink.

## 2. VSPS tokenization experiment

If the gating probe passes, design the tokenizer. VSPS = word/concept-level
tokens carrying V/S/P, NOT compression-optimal subwords. Two-tier vocabulary:

- **Grounded tokens**: concrete nouns with V/S/P bundles (one token per SENSE)
- **Abstract tokens**: function words / abstractions, S-only (no grounding)

Experiment: build the grounded vocab from the intersection of what can be
grounded (words with both an Objaverse/ShapeNet V and a P6 P signature), measure
coverage and the vocab-size blowup (expected 1-3x on grounded nouns, since most
words have 1-3 grounded senses).

```powershell
python scripts/build_vsps_vocab.py --visual data/blobs_shapenet `
  --physical results/p6_materials.json --out data/vsps/vocab.json   # TO BUILD
```

GATE: grounded vocab covers the common concrete-noun space at a manageable
blowup; minting a token = looking up its V and P (the open question: what to do
for groundable words with no asset yet).

## 3. VSPS tokenization run on TinyStories

Validate the tokenizer on a small, clean corpus before Planck 2.0. TinyStories
(src/tinystories.py exists) is the right scale: simple vocabulary, concrete
nouns, fast iteration.

```powershell
# tokenize TinyStories with VSPS, compare against the SentencePiece baseline
python scripts/tokenize_vsps.py --corpus data/tinystories `
  --vocab data/vsps/vocab.json --out data/tinystories_vsps   # TO BUILD
```

GATE: VSPS tokenization round-trips TinyStories, and polysemous words in the
corpus get sense-correct tokens (the disambiguation the whole thesis promises),
where SentencePiece merged them.

## 4. Planck 2.0 training

Only after 1-3 pass. Train a ~100M SGS LM on VSPS-tokenized data with the VSP
object representation. Same SGSLanguageModel scaffold (src/sgs_lm.py) adapted so
token embeddings carry V/S/P, not just S.

```powershell
python scripts/train_planck2.py --data data/tinystories_vsps `
  --vocab data/vsps/vocab.json --d-f 1000 --save-dir checkpoints/planck2 `
  --tokens 1B   # TO BUILD; mirror train_lm.py / train_planck11.py
```

GATE: Planck 2.0 beats the SentencePiece-baseline Planck on a disambiguation
benchmark (sense-correct retrieval / generation on polysemous prompts), at
matched params and tokens. That delta is the publication result.

## What needs to be built (honest gap list)

- `scripts/vsp_gating_probe.py` -- the ~20-word V/S/P-vs-S separation test (DO FIRST)
- `scripts/build_vsps_vocab.py` -- grounded+abstract two-tier vocab from V+P assets
- `scripts/tokenize_vsps.py` -- VSPS tokenizer over a corpus
- `scripts/train_planck2.py` -- VSP-aware SGS LM trainer (adapt train_planck11.py)
- `data/vsp/polysemous_20.json` -- the probe word/sense list

## Sequencing (gate-and-kill, do not skip the probe)

| Phase | What | Kill if |
|-------|------|---------|
| 1 | VSP gating probe (~1wk, no training) | senses don't separate under V/S/P |
| 2 | VSPS vocab design | grounding coverage too thin / blowup explodes |
| 3 | VSPS run on TinyStories | tokenizer doesn't round-trip / disambiguate |
| 4 | Planck 2.0 training | no disambiguation win vs baseline at matched compute |

## Papers (Path B feeds 2 of the 3 we will publish)

1. Alpha-Compositing > Softmax attention -- ALREADY submitted to JMLR
   ([[project_sgs_jmlr_submission]]), not Path B.
2. **VSPS Tokenization** -- from phases 2-3.
3. **VSP-based Models** -- from phase 4, extends the Physical Gaussians paper
   (docs/papers/physical_gaussians*.md).
