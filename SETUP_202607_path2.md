# SGS Path B (Planck 2.0, VSP architecture) Setup, July 2026

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
# hand-seeded V/P (proves the representation CAN separate):
python scripts/vsp_gating_probe.py --words scripts/assets/vsp_polysemous.json `
  --glove data/glove.6B.300d.txt --out results/vsp_gating.json
# AUTO-derived V/P (the honest test -- V/P from sense terms, not hand labels):
python scripts/derive_vsp.py --senses scripts/assets/vsp_sense_terms.json `
  --glove data/glove.6B.300d.txt --out results/vsp_derived.json
python scripts/vsp_gating_probe.py --derived results/vsp_derived.json `
  --glove data/glove.6B.300d.txt --out results/vsp_gating_derived.json
```

**RESULT (2026-06-22), and it reshapes Path B:**
- Hand-seeded V/P: separation gain **0.48, PASS**. The representation is sound --
  V|S|P bundling DOES separate senses S-only collapses (S-only sim = 1.000).
- AUTO-derived V/P (P from the P6 MLP on the term's GloVe; V from category-anchor
  similarity): gain **0.13, FAIL**. Words with distinct sense-terms separate
  (table 0.23, scale 0.26, mouse 0.21) but the cases that MOST need
  disambiguation -- IDENTICAL surface form (crane bird vs machine 0.00, plane
  0.00) -- collapse, because text-derived V/P has only the colliding word to go
  on, and text is the very signal that collapses. Circular.

**The bottleneck is sense->asset GROUNDING, not the bundling.** V and P must come
from NON-textual sources keyed to senses: actual Objaverse/ShapeNet 3D blobs (V)
and actual P6 MEASURED material values (P), mapped to senses by something other
than GloVe similarity. Until that grounding exists, deriving V/P from text adds
little over S alone on the hard (colliding) cases. NEXT before Planck 2.0: build
the sense->asset map (e.g. WordNet-synset -> ShapeNet-synset for V, sense ->
material label for P) and re-run the derived gate. Train Planck 2.0 only if the
GROUNDED-derived gate passes, not the hand-seeded one.

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
