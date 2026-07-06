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

The V source is now CLIP-image (§1a decision, 2026-06-23), NOT ShapeNet blobs.
So the real Phase 1 dependency is SD + CLIP in `.venv-sds`, and `data/blobs_shapenet`
is NO LONGER required (it only fed the rejected synset-grounded regime,
`ground_vsp.py`). A `shapenet blobs: false` is fine, ignore it for Phase 1.

```powershell
cd sgs
# Phase 1 (CLIP-image gate) runs in .venv-sds (needs diffusers + CLIP):
.venv-sds\Scripts\Activate.ps1
python -c "import diffusers, transformers; print('diffusers', diffusers.__version__, '| transformers', transformers.__version__)"
python -c "from transformers import CLIPModel; print('CLIP import ok')"
# P (physical): Physical Gaussians P6 material work (hardness R^2=0.54) -- text assets, always present
ls docs/papers/physical_gaussians*.md
# (legacy, NOT needed for the chosen CLIP-image path -- only ground_vsp.py used it)
python -c "import os; print('shapenet blobs (legacy, optional):', os.path.exists('data/blobs_shapenet'))"
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
from NON-textual sources keyed to senses. Two more regimes were then tested:

```powershell
# SYNSET-grounded: V from a curated ShapeNet synset id, P from P6 MEASURED table
python scripts/ground_vsp.py --map scripts/assets/vsp_grounded_map.json `
  --out results/vsp_grounded.json
python scripts/vsp_gating_probe.py --derived results/vsp_grounded.json `
  --glove data/glove.6B.300d.txt --out results/vsp_gating_grounded.json

# CLIP-grounded + sense DEDUP: V from CLIP embedding, senses discovered by
# embedding-space dedup (no hand sense labels). clip-text = CPU stand-in;
# clip-image = the real open-vocab path (box: SD view-gen -> CLIP image-encode)
python scripts/derive_vsp_clip.py --senses scripts/assets/vsp_sense_terms.json `
  --v-source clip-text --out results/vsp_clip.json
python scripts/vsp_gating_probe.py --derived results/vsp_clip.json `
  --glove data/glove.6B.300d.txt --out results/vsp_gating_clip.json
```

**FOUR-REGIME RESULT (2026-06-23) -- the decision-grade table:**

| V source | gain | verdict |
|----------|------|---------|
| hand-seeded | 0.48 | PASS (but grades its own labels) |
| text-derived (GloVe) | 0.13 | FAIL -- text collapses on colliding words |
| synset one-hot | 0.41 | PASS, but separation is near-TAUTOLOGICAL (different id -> orthogonal) |
| CLIP-text | 0.18 | FAIL -- CLIP's TEXT encoder still sees the shared word |
| **CLIP-image (2026-07-06)** | **0.355** | **PASS** -- auto-derived V, max-agg over V,S, P excluded (see §1a result) |

Every TEXT-based V fails; the synset one-hot passes only by construction. The
CLIP-text dedup even MERGED the senses that matter (crane/machine into bird at
sim 1.0). The unfaked test is **CLIP-IMAGE** -- embed GENERATED VIEWS of each
sense, where a bird photo and a construction-crane photo are visually unrelated
so they cannot collide. That test now PASSES (0.355) once the sense phrases,
the aggregation metric, and the P tautology are fixed -- see §1a.

## 1a. Image-grounded V (the chosen direction -- needs the box)

Decision (2026-06-23): ground V in GENERATED IMAGES, not ShapeNet. Why:
- **Open vocabulary**: ShapeNet caps at ~30-55 categories; generated images
  cover anything describable (lighthouse, pagoda, windmill). No category ceiling.
- **Visual senses can't collide**: text "a crane" collapses, but an IMAGE of a
  bird vs a construction crane is visually distinct -> CLIP-image V separates
  them where every text path failed.
- **Reuses owned infra**: SD is already in sds_refine.py; CLIP is one import.

The curation problem does NOT vanish, it AUTOMATES (user's idea, 2026-06-23):
Gemma 4 ENUMERATES a word's distinct senses (text) -> SD generates views per
sense -> CLIP image-embeds -> DEDUP in embedding space so once "crane-bird"
occupies a region, the next "crane" sample must land cosine<thr away to count as
a new sense (else it's the same sense from another angle). Human curation ->
a model (Gemma sense-recall) + a distance threshold (the dedup knob). Both are
real knobs: Gemma may miss a rare sense (coverage = model recall); the threshold
trades splitting one sense vs merging two.

```powershell
# on the box (.venv-sds + transformers/CLIP): real image-grounded V.
# --save-views dumps every SD image so view quality can be eyeballed first.
python scripts/derive_vsp_clip.py --senses scripts/assets/vsp_sense_terms.json `
  --v-source clip-image --gen-model runwayml/stable-diffusion-v1-5 `
  --n-views 4 --dedup-thr 0.85 --save-views results/vsp_views `
  --out results/vsp_clip_image.json
# score with max-aggregation over the honestly-derived blocks (V,S); P excluded
python scripts/vsp_gating_probe.py --derived results/vsp_clip_image.json `
  --glove data/glove.6B.300d.txt --aggregate max --gate-blocks v,s `
  --out results/vsp_gating_clip_image.json
```

**GROUNDED GATE RESULT (2026-07-06): PASS, 0.355 (max-agg over V,S).** The
CLIP-image gate went through three fixes before this honest number:
- **Sense-phrase bug**: `vsp_sense_terms.json` used the bare colliding word for
  both senses (crane->"crane", plane->"plane"), so SD drew its dominant prior
  (a machine, an airplane) for BOTH and dedup merged them to 0 senses. Rewrote
  every phrase to be concrete + unambiguous ("a crane bird with long legs" vs
  "a yellow construction crane machine"). crane/plane/pitcher now survive; their
  V vectors separate at cos 0.53 / 0.52 / 0.43.
- **Metric bug**: concat-of-unit-blocks cosine == MEAN of block cosines, so the
  identical (collapsed) S block mathematically floored separation at (k-1)/k and
  diluted the informative V. Switched to `--aggregate max` (separable if ANY
  block distinguishes the senses). This is the honest metric AND it matches the
  goal: disambiguation succeeds if any modality carries the signal (grey fur
  ball vs grey boulder -> V collapses, P/S separate).
- **P tautology**: P vectors are a hand-authored material table (MATERIAL_TABLE)
  keyed by a hand-assigned material tag (vsp_grounded_map.json). Two labeling
  layers = a curated PRIOR, not automatic grounding (same class as synset).
  So `--gate-blocks v,s` EXCLUDES P from the gate. Result rests on CLIP-image V
  (auto-derived); S is 0 on these identical-surface probe words (the collapse).

Per-word max-agg separation ranges 0.24 (nail) to 0.55 (pitcher), all 20 clear.
Train Planck 2.0 gate is PASSED on honestly-derived V. Caveat kept: generated-
view QUALITY drives V; --save-views + eyeball before trusting (garbage views ->
garbage V).

## 1b. Derive P honestly (phase B -- makes P count toward the gate)

(A) above passes on V alone with P excluded because P is hand-authored. To let
the full V/S/P bundle count -- and to make the "encode everything into ONE
vector, automatically" thesis real -- P must be DERIVED, not looked up:
- Replace the MATERIAL_TABLE lookup + hand material tag with the **P6 material
  MLP** (predict the 8 physical properties from the sense's V/S vector, the
  measured-P6 path in validate_p6_correlation.py / the Physical Gaussians work).
- Then re-run the gate with `--gate-blocks v,s,p`; P now contributes a derived
  signal (crane feather-vs-steel, seal leather-vs-rubber already show real
  material contrast, iron correctly does NOT separate two metals).
**PHASE B RESULT (2026-07-06): BUILT + PASS, 0.367.** `derive_p6_from_vector.py`
trains the P6 MLP on MATERIAL_TABLE, then predicts P for each sense from the
GloVe of its descriptive phrase (no material tag read; drops it; writes
p_source=p6_mlp_derived). Re-gate `--gate-blocks v,s,p` on derived P = **0.367,
PASS**. The important part: derived-P contrasts are now MODEST (crane 0.18, rock
0.06, iron 0.07) vs the old hand-P tautology (rock/table 1.0 by material-vs-None).
So the pass is carried by auto-derived CLIP-image V, with derived-P a small
honest bonus -- a STRONGER claim than hand-P's inflated 0.55, because nothing is
hand-labeled. V and P now BOTH come from a model applied to the sense phrase.

```powershell
python scripts/derive_p6_from_vector.py --in results/vsp_clip_image.json `
  --glove data/glove.6B.300d.txt --out results/vsp_clip_image_pderiv.json
python scripts/vsp_gating_probe.py --derived results/vsp_clip_image_pderiv.json `
  --glove data/glove.6B.300d.txt --aggregate max --gate-blocks v,s,p `
  --out results/vsp_gating_clip_image_pderiv.json
```

## 2. VSPS tokenization experiment (TOKENIZATION IMPROVEMENTS)

If the GROUNDED (CLIP-image) gate passes, design the tokenizer. VSPS =
Visual/Semantic/Physical SGS tokens: word/concept-level units carrying a V/S/P
bundle, NOT compression-optimal subwords. The probe results dictate the design:

1. **Token granularity = the SENSE, not the word.** The four-regime result is
   that a token keyed on the surface WORD collapses polysemy (every text path
   failed). So a VSPS token is minted per SENSE (crane-bird, crane-machine are
   two tokens), and the sense set comes from the Gemma-enumerate + CLIP-image-
   dedup pipeline (§1a), NOT from a subword splitter.
2. **Two-tier vocabulary**:
   - **Grounded tokens**: concrete senses with a CLIP-image V + a P6 P. One per
     sense.
   - **Abstract tokens**: function words / abstractions (justice, the, however),
     S-only, no V/P. The probe showed these (rock-music, table-data-grid) carry
     zero V/P and that is CORRECT, not a gap.
3. **V is the CLIP-image embedding** (the chosen ground, §1a), NOT a synset id
   (one-hot was tautological) and NOT a text vector (collapses). P is the P6
   measured-material vector. S stays GloVe/Planck.
4. **Vocab blowup is bounded**: most words have 1-3 grounded senses, so grounded
   nouns expand ~1-3x. Abstract vocab is unchanged. Measure actual coverage on
   TinyStories before assuming.

```powershell
python scripts/build_vsps_vocab.py `
  --senses results/vsp_clip_image.json `      # CLIP-image V + dedup'd senses
  --materials scripts/assets/vsp_grounded_map.json `   # P per sense
  --glove data/glove.6B.300d.txt `             # S
  --out data/vsps/vocab.json   # TO BUILD
```

GATE: grounded vocab covers the common concrete-noun space at a manageable
blowup; minting a token = (sense -> CLIP-image V, P6 P, GloVe S). Open question:
groundable words with no generated asset yet -> fall back to S-only or generate
on demand.

## 3. VSPS run on TinyStories (DATA-PREP IMPROVEMENTS)

Validate the tokenizer on a small clean corpus before Planck 2.0. TinyStories
(src/tinystories.py) is the right scale: simple vocabulary, concrete nouns, fast.

Data-prep pipeline (the steps the probe work implies):
1. **Sense-tag the corpus**: for each grounded word occurrence, assign its SENSE
   (the dedup'd sense whose V/S/P best matches context). This is word-sense
   disambiguation at tokenize time -- the step SentencePiece skips and the
   reason it collapses senses.
2. **Dedup at corpus scale**: the §1a embedding-dedup runs over the corpus
   vocabulary, not just the 20-word probe set, to mint the full grounded sense
   inventory. Watch the --dedup-thr: corpus-scale will surface more near-miss
   pairs than the probe did.
3. **Cache V/S/P per token** so training reads a lookup, not a live SD+CLIP pass
   (generation is the expensive step; do it ONCE at vocab-build, like the P6 MLP
   and blob index are prebuilt).

```powershell
python scripts/tokenize_vsps.py --corpus data/tinystories `
  --vocab data/vsps/vocab.json --out data/tinystories_vsps   # TO BUILD
```

GATE: VSPS round-trips TinyStories, and polysemous words get SENSE-CORRECT
tokens (the disambiguation the thesis promises) where SentencePiece merged them.

## 4. Planck 2.0 training (TRAINING IMPROVEMENTS)

Only after 1-3 pass. Train a ~100M SGS LM on VSPS-tokenized data. Same
SGSLanguageModel scaffold (src/sgs_lm.py) adapted so each token embedding is the
V/S/P bundle, not S alone. Training changes the probe work implies:

1. **Embedding = concat(V, S, P), not a lookup row.** The token embedding table
   becomes (or is initialized from) the cached V/S/P bundles. V is frozen CLIP-
   image (don't backprop into generated appearance early); S/P can be fine-tuned.
2. **Frozen-V warm start** (mirrors how Raum froze the Planck encoder): start
   with V/P frozen so the model learns to USE the grounded representation before
   it can corrupt it; unfreeze later if needed.
3. **Plain AdamW, stdout logging** (per [[project_sgs_accel_shelved]],
   [[feedback_sgs_wandb_default]] -- the accel track is shelved, wandb is paid).
4. **Disambiguation benchmark is the gate, not val loss.** Val loss alone won't
   show the win; build a polysemy benchmark (sense-correct next-token / retrieval
   on prompts like "the crane flew" vs "the crane lifted").

```powershell
python scripts/train_planck2.py --data data/tinystories_vsps `
  --vocab data/vsps/vocab.json --d-f 1000 --freeze-vp `
  --save-dir checkpoints/planck2 --tokens 1B   # TO BUILD; mirror train_planck11.py
```

GATE: Planck 2.0 beats the SentencePiece-baseline Planck on the disambiguation
benchmark at matched params and tokens. That delta is the publication result.

## What needs to be built (honest gap list)

BUILT + CPU-validated:
- `scripts/vsp_gating_probe.py` -- V/S/P-vs-S separation test (+ --derived path)
- `scripts/derive_vsp.py` -- text/GloVe-derived V/P (FAILED gate, kept as baseline)
- `scripts/ground_vsp.py` -- synset+P6-measured V/P (PASS but tautological V)
- `scripts/derive_vsp_clip.py` -- CLIP V + sense-dedup (clip-text FAIL; clip-image is the box test)
- `scripts/assets/vsp_polysemous.json`, `vsp_sense_terms.json`, `vsp_grounded_map.json`

STILL TO BUILD:
- run `derive_vsp_clip --v-source clip-image` on the box (the real grounded gate)
- `scripts/build_vsps_vocab.py` -- two-tier vocab from CLIP-image V + P6 P + GloVe S
- `scripts/tokenize_vsps.py` -- sense-tagging tokenizer over a corpus
- `scripts/train_planck2.py` -- VSP-bundle SGS LM trainer (adapt train_planck11.py)
- a polysemy disambiguation benchmark (the Planck 2.0 gate)

## Sequencing (gate-and-kill, do not skip the probe)

| Phase | What | Status / kill-if |
|-------|------|---------|
| 0 | gating probe, 4 regimes | DONE 2026-06-23: text fails, synset tautological, CLIP-image is the real test |
| 1 (A) | CLIP-image grounded gate (box) | **PASS 2026-07-06, 0.355** (max-agg V,S; auto-derived CLIP-image V; P excluded as hand-authored) |
| 1 (B) | derive P honestly (P6 MLP) | **DONE 2026-07-06, PASS 0.367** (derive_p6_from_vector.py; V+derived-P, all auto, nothing hand-labeled) |
| 2 | VSPS vocab design | grounding coverage too thin / blowup explodes |
| 3 | VSPS run on TinyStories | tokenizer doesn't round-trip / disambiguate |
| 4 | Planck 2.0 training | no disambiguation win vs baseline at matched compute |

## Papers (Path B feeds 2 of the 3 we will publish)

1. Alpha-Compositing > Softmax attention -- ALREADY submitted to JMLR
   ([[project_sgs_jmlr_submission]]), not Path B.
2. **VSPS Tokenization** -- from phases 2-3.
3. **VSP-based Models** -- from phase 4, extends the Physical Gaussians paper
   (docs/papers/physical_gaussians*.md).
