# SGS Glossary

Definitions for terms used across the Semantic Gaussian Splatting project:
whitepapers, plans, runbooks, roadmap, and code. Terms are grouped by theme.
When two terms are near-synonyms, the canonical one is listed first and
others link to it.

Last updated: 2026-05-06.

---

## 1. Core SGS primitives

### Gaussian
A probability distribution over a continuous space parameterised by a mean
`μ` and covariance `Σ`. In SGS a Gaussian is never just a distribution; it
is a *primitive* carrying a mean, a shape (covariance), an opacity, and a
feature payload (see **Semantic Gaussian**).

### Semantic Gaussian
The atomic unit of an SGS model. A four-tuple:

```
G = (μ, L, α, f)
```

- **μ** (mean, `d_s` dims, typically 64-128): position in **splatting
  space**. What this primitive *is about*.
- **L** (Cholesky factor of covariance, `d_s × d_s` triangular): shape and
  orientation of the Gaussian. Encodes semantic breadth / directional
  uncertainty. We store `L`, not `Σ`, so `Σ = L Lᵀ` stays positive-definite
  by construction.
- **α** (opacity / raw alpha, scalar): base salience. How "loud" this
  primitive is before compositing.
- **f** (feature payload, `d_f` dims, typically 300-1000): the rich
  semantic content that gets composited. Think of `μ, L, α` as *where* and
  *how much*; `f` is *what*.

Per-primitive count at `d_s=64, d_f=512`: 64 + 2080 + 1 + 512 ≈ 2,657
parameters.

The name "Semantic Gaussian" is the core analogy to 3DGS: a 3D Gaussian
has position in 3D + appearance (spherical harmonics); a Semantic Gaussian
has position in semantic space + features.

### Splatting space
The low-dimensional space (typically `d_s = 64`) where Gaussian means live
and where the rendering-equation kernel `K(q, μ, Σ)` is evaluated. Kept
small because Gaussian evaluation degenerates in high dimensions (a
Gaussian in 768-d evaluates to numerically zero almost everywhere).

### Feature space
The higher-dimensional space (typically `d_f = 300-1000`) where each
Gaussian's payload lives. This is what gets composited into the output.
Analogy: splatting space is 3D position; feature space is per-Gaussian
spherical-harmonic coefficients.

### Opacity / alpha
Per-Gaussian scalar in `[0, 1]` after a sigmoid on `raw_alpha`. Plays the
same role as alpha in image compositing: higher alpha means this Gaussian
contributes more and leaves less transmittance for later Gaussians in the
same render order.

### Gaussian kernel
The evaluation function:

```
K(q, μ, Σ) = exp(-½ · (q - μ)ᵀ Σ⁻¹ (q - μ) / τ)
```

where `q` is a query point and `τ` is a temperature (often `τ = d_s`).
`K` falls off to ~0 beyond ~4σ, which is why only a small subset of
Gaussians contributes to any query — the source of SGS's natural sparsity.

### Rendering equation (semantic)
The compositing operator that turns a set of Semantic Gaussians into a
single meaning vector given a query `q`:

```
Meaning(q) = Σᵢ fᵢ · wᵢ
  where
    wᵢ = αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
    Tᵢ = ∏ⱼ<ᵢ (1 − αⱼ · K(q, μⱼ, Σⱼ))     ← transmittance
```

Ordered: earlier Gaussians get more transmittance. Multi-pass rendering
(see below) corrects the front-loading bias by letting later Gaussians
grow α across passes.

### Transmittance
`Tᵢ = ∏ⱼ<ᵢ (1 − αⱼ K(q, μⱼ, Σⱼ))`. How much "budget" remains for
Gaussian `i` after earlier ones absorbed their share. Semantic analogue
of visual occlusion (3DGS) and **psychoacoustic masking** (Klang).

### Alpha-compositing
Synonym for the summation-over-Gaussians in the rendering equation. The
novel theoretical claim in the SGS whitepaper is that alpha-compositing
over Semantic Gaussians and softmax attention are mathematically related
operations; softmax is a special case.

### Semantic viewpoint
A pair `(P, q)` where `P: d_s → m` is a projection matrix and `q` is a
query point in the projected space. Viewpoints let the same Gaussian
scene be "read" from different task-specific angles (see `sgs_lm.py`).

### Adaptive density control
The 3DGS training trick we inherit: split Gaussians in regions of high
reconstruction loss, prune Gaussians with near-zero opacity, clone under-
represented ones. Lets the vocabulary of Gaussians self-organise to the
data.

### Multi-pass rendering
Iterative refinement within a single forward pass. On pass `p`:
1. Project to each viewpoint.
2. Evaluate the kernel.
3. Alpha-composite into a meaning vector.
4. Update each Gaussian's parameters based on the rendered meaning
   (the **operators**).
5. FFN on the features.

Typically 2-4 passes. Analogous to transformer layer depth.

---

## 2. SGS model components

### SGS model
Any model whose primary computational unit is a Semantic Gaussian
composed via the rendering equation, instead of (or in addition to) a
transformer layer. Concrete instantiations: `src/sgs_lm.py`
(SGSLanguageModel).

### SGS language model (SGS LM)
A specific SGS model for causal language modelling. Embeds tokens as
Semantic Gaussians, runs multi-pass rendering, decodes to next-token
logits. Implemented in `src/sgs_lm.py`. All Planck checkpoints are SGS
LMs.

### Gaussian scene
The set of Semantic Gaussians active for a given input. For an SGS LM
processing a sentence, the scene is the `T` per-token Gaussians plus any
retrieved **blobs**. For Raum, the scene is the rendered 3D cloud for
the predicted object list.

### Tokeniser-agnostic note
SGS primitives are token-anchored in practice (one Gaussian per
vocabulary item) but nothing in the rendering equation requires that.
Blob retrieval breaks the token-level anchoring deliberately; see
**blob** below.

---

## 3. Blobs and retrieval

### Blob
A single Semantic Gaussian (or a tight cluster of them) stored outside
the model, used as retrieval memory. Same shape as a per-token Gaussian
`(μ, log_var, α, f)` but:
- **Origin**: constructed externally from a corpus (Wikipedia, RSS
  news, ShapeNet meshes, audio reference clips), not learned with the
  LM weights.
- **Lifetime**: persists across runs; the LM does not modify blob
  parameters at inference.
- **Role**: gets composited *into the Gaussian scene alongside token
  Gaussians* during rendering. The LM "sees" the retrieved blob as if
  it were a token in the scene.

Blobs are how SGS separates "reasoning fabric" (what the LM learns)
from "knowledge" (what a cheap offline pipeline builds). See
**SETUP_202605 §2.4** for how Wikipedia blobs are built.

### Blob store / blob library / blob index
A on-disk collection of blobs plus a retrieval backend. Three current
stores:

| store | contents | build script | lookup |
|---|---|---|---|
| `data/blobs/tinystories` | TinyStories-derived clusters | `scripts/build_blobs.py` | flat Faiss |
| `data/blobs/wikipedia` | Planck 1.3 Wikipedia blobs | `scripts/build_blobs.py` (with `--data-dir data/wikipedia`) | flat Faiss |
| `data/blobs/<shapenet>` | one `.pt` per object class | `scripts/build_blobs_shapenet.py` (planned for Raum 1.1) | index-by-class |

"Blob store", "blob library", and "blob index" are used interchangeably.
Preferred terms by context: **library** for Raum's 3D object catalogue,
**index** for Faiss-backed text retrieval.

### Blob retrieval
Given a query (e.g. the prompt's meaning vector), find the top-k
nearest blobs by kernel or cosine similarity in splatting space.
Retrieved blobs are mixed into the Gaussian scene for the next render.

### Conditional blob decoder
A small MLP that takes a word embedding and outputs a fresh Gaussian
cloud on the fly, instead of indexing into a fixed blob library. Planned
for **Raum 0.2** to handle OOV objects at inference. Trained with
distillation (match stored blobs for known classes) + consistency (words
close in GloVe should produce similar clouds).

---

## 4. The Radiance Labs model portfolio

Each entry has a swimlane id on `roadmap.md`. All of these are SGS
models except where noted.

### Planck
Small SGS language models, ~100M params. The line where blob-conditioning
and training infrastructure are validated.

- **Planck 1.0**: foundation 100M LM, TinyStories base, no blobs.
- **Planck 1.1**: validated that blob retrieval at inference meaningfully
  improves a small LM *without retraining*. This is the blob-concept
  proof.
- **Planck 1.2 / 1.2.1 / 1.2.2**: accel-recipe sweep (SGS-native +
  third-party Muon/Liger/FA-2/FP8 lanes). Closed FAIL; track shelved
  2026-05-01 at 100M scale. Base training reverts to plain AdamW.
- **Planck 1.3**: Wikipedia base retrain (~4B tokens) + static Wikipedia
  blob index (1.3.1a) + live-news RSS blob index (1.3.1b). In progress
  as of 2026-05-06.
- **Planck 1.4**: conversation-memory blobs (per-turn blob writer +
  hybrid recency/similarity retrieval). Open.

### Hertz
Large SGS language models, ~1B+ params. The scale track.

- **Hertz 1.0**: shelved 2026-04-20 (infeasible wall-clock without an
  accel recipe).
- **Hertz 1.2**: plain-AdamW run, ~10 days for 10B tokens on a 4090.
  Open, re-sequenced to run last in the current queue.

### Helmholtz
Reserved model swimlane, TBD.

### Einstein
Reserved swimlane for a future frontier-scale SGS model.

### Klang
Audio-SGS model — the rendering equation applied to
**(time, frequency) → (amplitude, phase)**. Key physical correspondence:
psychoacoustic masking IS transmittance.

- **Klang 1.0 / 1.1**: early concept + variant sweep.
- **Klang 1.2**: complex-valued Gaussians + transmittance compositing +
  MRSTFT loss. Shipped with passing gates but still behind Klang 1.1
  Variant A on absolute quality.
- **Klang 1.3**: scale-up (1000-3000 Gaussians). 500g run shipped;
  1000/2000/3000g sweep deferred until just before Hertz 1.2.

### Raum
Text → 3D Gaussian cloud. Single learned **Raum bridge** sitting between
a frozen Planck encoder and an analytic renderer, operating over a blob
library.

- **Raum 1.0**: template-routing bridge (6 shapes), shipped 2026-04-27.
- **Raum 1.1**: 1-model architecture (see below), 30-60 ShapeNet classes,
  editable DSL.

### Raum bridge
The ~5-10M-param transformer at the heart of Raum 1.1. Input: token
features from a frozen Planck encoder. Output: a DSL description of a
3D scene (see **DSL v1**). Heads:

| head | shape | loss |
|---|---|---|
| object slot | `[B, T]` | BCE |
| position | `[B, T, 3]` | MSE |
| color | `[B, T, 3]` | MSE |
| scale | `[B, T, 3]` log | MSE |
| blob_id | `[B, T, N_BLOBS]` | CE |
| pair relation | `[B, T, T, N_REL]` | CE |

Trained end-to-end on procedurally-generated 3-object scenes. The bridge
is the *only* learned component in Raum 1.1.

### DSL v1
The JSON intermediate between Raum bridge output and the renderer. An
editable scene description:

```json
{
  "version": 1,
  "objects": [
    {"id": "car",  "blob": "car",  "color": "#c33", "scale": 1.0,
     "position": [0.0, 0.0, 0.0]}
  ],
  "relations": [
    {"subject": "car", "rel": "left_of", "anchor": "tree"}
  ]
}
```

The user can edit the DSL in the Raum 0.1 demo and re-render without
re-running the bridge — the "nano-banana edit loop". Schema in
`src/raum/dsl.py`.

### Frozen Planck encoder
Planck 1.1 (initially) or Planck 1.3 (once it ships) loaded
weights-only, no gradient flow. Provides token features `(d_f = 300)`
that the Raum bridge consumes. Swapping encoders is cheap — the bridge
is small and re-trains in hours.

---

## 5. The product portfolio

### Prisma
Reserved product swimlane, TBD.

### Raum (product)
Local text→3D web demo.

- **Raum 0.0**: shipped 2026-04-27 on the 1.0 bridge + 6 templates.
- **Raum 0.1**: 1-model architecture + 30-60-class ShapeNet blob library
  + editable DSL. Foreground track.

### Satz
Local text demo for the Planck LM + blob retrieval path. Prompt textbox,
streamed generation, right-panel retrieved-blob list with transmittance
weight bars, k-slider. No new training.

- **Satz 0.1**: primary path is Planck 1.3 + Wikipedia blobs. Planck 1.1
  + TinyStories blobs is a flagged-placeholder fallback.

### Klang (product)
Audio demo / app surface for the Klang model. Distinct from **Klang
(model)**; same name, separate swimlane on `roadmap.md` (lane 9 vs lane
5).

---

## 6. Training and evaluation vocabulary

### Two-stage training
Planck 1.1's recipe: stage 1 learns token Gaussian parameters; stage 2
adds blob attention with the stage-1 weights frozen. Three epochs each.
~3 hours on 4090 for ~1-2B TinyStories tokens.

### Gate
A numeric threshold that must pass before a roadmap row flips to `done`.
Stated per-version in the plan doc (e.g. Raum 1.1 §7). Gates can FAIL
(e.g. Planck 1.2 gate FAIL: val loss 2× worse, speedup 1.07×) — a
failing gate shelves the track, not the roadmap row.

### MMLU-lite
A small internal knowledge benchmark (~200 questions) used to verify
that a new base model got *smarter without blobs*, before declaring
blob retrieval is what made it better. See `planck_13_plan.md` §1.3.0.

### Ablation
Standard meaning: run the eval with component X turned off, so the
contribution of X is isolated. In SGS, the canonical ablations are:
(a) model with vs. without blobs, (b) Planck 1.3 with vs. without
Wikipedia blob index.

### Accel / accel recipe
Any training-time speed-up applied to an SGS model. Current menu: Muon
optimiser, Liger kernels, FlashAttention-2, FP8 training, SGS-native
(transmittance-weighted loss). The 100M-scale SGS-native work is
shelved (see `project_sgs_accel_shelved`).

### Blob-concept validation
Planck 1.1's deliverable: prove blobs improve a frozen LM at inference.
The bar SGS has to clear before committing GPU-weeks to Hertz.

---

## 7. Corpora and datasets

### TinyStories
A synthetic narrative-fiction corpus for small-LM training, ~2B tokens.
Planck 1.0/1.1 base corpus. Phased out at Planck 1.3 because
definitional / encyclopedic priors are absent.

### Wikipedia (corpus)
English Wikipedia via the HuggingFace `wikimedia/wikipedia` dataset,
pinned to revision `20231101.en` (~6.4M articles, ~4B tokens at our
tokenizer). Source corpus for Planck 1.3's base training AND the
static blob index — same corpus, two roles.

### BabyLM
Grammar-clean small corpus. Considered and rejected as the Planck 1.3
base because it is factually threadbare.

### ShapeNet Core v2
~51K 3D models across 55 categories (CC-BY-SA). Source for Raum 1.1's
blob library; meshes are converted to Gaussian clouds offline.

### Live-news RSS
Reuters + AP + BBC + Al Jazeera RSS, ingested every ~5 min for Planck
1.3.1b's dynamic blob index. Each article becomes ~150-token blobs
tagged `{source, publish_ts, headline, url}` with a 72h TTL.

---

## 8. Repo structure quick reference

| path | purpose |
|---|---|
| `src/sgs_lm.py` | SGSLanguageModel — the LM class |
| `src/gaussian.py` | Core Semantic Gaussian ops |
| `src/tinystories.py` | Data pipelines: TinyStories + Wikipedia (legacy filename) |
| `src/blob_store.py` | Blob retrieval backend |
| `src/raum/render_3d.py` | Raum analytic renderer |
| `src/raum/dsl.py` | Raum DSL schema + validator |
| `scripts/train_lm.py` | SGS LM base trainer (Planck 1.0, 1.3) |
| `scripts/train_planck11.py` | Blob-attention finetuner (Planck 1.1) |
| `scripts/build_blobs.py` | Text blob builder |
| `scripts/build_blobs_shapenet.py` | Raum 1.1 blob library builder (planned) |
| `scripts/train_raum_bridge.py` | Raum bridge trainer |
| `scripts/validate_klang.py` | Klang audio-decode validator |
| `roadmap.md` | Single source of truth for versions |
| `pm/index.html` | Swimlane visualizer of `roadmap.md` |
| `SETUP_202605.md` | Live runbook (supersedes earlier SETUP files) |
| `docs/plans/` | Per-version plan docs (planck_13, raum_11, satz_01, etc.) |
| `docs/whitepaper/` | Research proposal (v1-v4) |
| `docs/course/` | Pedagogical articles (SGS from first principles) |

---

## 9. Deprecated / avoid

- **Planner LM**: an earlier Raum design with a separate prompt→DSL
  model. Dropped 2026-05-01 in favour of the 1-model architecture. Do
  not reopen without a concrete capability gap (composite-object
  decomposition is *not* one — that's a `group` DSL node + conditional
  decoder).
- **`wikiextractor`**: raw Wikipedia XML cleaner. Broken on Python
  3.11+. Use the HF `wikimedia/wikipedia` dataset instead.
- **TinyStories as Planck base**: correct for 1.0/1.1, phased out at
  1.3. Satz 0.1 on TinyStories blobs is a flagged-placeholder fallback
  only.
