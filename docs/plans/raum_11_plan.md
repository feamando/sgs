# Raum 1.1 — 1-model text→3D with a blob library

*Status: draft plan. Written 2026-05-01, re-sequenced 2026-05-04.
Paired with Raum 0.1 product stages
(`docs/plans/d1c_raum_01_plan.md`).*

*Run order change 2026-05-04: Raum 1.1 is now the FOREGROUND track.
Stages 1.1.A and 1.1.B start against the frozen Planck 1.1 encoder
on disk. Planck 1.3 trains in the background (~3-7 days, plain
AdamW, see `docs/plans/planck_13_plan.md`); once 1.3 is ready we
re-train the bridge against it. The bridge is ~5-10M params and
re-trains in hours, so the encoder swap is cheap. Satz 0.1
(`docs/plans/satz_01_plan.md`) runs after Planck 1.3 ships and
independently validates the encoder choice.*

Raum 1.0 proved the `text → template-routing bridge` concept on a
six-shape library. Raum 1.1 is the model half of Raum 0.1 and the
point where we commit architecturally to a **single learned bridge**
running against a **library of Gaussian blobs** — the blobs are the
semantic-and-physical primitive, the bridge is the only learnable
component, and the renderer + DSL are pure functions on its output.

This replaces the earlier idea of a separately-trained Planner LM
feeding an Executor. We keep the DSL as an editable intermediate
(the "nano-banana edit loop" is a feature), but the DSL is a product
of the bridge, not a second model's output.

Everything below is a plan, not a commitment.

---

## 1. Architecture — one learned bridge

```
prompt
  │
  ▼
[frozen Planck 1.1 encoder] ──► token embeddings  (d_f = 300)
  │                                           │
  ▼                                           │
[Raum bridge]  ◄───────────────────────────────┘
  │
  ├─► per-token object slot, position, scale, color
  ├─► per-token blob_id head  (softmax over N_BLOBS)
  ├─► pair-level relation head
  │
  ▼
[DSL v1]  ◄─── editable intermediate (JSON, node tree in UI)
  │
  ▼
[analytic renderer]  ──► Gaussian cloud  ──► viewer.js
```

Three moving parts:

- **Frozen encoder.** Planck 1.1 to start (it is already on disk
  with a blob-attention mechanism, so token features already live
  in a space where "object-ness" is separable). Swap to Planck 1.3
  once that ships — 1.3's Wikipedia base adds encyclopedic priors
  that strictly help the bridge's blob-id head over 30-60 classes.
  Bridge re-training under the new encoder is hours, not days.
  Loaded weights-only, no gradient flow; we do not run autoregressive
  generation. Satz 0.1 independently validates the encoder choice.
- **Raum bridge.** A small transformer (target ~5-10M params, see
  §3) that maps token features to DSL-shaped outputs. This is the
  *only* thing we train.
- **Blob library + renderer.** Read-only at inference. Blobs are
  fixed Gaussian clouds (one per object class, §4). The renderer
  is the existing `src/raum/render_3d.py`.

The DSL sits between bridge output and renderer, so a user can edit
the DSL in the demo and get a new render without re-running the
bridge. This is the promised edit loop; no second model required.

### Why one model, not two or three

Considered and rejected:
- **Three-model stack** (semantic encoder → planner LM → executor
  bridge). Three components, three training loops, three schemas at
  the boundaries; every schema boundary is a retest surface when a
  tokenizer or GloVe dim changes.
- **Two-model stack** (planner LM → executor bridge). Better than
  three, but we'd be separately training a Planner to emit a DSL
  our bridge already has to produce internally. Duplication without
  a capability gain at Planck-1.1 scale.
- **One model, end-to-end to splats, no DSL.** Loses the editable
  intermediate. The DSL is the product feature; we keep it as a
  pure function of the bridge output.

Selected: **one learned model (the bridge), frozen Planck encoder,
DSL as pure output, blob library + analytic renderer.**

---

## 2. Heads

The bridge keeps the Raum 1.0 heads and replaces the template argmax
with a blob-id head of the same shape but much larger range.

| head | shape | loss | source of supervision |
|---|---|---|---|
| object slot | `[B, T]` binary | BCE | data generator `role` labels |
| position | `[B, T, 3]` | MSE | analytic scene positions |
| color | `[B, T, 3]` | MSE | generator's per-object color |
| scale | `[B, T, 3]` log | MSE | generator's per-object scale |
| **blob_id** | `[B, T, N_BLOBS]` softmax | CE | generator's ground-truth class id |
| pair relation | `[B, T, T, N_REL]` | CE | generator's pairwise relation labels |

Nothing exotic. The interesting change is §4 — what `blob_id`
indexes into.

---

## 3. Bridge scale — parameter sweet spot

Raum 1.0 shipped at ~478K params (`d_model=128`, 2 layers, 4 heads)
for a 6-template library. Scaling to a blob library pushes us
toward more capacity, but memory on a 24 GiB 4090 caps us before
the model quality plateaus.

| variant | d_model | layers | heads | params | blob-library fit |
|---|---|---|---|---|---|
| 1.0 baseline | 128 | 2 | 4 | ~478K | 6 blobs |
| 1.1 small | 192 | 4 | 6 | ~2M | 30-50 blobs |
| **1.1 target** | **256** | **6** | **8** | **~5-10M** | **~200 blobs** |
| 1.1 large | 384 | 8 | 8 | ~25M | ~500 blobs, likely OOM at the Raum 0.1 data scale |

Sweet spot is the **1.1 target** row. Reasoning:
- 1.1 small is very close to 1.0 and will not obviously beat a
  keyword router on 200 blobs.
- 1.1 large crosses into territory where the per-pair relation head
  blows up activation memory (`[B, T, T]`) at T = 16+ on 3-object
  sentences with paraphrase. We'd need gradient checkpointing to
  avoid that, and we already know (see `project_sgs_accel_shelved`)
  that we do not want to be debugging Windows-4090 activation
  spills on Raum.
- 1.1 target fits comfortably at batch 128 on 4090 bf16, trains in
  a few hours, and gives the bridge enough capacity that the blob
  head is the bottleneck rather than the transformer.

If 200 blobs are not enough to cover the Raum 0.1 object vocabulary
we want, the right next step is **not** bigger bridge — it's a
*conditional blob decoder* (§4, stage C), which is a 0.2-scope head.

### Budget
| hyper | value |
|---|---|
| d_model | 256 |
| d_f | 300 (GloVe, fixed) |
| d_s | 64 (fixed) |
| n_layers | 6 |
| n_heads | 8 |
| ffn_mult | 4 |
| N_BLOBS (target) | 60-200 (see §4) |
| N_REL (from generator grammar) | 12 |
| max_len (tokens) | 32 |

---

## 4. Blob library — where blobs come from

Raum 1.0's six templates are procedural primitives. Raum 1.1's
library has to cover "everyday objects a user would type": car,
tree, chair, bottle, cup, dog, house, etc.

Decision: **ShapeNet Core v2** as the source, converted to
Gaussian clouds offline. No conditional decoder in 1.1.

### Stage A — ShapeNet import (~3-5 days)

1. **Download.** ShapeNet Core v2 (~5 GB, 51K models across 55
   categories, CC-BY-SA). One-time fetch.
2. **Category filter.** Start with ~30 categories matching the
   Raum 0.1 object vocabulary (`car`, `chair`, `lamp`, `sofa`,
   `table`, `bottle`, `cup`, `guitar`, `airplane`, `bookshelf`,
   ...). Drop categories whose canonical mesh is too thin to
   Gaussian-fit well (rifle, pistol).
3. **Canonical mesh per class.** Pick one mesh per class (the
   one closest to the category centroid by some simple geometric
   feature, or just curate by hand; curation is tractable at 30
   classes).
4. **Mesh → Gaussian cloud.** For each canonical mesh, sample N
   surface points via Poisson-disc sampling (`trimesh.sample`),
   fit per-point log-scale from the local neighbour radius, set
   opacity to 1.0, set color from the mesh's face albedo. Save as
   `data/blobs/<class>.pt` with fields
   `{means, scales_log, opacities, colors}` matching the existing
   template format.
5. **Naming.** Blob class names must exactly match Raum bridge
   vocabulary (GloVe words). Store `data/blobs/index.json` mapping
   `blob_id → class_name`.

Scripted end-to-end in `scripts/build_blobs_shapenet.py` (new, ~200
LOC). Pattern mirrors `scripts/build_blobs.py` (text blobs for
Planck) for consistency.

### Stage B — grow to ~60 classes (~1 week)

Once Stage A's 30 classes work end-to-end (bridge trains, renderer
consumes, demo loads), grow the library by adding the next 30
ShapeNet categories that match the Raum 0.1 data generator's object
vocabulary. This is almost entirely mechanical; new class means:
- one new row in `data/blobs/index.json`,
- one new `.pt` file,
- retrain the bridge with `N_BLOBS` bumped.

### Stage C (deferred, 0.2 scope) — conditional blob decoder

When the library grows beyond ~200 classes, adding one `.pt` file
per class becomes storage-bound and the blob_id head becomes a
noisy multi-class problem. The path is a **conditional blob
decoder**: a small MLP that takes a word embedding and outputs a
Gaussian cloud directly.

```
word_embedding (300-d GloVe)
  │
  ▼
[decoder_net: 2 hidden layers, ~1M params]
  │
  ▼
Gaussian cloud: (means, scales_log, opacities, colors) × K_gaussians
```

Training signal:
- **Distillation**: for the classes already in the library,
  penalise the decoder for diverging from the stored `.pt` cloud.
- **Consistency**: two words close in GloVe should produce
  Gaussian clouds whose Chamfer distance is small.

This is real research, not 0.1 infrastructure. Deferred. The
architectural hook we need to leave in 1.1 is: the renderer must
accept blobs that are either indexed from the library **or**
constructed on the fly by a decoder. That's a single interface
boundary in `src/raum/render_3d.py`, cheap to design for now.

---

## 5. DSL v1 — editable intermediate

Sketch (versioned, executor refuses unknown versions):

```json
{
  "version": 1,
  "objects": [
    {"id": "car",  "blob": "car",  "color": "#c33", "scale": 1.0,
     "position": [0.0, 0.0, 0.0]},
    {"id": "tree", "blob": "tree", "color": "#263", "scale": 1.4,
     "position": [1.5, 0.0, 0.0]}
  ],
  "relations": [
    {"subject": "car", "rel": "left_of", "anchor": "tree"}
  ]
}
```

Changes from the old DSL draft in `d1c_raum_01_plan.md`:
- `template` → `blob`. Name aligns with the "Gaussian as semantic
  and physical primitive" pitch.
- `repeat` stays as a stretch feature for 0.1 stage 0.1.5 (tree
  scatter for a jungle scene).
- `group` deferred to 0.2 (composite objects like `castle = towers
  + keep + battlements`).

Schema lives in `src/raum/dsl.py` (new file). `validate(dsl)`
returns `(is_valid, errors)`. Executor path: `DSL → tensor pack →
renderer`.

---

## 6. Training loop

Reuses `scripts/train_raum_bridge.py` with changes:

| flag | purpose | new/existing |
|---|---|---|
| `--encoder-checkpoint` | path to Planck 1.1 `best.pt` | **new** |
| `--freeze-encoder` | bool, default True | **new** |
| `--blobs-dir data/blobs` | path to blob library | **new** |
| `--n-blobs-max` | cap on library size for ablations | **new** |
| `--with-relation-head` | pair relation head | existing (1.0) |
| `--d-model 256 --n-layers 6 --n-heads 8` | bridge scale | existing |

Loss (weighted sum):
```
L = w_pos * L_pos + w_scale * L_scale + w_color * L_color
  + w_blob * L_blob_ce + w_rel * L_rel_ce
  + w_dir * L_direction_pairwise
```

Defaults: `w_pos=1.0 w_scale=0.5 w_color=0.2 w_blob=1.0 w_rel=0.5
w_dir=0.5`. Tune only if one metric is the bottleneck on the 3-object
val set.

### Expected wall-clock
On 4090, Raum 1.0 trained in ~2-3 hours at 478K params. 1.1 target
(~5-10M params, 3-object data, 60-class library) should be ~6-10
hours for a good checkpoint. ShapeNet import adds ~1-2 hours of
one-time preprocessing.

---

## 7. Gates

Flip `6-raum-1-1` to `done` when all of:
- **Object accuracy**: per-token blob_id argmax accuracy > 85% on
  3-object val.
- **Position MSE**: < 0.15 (relative to unit-box scene).
- **Pair-direction accuracy**: > 90% on consecutive pairs.
- **Relation head accuracy**: > 85% on all pairs.
- **Analyzer pass**: `scripts/analyze_raum_bridge.py` outputs at
  least 5 end-to-end correct 3-object scenes on the held-out
  benchmark.

If blob_id accuracy underperforms at 60 classes, ablate with
N_BLOBS=30 to check whether it's a capacity problem (smaller
library recovers accuracy) or a data problem (doesn't). Capacity
problem → one more bridge scale bump (up to `d_model=320 n_layers=6`,
still under the 25M-param line). Data problem → review the scene
generator for class imbalance.

---

## 8. Rollout

| stage | scope | time | deps |
|---|---|---|---|
| 1.1.A | ShapeNet ingest → 30-class blob library | ~3-5 days | ShapeNet download |
| 1.1.B | Bridge at 1.1 target scale, freeze encoder, blob_id head | ~2-3 sessions | 1.1.A + Planck 1.1 ckpt |
| 1.1.C | Relation head + pair direction loss extension | ~1-2 sessions | 1.1.B |
| 1.1.D | DSL v1 + renderer consumes DSL | ~2 sessions | 1.1.C |
| 1.1.E | Grow library to 60 classes, retrain | ~1 week | 1.1.D |

1.1.A through 1.1.D is the "model shippable with Raum 0.1 product
stage 0.1.3". 1.1.E is the "enough objects to feel like a scene
engine" deliverable; it can ship after 0.1.3 lands.

---

## 9. Open questions

- **Encoder features vs. logits.** Do we feed the bridge
  Planck 1.1's final-layer embeddings (before unembed) or the
  unembed features themselves? Final-layer embeddings are the
  richer signal; default to that, reconsider if the bridge
  underfits.
- **Scene scale calibration.** ShapeNet meshes are in their own
  unit conventions (per category). We need a per-class scale
  normaliser baked into the blob `.pt` file so "car" and "cup"
  are roughly the right relative size in the scene. Hand-author
  a `scale_hint` column in `data/blobs/index.json` as part of
  Stage A.
- **Color from mesh vs. from prompt.** Meshes have canonical colors
  (a blue car mesh is blue). We want the prompt's color to win.
  Default: the renderer multiplies mesh color by the predicted
  color from the color head (so the head predicts a tint, not an
  absolute). Revisit if tinting looks washed out.
- **Frozen encoder drift.** If Planck 1.3 ships a stronger base
  model, the 1.1 bridge is still tied to Planck 1.1. Retrain
  cost is small (~1 day); no architectural change.
