# Raum 0.1 — product: compositional scenes, common-object vocab, OOV, editable DSL

*Status: rewritten 2026-05-01 for the 1-model architecture. Supersedes
the earlier Planner+Executor draft. Depends on Satz 0.1
(`docs/plans/satz_01_plan.md`) validating Planck 1.1 as a frozen
encoder, and on Raum 1.1 (`docs/plans/raum_11_plan.md`) delivering the
bridge + blob library.*

Raum 0.0 handled "A [rel] B" on six primitives. Raum 0.1 is the
**product** rollout of Raum 1.1: it exposes the bridge + blob library
as a demo that can render 3-object scenes with a ~60-class common-object
vocabulary, route OOV words gracefully, and surface an editable DSL
("nano-banana edit loop") next to the prompt.

The big change from the earlier draft: **there is no separately
trained Planner LM**. The bridge itself emits the DSL; Planck 1.1 is
a frozen encoder, not a generator. The DSL stays because it is the
editable intermediate that makes the edit loop a feature, not because
it is the protocol between two models.

Everything below is a plan, not a commitment. Each section states the
goal, the cheapest sharp version, a harder version, and the main
tradeoff.

---

## 1. Complex scenes from prompt context

### Goal
Render scenes with 3+ objects, longer sentences ("a small red sphere
above a large blue cube to the left of a green cone"), and nested /
conjoined relations.

### Current limit
Per-token position/template/color/scale heads plus a pairwise-direction
loss restricted to slots (0, 1). The data generator builds only 2-object
phrases.

### Proposed changes
- **Data generator**: extend `src/raum/data.py::generate_dataset` to
  emit N-object scenes, N sampled in {1, 2, 3, 4}. Relation words
  still anchor pairs, but with an "anchor object" pointer so "C to
  the right of B" uses B as anchor, not A.
- **Object slot assignment**: keep the per-token object predictions.
  Hungarian matching between predicted object tokens and ground-truth
  slots is deferred to 0.2.
- **Relation graph head**: add a small per-pair head (handled in
  Raum 1.1 §2, not here).

### Sharp version (ship first)
- N in {1, 2, 3}; keep 2-hop relations ("above", "left of", "on") only.
- Still train with analytic labels (no renderer).
- Pair loss over all consecutive object pairs (no Hungarian yet).
- Success metric: > 90% direction accuracy on 3-object scenes in val.

### Harder version (later)
- Nested phrases ("the cube that is below the cone above the plane"),
  quantifiers ("three red spheres"), reference resolution ("the small
  one"). Probably Raum 0.2.

### Tradeoff
Longer sequences mean more tokens routed to objects; softmax over
`N_BLOBS` templates gets noisier when the context window has 3 objects.
Raum 1.1 §3 bumps the bridge to ~5-10M params specifically to absorb
this; if that still underfits, we fall back to DETR-style object
queries in 0.2.

---

## 2. Train with common objects (~60-class blob library)

### Goal
Expand the library and training vocabulary from the 6-shape hexad to
~30 (then ~60) everyday objects so the demo feels like a scene engine.

### Approach (locked)
- Source: **ShapeNet Core v2**, canonical mesh per class, offline
  conversion to Gaussian clouds. Full pipeline in
  `docs/plans/raum_11_plan.md` §4 (Stages A and B).
- No procedural low-poly shortcut. We considered it and chose
  ShapeNet because the pitch value of demo visuals outweighs the
  speed win from hand-authoring low-poly primitives.
- No text-to-3D distillation. Over-kill for 0.1; revisit if ShapeNet
  coverage is insufficient.

### Sharp version (ship first)
30 ShapeNet classes (Raum 1.1 stage 1.1.A + 1.1.B). Bridge retrained
at the 1.1 target scale.

### Harder version
60 classes (Raum 1.1 stage 1.1.E). Ships as part of 0.1.5 below.

### Tradeoff
Category curation is the only real cost at 30 classes; grows
modestly to 60. The architectural boundary (renderer handles indexed
blobs OR decoder-generated blobs, see Raum 1.1 §4 stage C) is built
in 0.1 so the 0.2 conditional-decoder upgrade is a drop-in.

---

## 3. Objects the model doesn't know (OOV)

### Goal
When the prompt contains "a red xylophone on a cube" and "xylophone"
is not in our blob library, do the right thing.

### Current behaviour (0.0)
If the blob head confidence is low, we flag the token unresolved and
surface a warning; nothing renders for that object.

### Four candidate policies
1. **Error-surface** (already in 0.0).
   Skip stamping; render the rest of the scene; show a warning.
   *Pro*: honest, zero surprise. *Con*: demo has a hole.
2. **Embedding nearest neighbour over blobs**.
   At inference, compute the GloVe embedding of the OOV word and pick
   the closest known blob by cosine similarity ("xylophone" → "cube").
   *Pro*: zero training change, gives something to render. *Con*:
   semantically rough.
3. **Generative splat blob** (deferred to 0.2; see Raum 1.1 §4
   stage C "conditional blob decoder"). Treat OOV words as a
   conditionally-decoded Gaussian cloud driven by the word embedding.
   Cleanest SGS-native story; needs new head + training.
4. **External LLM router**. Map OOV → nearest known class via a small
   local LM. Not SGS-native; explicitly out of scope.

### Sharp version
Ship (1) + (2) together: NN lookup with a cosine threshold; below
threshold, fall back to the unresolved warning. One evening of
work. Gives the demo graceful degradation.

### Harder version
(3) as the canonical path — the conditional blob decoder is a 0.2
milestone and a Raum 1.1 stage C architectural hook, not 0.1
infrastructure. Leave the renderer interface clean so 0.2 drops in.

### Tradeoff
(3) is the interesting story but the slowest path. (1)+(2) as default
with (3) later is the right default. (4) is explicitly rejected —
"small SGS model as the whole stack" is the pitch; bolting a vendor
LLM onto the demo undermines it.

---

## 4. Editable DSL (the nano-banana edit loop)

### Goal
The bridge's output is visible and editable. User types "a red car
behind a tree"; the demo renders the scene **and** shows the DSL that
produced it. User can edit either side — prompt or DSL — and re-render.
Prompt edits re-run the bridge; DSL edits re-run only the renderer.

### Architecture (simplified vs. earlier draft)
```
prompt ──► [frozen Planck 1.1] ──► [Raum bridge (1.1)] ──► DSL ──► renderer ──► splats
                                                  │
                                                  └── user can edit DSL directly,
                                                      bypass bridge, re-render
```

No Planner LM. No second model. No grammar-constrained JSON decoder.
The DSL is a **pure function of the bridge output** produced by
`src/raum/dsl.py::bridge_output_to_dsl(out)`. The same function is
the inverse of the bridge: if a user edits the DSL and we want the
bridge to stay self-consistent with it, we just skip the bridge and
run the renderer.

### DSL v1 (locked, see Raum 1.1 §5)
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

Schema in `src/raum/dsl.py` (new). `validate(dsl)` returns
`(is_valid, errors)` and is run on every edit. Unknown `version`
values are refused.

### Why we dropped the Planner LM
- The bridge already has to produce a DSL-shaped output internally.
  Training a second model (a Planck-class planner fine-tuned on
  `(prompt → DSL)`) to do the same thing is duplication at 0.1
  scale, not a capability gain.
- The earlier pitch for a Planner was semantic decomposition
  ("castle = towers + keep + battlements"). That is `group` nodes
  in the DSL, deferred to 0.2, and a conditional-blob-decoder
  (Raum 1.1 stage C) problem, not a separate-LM problem.
- Every schema boundary between components is a retest surface.
  Removing the bridge→Planner boundary removes a whole class of
  schema-drift bugs.
- We still validate Planck 1.1 is the right encoder via Satz 0.1
  before freezing it into Raum's stack.

### Demo UI
- Two tabs on the right panel: **Prompt** and **Scene graph
  (editable)**.
- Prompt tab: textbox + [Render] button. Edits run the bridge
  (and thus update the DSL below).
- Scene graph tab: editable JSON (monaco editor or textarea with
  syntax highlight). Edits run only the renderer. Save button
  pushes the edited DSL into the Prompt tab's "last DSL" so round-trip
  re-edits work.
- A "randomise" button re-samples the bridge at `temperature > 0`
  (we add small noise to the position head at inference; no
  retraining) to give the user variety without changing the prompt.

---

## 5. Rollout

| stage | scope | time | deps |
|---|---|---|---|
| 0.1.0 | 3-object data + extended direction loss | ~1 session | Raum 1.1.B |
| 0.1.1 | 30-class ShapeNet blob library live in demo | ~1 week | Raum 1.1.A + 1.1.B |
| 0.1.2 | OOV policy (NN-over-blobs + cosine gate, fallback warning) | ~1 session | 0.1.1 |
| 0.1.3 | DSL v1 frozen + read-only DSL panel in demo | ~1 week | 0.1.1 |
| 0.1.4 | Editable DSL tab (scene-graph edits re-run renderer only) | ~1 week | 0.1.3 |
| 0.1.5 | 60-class library + "randomise" + polish | ~1 week | 0.1.4, Raum 1.1.E |

Flip `10-raum-0-1` to `done` when 0.1.0 through 0.1.4 are live. 0.1.5
can land as a follow-up post.

---

## 6. Tradeoffs that outlived the 1-model pivot

- **Quality ceiling of a 5-10M bridge.** A mid-scale bridge asked to
  place 3 objects with a 60-class library will be *rough*. That's
  the honest pitch: small SGS model, editable DSL, honest renderer.
  Not a frontier LLM pipeline.
- **Two-step latency is gone.** One model, one render per prompt.
  DSL edits skip the bridge entirely — strictly faster than the old
  Planner+Executor design.
- **Schema drift is minimal.** Single DSL version, one producer (the
  bridge), one consumer (the renderer). Version the schema; executor
  refuses unknown versions; no retraining of a Planner on schema bumps.
- **Demo honesty.** Showing the DSL is still a feature. The
  editable graph turns "the bridge missed this" from a bug into the
  user's own input.

---

## 7. Open sub-questions

- **Where does "castle" live?** A composite object ("castle = towers
  + keep + battlements") still decomposes into a `group` node in the
  DSL, and `group` is a 0.2 feature. Ship without it first: on OOV
  composite words, fall back to the cosine-NN policy from §3.
- **`blob_confidence_threshold` global or per-blob?** The historical
  argmax attractors in 0.0 were `plane` and `torus`. With a
  ShapeNet 30-60 library, we re-measure the attractor distribution
  once 0.1.1 trains and set per-blob gates if a few classes dominate
  the argmax.
- **Naming.** The codebase alternates `template` / `blob` / `class`.
  DSL v1 standardises on `blob`. Existing code that says `template`
  stays as-is for migration hygiene; new code uses `blob`.
- **Does Raum share embeddings with Planck?** Yes, via the frozen
  Planck 1.1 encoder. That's the whole point of the 1-model
  architecture. Tokenizer stays Planck's SentencePiece; GloVe is
  only used by the blob-label side of the scene generator.
