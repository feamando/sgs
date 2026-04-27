# Raum 0.1 — compositional scenes, richer vocabulary, OOV

*Status: draft plan. Depends on 0.0 polish landing (labels, splat density, unresolved
warning, cylinder seam, splat logo). Written 2026-04-26.*

Raum 0.0 proved that a text → template-routing bridge trained on analytic labels
produces clean two-object scenes from the six-shape template library. Three things
keep it from being interesting to show off:

1. **Scene complexity** — it only handles "A \[rel\] B" patterns and cannot compose
   more than 2 objects or nested relations.
2. **Vocabulary** — the object set is a math-y hexad (sphere/cube/cylinder/cone/
   plane/torus). A prospect wants to type "a red car behind a tree".
3. **OOV handling** — unknown object words get silently routed to the argmax
   template (we just put a warning band around that in 0.0; the real fix is a
   policy).

Everything below is a plan, not a commitment. Each section states the goal,
the cheapest sharp version, a harder version, and the main tradeoff.

---

## 1. Complex scenes from prompt context

### Goal
Render scenes with 3+ objects, longer sentences ("a small red sphere above a
large blue cube to the left of a green cone"), and nested / conjoined relations.

### Current limit
Per-token position/template/color/scale heads plus a pairwise-direction loss
restricted to slots (0, 1). The data generator builds only 2-object phrases.

### Proposed changes
- **Data generator**: extend `src.raum.data.build_dataset` to emit N-object
  scenes, N sampled in \{1, 2, 3, 4\}. Relation words still anchor pairs, but
  with an "anchor object" pointer so "C to the right of B" uses B as anchor,
  not A.
- **Object slot assignment**: keep the per-token object predictions but
  replace the slot-0/slot-1 bias in the pair loss with an **order-invariant
  Hungarian match** between predicted object tokens and ground-truth slots
  (this is the standard DETR-style set-loss trick). That removes the
  "objects must appear in sentence order" hack.
- **Relation graph head**: add a small per-pair head that classifies relation
  between any two object tokens. Supervise from the generator's relation
  labels. This lets the analyzer verify "C is right-of B" at inference.

### Sharp version (ship first)
- N in \{1, 2, 3\}; keep 2-hop relations ("above", "left of", "on") only.
- Still train with analytic labels (no renderer).
- Just one anchor per sentence, so no Hungarian matching yet; stamp in
  sentence order but extend the direction loss to all consecutive object
  pairs.
- Success metric: \>90% direction accuracy on 3-object scenes in val set.

### Harder version (later)
- Nested phrases ("the cube that is below the cone above the plane"),
  quantifiers ("three red spheres"), reference resolution ("the small one").
- Needs a proper parse or LLM-in-the-loop for scene graph extraction.
- Probably a separate Raum 0.2.

### Tradeoff
Longer sequences mean more tokens routed to objects; softmax over N_OBJECTS
templates gets noisier when the context window has 3 objects. We need
per-token L2 regularization or a "query slot" (learned latent) design, which
pushes architecture toward DETR-style object queries. Worth it only once
1-and-2 object scenes are reliable on new object vocabulary.

---

## 2. Train with common objects

### Goal
Expand the template library and training vocabulary from the 6-shape hexad to
~30-60 everyday objects ("car", "tree", "chair", "bottle", "cup", "dog", ...)
so the demo feels like a scene engine, not a geometry set.

### Blockers
- We need a Gaussian template for each object. Making them by hand (as we did
  for the hexad) does not scale.
- The category labels must match GloVe words for the vocab-side embedding to
  be useful.

### Three candidate template sources
1. **Procedural low-poly templates** (cheapest).
   Hand-author a library of ~30 primitive-composed shapes ("car" = cuboid
   body + 4 cylinders as wheels). Each becomes a `GaussianTemplate` built
   from its primitives. ~1-2 days to author 30; looks OK from far away,
   hacky up close.
2. **ShapeNet → point cloud → Gaussian fit** (medium).
   Use a couple of ShapeNet categories, sample N surface points with
   `pytorch3d` or `trimesh`, convert to a Gaussian cloud with per-point
   log-scale from local neighbour radius. One class = one canonical mesh
   (class mean), which keeps the demo simple (no intra-class variance yet).
   ~3-5 days.
3. **Text-to-3D distillation** (hard).
   Run DreamFusion / MVDream / InstantMesh once per category, dump as
   Gaussian clouds. Best visuals; heavy infra; slow iteration.

### Sharp version
Go with (1) for speed. Build 30 templates. Keep the bridge architecture as
is (it's agnostic to template count: just change `n_templates`). Retrain on
a generator that pairs each new object with plausible relations ("tree
behind car", "cup on chair"). Needs a small grammar file declaring
object→allowed-relation → allowed-object triples (e.g. cup can go "on" a
table but not "on" a sky).

### Harder version
Move to (2) for better visuals once (1) lands. ShapeNet gives us ~55 common
categories with CC-BY variants. We pick one canonical mesh per category,
dump ~1500 Gaussians, and that's the new template library.

### Tradeoff
(1) is fast but looks crude; (2) is slow upfront but gives us
compellingvisuals for recording demos / pitch updates. If the goal is a
screenshot for the pitch deck, skip (1) and do (2) directly. If the goal is
iterating on the bridge architecture, (1) is correct because template
changes don't affect learning.

---

## 3. Objects the model doesn't know

### Goal
When the prompt contains "a red xylophone on a cube" and "xylophone" is not
in our template vocabulary, do the right thing.

### Current behaviour
(With 0.0 fixes) If the template head confidence is low we flag the token
unresolved and surface a warning. Nothing renders for that object.

### Four candidate policies
1. **Error-surface** (already in 0.0).
   Skip stamping; render the rest of the scene; show a warning.
   *Pro*: honest, zero surprise. *Con*: demo has a hole.
2. **Embedding nearest neighbour over templates**.
   At inference time, compute the GloVe embedding of the OOV word and pick
   the closest known template by cosine similarity (e.g. "xylophone" → "cube"
   via its feature embedding). Confidence becomes the cosine score.
   *Pro*: zero training change, gives something to render. *Con*: semantically
   rough; "xylophone" may map to "plane" depending on GloVe geometry.
3. **Generative splat blob**.
   Treat OOV words as a "blob" of Gaussians whose shape is driven by the
   word's feature vector through a small conditional decoder. Each unknown
   object becomes a unique amorphous splat coloured by the predicted colour
   head. This is the cleanest SGS-native story (it plays directly into the
   blob-plugin design in our v1.1 roadmap).
   *Pro*: every unknown renders as something, and it visually signals
   "I don't know what this is". *Con*: needs a new head + train loop.
4. **LLM router**.
   Use a small local LM (e.g. Phi-mini) at inference to map OOV → closest
   known template category ("xylophone" → "plane"). Not SGS-native, but fast
   to wire; useful as a fallback.
   *Pro*: easy to ship. *Con*: external dependency, not the story we're
   telling in Raum.

### Sharp version
Ship (1) + (2) together: try NN lookup with a cosine threshold; if below
threshold, fall back to the unresolved warning. This is maybe 1 evening of
work and gives the demo graceful degradation.

### Harder version
(3) as the canonical path — small conditional Gaussian decoder trained
with the rest of the bridge. This is a real research direction that
motivates the blob-plugin thesis in the v5 pitch.

### Tradeoff
(3) is the interesting story but the slowest path. (2) is a cheap
visual win that is honest ("top guess"). (4) is the fastest, most
predictable demo but has nothing to do with SGS. The combination of
(1)+(2) as default with (3) later is the right default.

---

## Rollout order (suggested)

1. **0.1.0** — data-generator emits 3-object scenes; bridge learns; analyzer
   verifies 3-object direction accuracy; no vocab change. (~1 session)
2. **0.1.1** — procedural library of 30 common objects; retrain bridge;
   regenerate analyzer fixtures; update demo to show the new object names.
   (~1 week with template authoring)
3. **0.1.2** — OOV policy: NN-over-templates with cosine gate; update
   unresolved-warning copy to say "closest guess: X (%)". (~1 session)
4. **0.2** — Hungarian matching / DETR-style object queries, nested relation
   parsing, conditional-blob OOV decoder. Write its own plan doc when 0.1 ships.

## Open questions

- Do we want Raum to share embeddings with Planck? The blob-plugin thesis
  says yes; the Raum-only training regime says no. Revisit once 0.1.1 is in.
- Should `template_confidence_threshold` be global or per-template? Plane
  and torus are the historical argmax attractors in 0.0; they may need a
  higher gate.
- Naming: is "template" still the right word once we have 30+ objects, or do
  we start calling them "blob classes"? Align with the v5 pitch terminology.
