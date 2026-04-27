# Raum 0.1 — compositional scenes, richer vocabulary, OOV

*Status: draft plan. Depends on 0.0 polish landing (labels, splat density, unresolved
warning, cylinder seam, splat logo). Written 2026-04-26.*

Raum 0.0 proved that a text → template-routing bridge trained on analytic labels
produces clean two-object scenes from the six-shape template library. Four things
keep it from being interesting to show off:

1. **Scene complexity** — it only handles "A \[rel\] B" patterns and cannot compose
   more than 2 objects or nested relations.
2. **Vocabulary** — the object set is a math-y hexad (sphere/cube/cylinder/cone/
   plane/torus). A prospect wants to type "a red car behind a tree".
3. **OOV handling** — unknown object words get silently routed to the argmax
   template (we just put a warning band around that in 0.0; the real fix is a
   policy).
4. **Interpretation**. Even with 30 templates and OOV fallback, the bridge is
   one end-to-end model. A user typing "castle on a hill in the jungle" needs
   *semantic decomposition* (what *is* a castle in terms of primitives I have)
   before placement. That is a planner-executor split, covered in §4.

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

## 4. Planner + executor split (free-form prompts)

### Goal
User types "castle on a hill in the jungle" → renders a 3D Gaussian-splat
scene from that free-form prompt, not from a grammar-constrained one.
The user can then edit the scene by touching either the prompt or the
intermediate scene graph. Mental model: nano-banana for SGS scenes.

### Current limit
The 0.0/0.1 bridge is a single model that goes straight from tokens to
render parameters. It only understands words that exist in its template
vocabulary and relation words it was trained on. "Castle" and "jungle"
are OOV and decompose into nothing.

### Architecture
Two models in series, both SGS-native (Planck/Hertz family, **not**
external LLMs).

```
prompt ──► [Planner]  ──► scene_graph ──► [Executor]  ──► Gaussian cloud
"castle on   (Planck-      (DSL JSON)     (Raum 0.1         (existing
 a hill in    class LM,                    bridge + 30       viewer.js
 the jungle") instruction-                 templates)        consumer)
              tuned)
```

- **Planner.** A small Planck-class LM (Hertz for the bigger version)
  fine-tuned on `(prompt → scene_graph)` pairs. It does the *semantic
  decomposition*: "castle" → a composition of known primitives
  (towers = tall cylinders, keep = cube, battlements = small cubes
  on top); "hill" → shallow large sphere half-buried; "jungle" →
  crowd of "tree" templates. It is trained to emit a fixed-schema
  DSL the executor consumes, not to render.
- **Executor.** The Raum 0.1 bridge from §1-§3 of this plan, extended
  to consume the DSL as input instead of a grammar-constrained prompt.
  It already handles placement, relations, and the 30-object template
  library. Its job shrinks from "interpret English" to "place these
  labelled primitives with these relations".
- **Edit loop.** The DSL is human-readable. The demo surfaces it as a
  collapsible JSON / visual node tree. Editing any node re-runs only
  the executor, not the planner, so edits are fast.

Why our own models: (a) cost, every prompt hitting a frontier API is
~cents × demo traffic, plus vendor lock-in on what should be the
signature of the SGS pitch; (b) data flywheel, prompts the planner
handles badly become training data for the next version; (c) thesis
consistency, "small SGS model outperforms frontier RAG" only holds
if the planner is ours too.

### DSL sketch
Small JSON schema, versioned so the executor can reject unknown
versions rather than misinterpret them:

```json
{
  "version": 1,
  "objects": [
    {"id": "castle",  "template": "castle",      "color": "#888", "scale": 1.0},
    {"id": "hill",    "template": "hill",        "color": "#6a4", "scale": 2.0},
    {"id": "tree_*",  "template": "tree",        "color": "#263", "scale": 0.7,
     "repeat": {"count": 30, "jitter": 0.4}}
  ],
  "relations": [
    {"subject": "castle",  "rel": "on_top_of",   "anchor": "hill"},
    {"subject": "tree_*",  "rel": "scattered_around", "anchor": "hill"}
  ]
}
```

Keys the executor must handle: `template` (string in known library or
OOV), `relations` (extended set from §1, including `scattered_around`,
`inside`, `on_top_of`, etc.), `repeat` (multi-instance stamping with
jitter), and `color`/`scale` overrides. The planner model never
invents relation names; if it isn't sure, it writes a relation
allowed by the grammar and marks `confidence` low.

### Proposed changes
- **Data pipeline (planner training).** Build a dataset of
  `(free_prompt, scene_graph_json)` pairs. Sources, in priority order:
  1. **Procedural**: for each template object in the §2 library, write
     a prompt generator that produces 5-20 phrasings per scene config
     ("a red car", "red car behind tree", "a castle standing on a hill
     with trees around it"). Pair with the canonical DSL. Free and
     large; scales with the grammar we already own. ~50k samples from
     the §1 grammar extended with paraphrase templates.
  2. **LLM-bootstrapped** (one-time): have a frontier LLM (yes, once)
     generate `(prompt, scene_graph)` pairs for harder compositions,
     e.g. "a medieval village at dawn" → a multi-object graph. We manually
     filter ~2-5k gold pairs and never ship the frontier model in the
     loop.
  3. **User-corrected**: every time a user edits a scene graph in the
     demo, the `(prompt, final_graph)` pair is eligible for the next
     training round.
- **Planner architecture.** Planck-class LM (n=100M-ish for 0.1) with
  a grammar-constrained decoder so it can only emit valid JSON in the
  DSL schema. For 0.1 we use a `jsonformer`-style constrained
  sampler; in 0.2 we can bake the grammar into training with loss
  masking. Reusing Planck's tokenizer + checkpoint + `generate()`
  means zero new infra; we only add a fine-tune script
  `scripts/train_raum_planner.py`.
- **Executor changes.** Swap the `src/raum/data.py` tokenizer for a
  DSL → tensors converter. Keep per-object template routing, relation
  head, and rendering untouched. Adds `repeat` stamping and a small
  relation set beyond the 0.1 baseline.
- **Demo.** Split the right-side panel into "prompt" + "scene graph
  (editable)" tabs. Prompt edits re-run planner + executor; graph
  edits re-run executor only. Add a "randomise" button that samples a
  new planner output without changing the prompt (temperature > 0).

### Sharp version (ship first)
- Planner is the existing Planck 1.1 checkpoint fine-tuned on the
  procedural dataset only. No frontier-LLM bootstrap yet.
- DSL supports: template, color, scale, relations already in §1,
  `repeat` with count-only jitter.
- Executor consumes DSL, drops any node it doesn't understand with a
  warning (reuses 0.0's unresolved pathway).
- Demo exposes the DSL as read-only JSON below the prompt, no graph
  edit UI.
- Success metric: 70% of procedural-set held-out prompts produce a
  DSL that renders without unresolved warnings and matches at least
  the dominant relations in ground truth.

### Harder version (follow-up, still in 0.1 per user scope call)
- LLM-bootstrapped gold pairs added to training set for
  hard-composition coverage.
- Editable graph UI in the demo (node tree with add/remove/swap
  template).
- Planner gets a reasoning-trace output (short natural-language
  "here's what I think you mean" string) before the JSON, for
  debuggability.
- Hertz-class planner (bigger) swappable via config once Hertz 1.2
  exists.

### Tradeoffs
- **Quality ceiling of a Planck-class planner.** A 100M LM asked to
  decompose "medieval village at dawn" will be *rough*. Accept this as
  the known limitation of 0.1; the pitch is that quality improves
  with the SGS model family over time (Hertz 1.2 planner is the
  obvious next step). Framing the demo honestly ("this is a small
  model decomposing your prompt, not a frontier model") is the
  correct positioning.
- **Two-step latency.** Plan + render is slower than single-pass
  routing. For interactive editing we cache the last scene graph and
  only re-run the executor on graph edits; the planner runs once
  per prompt change.
- **Schema drift.** If the DSL evolves, old planner checkpoints
  produce invalid graphs. Version the DSL; executor refuses unknown
  versions; planner training data is re-paired from the canonical
  generator on every schema bump.
- **Demo honesty.** A novice user sees the prompt and expects
  magic; showing the intermediate DSL is a feature (explainability)
  but also reveals the crudeness of the decomposition. The edit UI
  turns this from a bug into a selling point: *user* fills in what
  the planner couldn't.

### Open sub-questions
- **Tokenizer mismatch.** The planner's tokenizer (Planck SentencePiece)
  and the executor's GloVe-seeded object embeddings don't share a
  vocabulary. We don't need them to (the DSL is the interface), but
  the planner's `template` values must exactly match executor labels.
  Enforce via a grammar constraint on the planner's JSON decoding.
- **Where does "castle" live?** If "castle" is not in the 30-object
  template library, the planner must decompose it into a *composition*
  of known templates. That means the DSL needs a `group` node ("a
  castle is: [tall cylinder, tall cylinder, cube on top, ...] arranged
  by these relations"). Implementing `group` is roughly another two
  weeks of executor work and another paragraph of planner training
  data. Ship without it first: planner falls back to "unresolved"
  for unknown-composite words; then add groups as the next step.

---

## Rollout order (suggested)

1. **0.1.0**. Data-generator emits 3-object scenes; bridge learns; analyzer
   verifies 3-object direction accuracy; no vocab change. (~1 session)
2. **0.1.1**. Procedural library of 30 common objects; retrain bridge;
   regenerate analyzer fixtures; update demo to show the new object names.
   (~1 week with template authoring)
3. **0.1.2**. OOV policy: NN-over-templates with cosine gate; update
   unresolved-warning copy to say "closest guess: X (%)". (~1 session)
4. **0.1.3**. DSL v1 frozen; executor rewritten to consume DSL; existing
   bridge wrapped by a trivial "prompt → DSL" shim (rule-based) so we can
   validate the executor end-to-end before the planner exists. (~1 week)
5. **0.1.4**. Procedural `(prompt → DSL)` dataset + planner fine-tune
   (Planck 1.1 checkpoint). Ship sharp-version planner + read-only DSL
   panel in the demo. (~2 weeks)
6. **0.1.5**. Editable scene-graph UI in the demo; planner gains
   reasoning-trace output; user-corrected pairs start accumulating.
   (~1 week)
7. **0.1.6**. LLM-bootstrapped gold pairs + retrain planner; `group`
   nodes for composite objects. (~2-3 weeks)
8. **0.2**. Hungarian matching / DETR-style object queries, nested
   relation parsing, conditional-blob OOV decoder, Hertz-class planner.
   Write its own plan doc when 0.1 ships.

## Open questions

- Do we want Raum to share embeddings with Planck? The blob-plugin thesis
  says yes; the Raum-only training regime says no. Revisit once 0.1.1 is in.
- Should `template_confidence_threshold` be global or per-template? Plane
  and torus are the historical argmax attractors in 0.0; they may need a
  higher gate.
- Naming: is "template" still the right word once we have 30+ objects, or do
  we start calling them "blob classes"? Align with the v5 pitch terminology.
