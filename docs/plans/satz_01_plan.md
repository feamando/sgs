# Satz 0.1 — product demo for Planck LM + blob retrieval

*Status: draft plan. Written 2026-05-01, updated 2026-05-04 to make
Planck 1.3 + Wikipedia blobs the primary path; the Planck 1.1 +
TinyStories configuration is now the fallback, used only if 1.3 is
materially delayed. Prereq for using Planck as the Raum frozen encoder
(`docs/plans/d1c_raum_01_plan.md`). No new training — Satz 0.1 is a
thin web layer over artefacts produced elsewhere in the tree.*

Satz is the product track for **text**. Where Raum is the text → 3D
bridge and Klang is text → audio, Satz's job is to demo "Planck LM +
blob retrieval" as a standalone story: a small (~100M) model that can
be conditioned on external blobs at inference time without retraining.

Raum 0.1 (the 1-model rewrite) re-uses Planck as a frozen encoder.
Before we commit to that, we want a working demo that proves Planck +
blobs is already a believable standalone product. Satz 0.1 is that
demo. If Planck embeds prompts sensibly in this stripped-down setting,
it is also the correct encoder for Raum.

**Primary path: Planck 1.3 + Wikipedia blobs.** TinyStories-trained
Planck 1.1 was the right proof that blobs work; it is the wrong
substrate for a "reasoner + knowledge via blobs" demo. Wait for
Planck 1.3's Wikipedia base and the 1.3.1a blob index
(`docs/plans/planck_13_plan.md`). If 1.3 is materially delayed, the
Planck 1.1 + TinyStories configuration is the fallback and must be
flagged as a placeholder in the UI.

Everything below is a plan, not a commitment.

---

## 1. Scope

### Goal
Local web app that:
- loads `checkpoints/planck11/best.pt` and the TinyStories tokenizer,
- loads the existing blob index (`build_blobs.py` output),
- accepts a free-form prompt in a textbox,
- streams generated text in the main panel,
- shows a right-panel list of retrieved blobs with transmittance
  weight bars (which blobs the render over the prompt tokens actually
  attended to, not just top-k cosine),
- exposes a `blob-k` slider so a user can see how retrieval depth
  changes the output live.

### Non-goals for 0.1
- No fine-tuning, no new training run, no new datasets.
- No 3D viewer (that's Raum's job).
- No editable blobs / blob authoring UI (that's a 0.2 feature).
- No live RSS blobs (Planck 1.3.1b ships that later). Satz 0.1
  consumes the static Wikipedia blob index built by Planck 1.3.1a.

### Why now
- Validates Planck 1.1 works as a standalone LM before it becomes
  Raum's text encoder. If Planck 1.1 is already the wrong encoder,
  Raum 0.1 changes shape and we'd rather find out before writing
  `scripts/train_raum_planner.py`.
- Ships product-track value on the **Satz** swimlane, which is
  currently "TBD" on the roadmap.
- 2-3 sessions of work, zero GPU-weeks.

---

## 2. Architecture

```
prompt ──► [Planck 1.1 LM] ──► generated text
             │
             └─► (internal) blob retrieval over TinyStories blobs
                 with per-step transmittance weights
                             │
                             ▼
                     right-panel UI: top-K blobs
                     + transmittance weight bar per blob
                     + slider to change K live
```

One process, one model, one tokenizer, one blob store, no external
services. The app is a FastAPI backend + static HTML frontend, mirror
of `demo/app.py`'s shape but without Three.js.

### Components
- **Backend** — `satz/app.py` (new). Pattern copied from
  `demo/app.py`: argparse → runtime object → FastAPI routes
  (`GET /`, `GET /health`, `POST /generate`). Reuses
  `scripts/generate.py::infer_arch`, `load_model`, `generate_text`
  verbatim where possible (import, don't copy).
- **Blob retrieval hook** — thin wrapper around
  `src/blob_store.py` that returns retrieved blobs **plus** their
  per-step transmittance weights. The existing blob path already
  attends to blobs; we expose that attention instead of hiding it.
- **Frontend** — `satz/static/index.html`, `app.js`, `style.css`.
  Two columns: left = prompt textbox + streaming response, right
  = retrieved-blob list with a weight bar per blob and a `k` slider.
  Streaming via Server-Sent Events (or a chunked POST; whichever is
  least invasive given `SGSLanguageModel.generate`).

### What exists vs. what's new

| Component | Status |
|---|---|
| `checkpoints/planck11/best.pt` | **exists** (Planck 1.1, shipped) |
| `src/blob_store.py` | **exists** (blob storage + retrieval) |
| `scripts/build_blobs.py` | **exists** (blob builder, TinyStories) |
| `scripts/inspect_blobs.py` | **exists** (blob debug tool) |
| `scripts/generate.py` | **exists** (sampling, tokenizer load) |
| `demo/app.py` + `demo/static/` | **exists** (reference scaffold) |
| `satz/app.py` | **new** (~150 LOC mirrored from demo/) |
| `satz/static/index.html` | **new** (single-page, two-column) |
| `satz/static/app.js` | **new** (fetch + stream render) |
| Transmittance-weights API on blob store | **new** (small hook) |

### Why we do NOT copy `demo/app.py`
`satz/app.py` and `demo/app.py` have different payloads (text stream
vs. splat cloud) and different frontends (no Three.js for Satz).
Sharing a helper module would couple two independent product tracks
for no gain at 0.1 scale. Revisit after 0.2 if shared auth / logging
appears.

---

## 3. UI

```
┌────────────────────────────────────────────────────────────────┐
│ Satz 0.1                                                       │
├──────────────────────────────┬─────────────────────────────────┤
│                              │  Retrieved blobs (k = 8)        │
│  Prompt                      │  ┌─────────────────────────────┐│
│  ┌────────────────────────┐  │  │ #42 knight_story_3       ▓▓▓││
│  │ Once upon a time a     │  │  │ #17 forest_story_11      ▓▓ ││
│  │ knight ...             │  │  │ #08 dragon_story_6       ▓  ││
│  └────────────────────────┘  │  │ ...                         ││
│    [Generate]                │  └─────────────────────────────┘│
│                              │                                 │
│  Response                    │  k: [==========•  ] 8           │
│  The knight walked into the  │                                 │
│  forest and met a dragon ... │  temperature: 0.8  top-k: 50    │
│                              │                                 │
└──────────────────────────────┴─────────────────────────────────┘
```

Key interactions:
- **Prompt textbox**: plain textarea, Enter to generate.
- **Response area**: tokens stream in as they're sampled (best-effort;
  if streaming is awkward, settle for complete-response display in
  0.1.0 and stream in 0.1.1).
- **Blob list**: top-K retrieved blobs, sorted by per-step
  transmittance weight averaged over the generated tokens. Each row
  shows blob id, short name (from blob metadata), and a proportional
  bar.
- **k slider**: integer, 1 to 32. Re-runs generation on release (do
  not fire per keystroke). Shows the name of the weakest-admitted
  blob live so the user intuits "4 blobs is basically just the
  top one, 16 is noise".
- **Temperature / top-k**: editable inline, default 0.8 / 50
  (`generate.py` defaults).

---

## 4. Rollout

### 0.1.0 — minimum viable local demo (~1 session)
- `satz/app.py` loads Planck 1.1 + tokenizer, serves `/generate`
  with a non-streaming response. No blobs yet.
- `satz/static/index.html` renders prompt box + response box.
- Gate: type a prompt, get a full TinyStories-style continuation.
  Manual inspection only; same sanity check we'd run from
  `scripts/generate.py --interactive`.

### 0.1.1 — blob panel (~1 session)
- Wire `src/blob_store.py` to return retrieved-blob metadata +
  per-step attention weights. If the weights aren't already
  observable in the current blob path, expose them via a small
  hook (return them alongside the generated ids; the blob path
  already computes them internally).
- Right-panel list + weight bars.
- `k` slider re-runs generation on release.
- Gate: for the prompt "the knight met a dragon", the top retrieved
  blob should be a knight/dragon blob, not a cow/forest blob. Sanity
  only.

### 0.1.2 — streaming + polish (~1 session)
- Token-level streaming via SSE or chunked response. `SGSLanguageModel.generate`
  is batched / all-at-once today, so this may need a
  `generate_stream` variant that yields after each sampled token.
  If that's more than ~50 LOC of surgery, defer to 0.2.
- Style pass on the frontend so it's not raw HTML.
- Short GIF for a LinkedIn post if the blob story comes through
  clearly on-screen.

### Stop conditions
- If Planck 1.1 generation quality is too weak to demo honestly
  (e.g. it never stays on topic), mark the swimlane "blocked on
  Planck 1.3" and stop. Don't paper over a weak base model with
  UX polish.
- If blob attention weights are too flat to be visually informative
  (all bars ≈ 1/k), the UI is fine but the story is weak; log it
  as evidence for the Planck 1.3 freshness pitch and ship anyway
  with a caveat.

---

## 5. Files touched

| Path | Change |
|---|---|
| `satz/app.py` | new, FastAPI server, ~150 LOC mirroring `demo/app.py` |
| `satz/static/index.html` | new, two-column layout |
| `satz/static/app.js` | new, fetch + render + slider |
| `satz/static/style.css` | new, minimal |
| `satz/requirements.txt` | new, FastAPI + uvicorn + sse-starlette if streaming |
| `src/blob_store.py` | +1 hook: `retrieve_with_weights()` returning `(blob_ids, weights_per_token)` |
| `scripts/generate.py` | no change (imported by `satz/app.py`) |
| `SETUP_202605.md` | new §3 inserted between Klang 1.3 and Raum |
| `roadmap.md` | new row `11-satz-0-1` |

---

## 6. Success metrics

- **Functional**: user types a prompt, sees a generated continuation,
  sees ≥ 1 non-empty blob in the right panel with non-zero weight,
  can change `k` and see the list re-rank. All on `localhost`.
- **Story**: at least one demo-worthy prompt where the top blob is
  obviously relevant to the generated text (used as the screenshot
  for the LinkedIn post).
- **Cheap**: total work ≤ 3 sessions. If 0.1.1 takes more than 2
  sessions because the blob-weight hook is harder than expected,
  stop and reassess before starting 0.1.2.

---

## 7. Open questions

- **Blob-weight hook cleanliness.** `src/blob_store.py` already
  computes retrieval scores; do we also have per-generated-token
  attention weights over the retrieved set, or only at encode time?
  If only at encode time, display those instead of per-step weights
  and note the limitation in the UI caption. Don't invent a new
  attention path just for the demo.
- **Blob bundle provenance.** Primary bundle is Planck 1.3.1a's
  Wikipedia lead-section index. Fallback is the Planck 1.1
  TinyStories bundle, flagged in the UI as placeholder-only.
- **Prompt length limits.** Planck 1.1's `max_len` is 512 tokens;
  textarea must enforce that (or truncate with a visible warning).
- **Safety / content.** No filtering at 0.1. TinyStories is benign
  so we don't need a moderation layer; if we ship the Wikipedia
  bundle later we revisit.
