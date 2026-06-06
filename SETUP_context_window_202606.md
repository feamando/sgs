# SGS — Context Window: limits, options, and the blob distinction (June 2026)

*Research spike, post-Hertz. NOT part of the Hertz 1.2 run — do not change
positional encoding mid-training. This captures how to raise the context window
beyond Planck's 256 / Hertz's 512, and what blobs do (and don't) for it.*

Code refs are against `src/sgs_lm.py` as of 2026-06-05.

## What sets the window today (the two walls)

Two independent limits, often conflated:

### Wall 1 — learned absolute positions (the correctness wall)

```python
self.pos_mu = nn.Embedding(max_len, d_s)          # sgs_lm.py:104
...
pos = torch.arange(L, device=device)
mu = mu + self.pos_mu(pos).unsqueeze(0)            # sgs_lm.py:485-486
```

Position is a **learned absolute embedding** added to each token's Gaussian
mean. Consequences:
- There is **no embedding row for position >= max_len**. Beyond the table it
  index-errors. So the window is a **hard wall, not a soft limit** — unlike
  RoPE/ALiBi models, SGS **cannot extrapolate at all** past `max_len`.
- The table is **learned**, so a longer window means **retraining** — you can't
  transplant a 512-trained `pos_mu` into a 2048 model.
- Inference already clips to the wall: `ctx = ids[:, -self.max_len:]`
  (`sgs_lm.py:595`).

### Wall 2 — O(L^2) render cost (the resource wall)

`forward` builds a full `[L, L]` causal mask (`torch.tril`, sgs_lm.py:489) and
the render passes materialize `[B*H, L, k, d_f]` (the tensor the accel
whitepaper flags as the OOM source). Cost scales **O(L^2)** in the rendering
passes. This is why the `train_hertz.py` comment justifies `context-len 512`
over 1024 as "4x smaller kernel matrices." Doubling L ~= 4x the render cost;
2048 ~= 16x.

**Current settings:** Planck `max_len=256`, Hertz `max_len=512`.

## Three routes to a larger window

### Route 1 — brute force: bigger table, train longer
Set `max_len=2048`, grow `pos_mu` to 2048 rows, train on 2048-token sequences.
- Works, but **requires retraining** and pays the full O(L^2) cost (~16x at
  2048). Straight into the OOM regime the accel doc warns about.
- Low ceiling, high cost. Only sensible for a modest bump (e.g. 512 -> 1024)
  if a true longer window is genuinely needed.

### Route 2 — extrapolable positional scheme (the real architectural fix)
Replace learned-absolute `pos_mu` with a **relative, table-free** scheme:
- **RoPE** (rotary): rotate Q/K by position; no table, extrapolates (further
  with position-interpolation / YaRN).
- **ALiBi**: linear distance bias on attention scores; cheap, extrapolates well.

This is how every long-context LLM does it. **SGS-specific open question:** today
position is *added to the Gaussian mean* `mu`, not applied to Q/K as in standard
attention. Whether rotary/relative positions compose cleanly with the
mean-modulation + Mahalanobis-kernel rendering is unknown and needs a small
experiment. **This is the right long-term lever** (raises the *true* window with
extrapolation, no hard wall) but it changes the architecture — a dedicated spike,
not a mid-run flag flip.

### Route 3 — blobs (the SGS-native lever; see next section)
Don't enlarge the attention window at all; expand **effective** context via
retrieval. Already partly built (`sparse_k` top-k, Mahalanobis selection at
sgs_lm.py:285; blob store + Faiss index).

## What blobs do — and don't — for context

Blobs **decouple effective context from the attention window.** Two mechanisms:

1. **Knowledge that never enters the window.** Normally, to use a fact the model
   must hold its tokens in the `max_len` window and pay O(L^2). With blobs the
   fact lives in the Faiss index as a pre-computed Gaussian and is retrieved
   top-k at inference — so the model *uses* far more information than `max_len`
   tokens **without those tokens ever costing attention**. Cost ~ O(L*k), not
   O(L^2).
2. **Conversation memory as blobs (the Planck 1.4 path).** Long history is
   compressed into blobs and retrieved rather than kept verbatim. "40% retrieval
   recall at 90 turns" through a 256-token window is exactly this.

**The honest distinction — say this out loud whenever the topic comes up:**

| | True attention window | Blob-augmented effective context |
|---|---|---|
| What it is | tokens that can directly attend to each other | knowledge retrievable into a short window |
| Cost | O(L^2) | O(L*k) retrieval + small window |
| Limited by | `pos_mu` table + render memory (Walls 1 & 2) | Faiss index size (cheap, CPU) |
| Good for | dense reasoning *across* a long span | recall the right fact/passage (QA, freshness) |
| Blobs help? | **No** | **Yes — this is the whole point** |

Blobs do **not** widen the true window: tokens still attend only locally over
`max_len`, and a long passage's tokens still can't all see each other beyond the
window. Blobs give **retrieval-augmented breadth**, not **long-range coherence**.
For "find and use the right knowledge" (the Hertz/Planck pitch) blobs win cheaply.
For long-document coherence or multi-step reasoning over a big input, you still
need Route 2.

## Recommendation

- **Hertz 1.2 run (now): keep `context-len 512`, unchanged positional encoding.**
  512 + blobs is the coherent story and matches the throughput/VRAM budget.
  Do not flip positional encoding mid-run.
- **Primary lever for "more context" = blobs.** Architecturally aligned, already
  built, O(L*k). Frame externally as *effective context*, not window size.
- **Post-Hertz spike (this doc's real deliverable): "Relative positions for SGS."**
  Prototype RoPE/ALiBi in place of `pos_mu`, test whether it composes with
  mean-modulation + Mahalanobis rendering, measure extrapolation past training
  length. If it holds, it's the one change that raises the *true* window.

## Spike plan (when picked up)

1. **Baseline:** confirm the wall — run a 512-trained checkpoint at L=600, show
   the index error / degradation. Document O(L^2) memory at L = 256/512/1024.
2. **ALiBi first** (cheaper to add than RoPE): add a distance bias to the
   Mahalanobis kernel scores; retrain a *small* Planck-scale model at L=512,
   eval perplexity at L = 512 and **L = 1024 (extrapolation)**. Gate: no
   collapse beyond training length.
3. **RoPE variant:** apply rotary to the query/key projections in
   `_render_pass` (not to `mu`); same eval. Compare against ALiBi.
4. **Decision:** if either extrapolates without quality loss, adopt for the next
   model generation (Hertz 1.3 / Helmholtz). If neither composes with the
   rendering kernel, stay on learned-absolute + lean entirely on blobs, and
   document that SGS's long-range ceiling is the window — blobs cover breadth.

Do NOT start this until Hertz 1.2 is trained and evaluated. It is a research
spike on the *next* generation, not a change to the in-flight run.
