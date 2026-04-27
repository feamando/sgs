# Planck 1.4 — Conversation-memory blobs

*Status: stub plan. Written 2026-04-27. Depends on Planck 1.3.2a (blob
retrieval proven end-to-end on fresh knowledge). Reuses the retrieval stack
from 1.3.*

The blob concept so far has always been **write-once, read-many**: blobs are
built from a static curriculum (1.1) or a slow-moving news feed (1.3). Planck
1.4 asks whether the same machinery can absorb **per-turn conversation
history**, so a long chat no longer needs to re-send the full context window
every turn.

The pitch: instead of a chat whose cost grows quadratically in turn count
(every turn sends all prior turns), every user/assistant pair becomes a
blob. Retrieval pulls back only the relevant history plus the last few
turns verbatim. **Cost per turn stays roughly flat regardless of
conversation length.**

This is close to MemGPT / Letta / long-term-memory RAG; the novelty is that
Planck's blob path is already baked into the model, so we aren't gluing a
RAG wrapper onto a frontier LLM, we're letting a small model do what the
frontier stack does with external scaffolding.

**Important framing**: this does not "solve" the context window. The current
prompt, retrieved blobs, and output still live in-context. What it solves is
the *growth* of in-context memory with conversation length. The last 2-3
turns still need to be verbatim because similarity retrieval is systematically
worse than recency for "what did I literally just say".

Everything below is a plan, not a commitment.

---

## 1.4.0 — Per-turn blob writer

### Goal
Every `(user_msg, assistant_reply)` pair becomes a blob at the end of each
turn, tagged with `{session_id, turn_idx, ts}`.

### Architecture
- Reuse the Planck blob-embedding head from 1.1 / 1.3
- Blob store is per-session (not shared across users) with a separate
  namespace from the news store
- Summarization option: for long turns, embed a short auto-summary
  alongside the raw text so retrieval can hit either

### Deliverable
A chat wrapper that writes one blob per turn and exposes the store for
the next turn's retrieval step.

### Tradeoff
Embedding every turn adds ~50-100ms latency per turn. Acceptable. The
harder call is whether to blob the *raw* turn, a *summary*, or both —
raw is lossless but noisy, summary is clean but distorts. Start with raw
+ auto-summary, drop the summary if it doesn't help retrieval quality.

---

## 1.4.1 — Hybrid recency + similarity retrieval

### Goal
At prompt time, build the effective context from three sources:
1. **Last N turns verbatim** (N = 2-3) — because "you just said X" needs
   exact recall, not semantic fuzzy match
2. **Top-k older turns by similarity** to current query, gated by a
   recency-weighted score
3. **System prompt + current user turn**

### Retrieval score
`score(blob) = cos_sim(query, blob) * exp(-decay * age_turns)` — tune
`decay` so old context can still surface when strongly relevant but
doesn't dominate.

### Deliverable
A retrieval module that emits the final in-context string, with an
ablation switch for (a) no retrieval, truncate at N turns; (b) retrieval
with no recency bias; (c) hybrid.

### Tradeoff
The most common failure mode for memory-RAG is "lost-in-the-middle" — the
retrieved blob gets buried and the model ignores it. Worth measuring how
position in the prompt affects usage. May need to re-order by recency
after retrieval, not by similarity score.

---

## 1.4.2 — Long-conversation eval

### Goal
Prove that cost-per-turn is flat and quality degrades gracefully as
conversations grow long.

### Benchmark construction
Use or adapt one of:
- **LongBench-Chat** — existing multi-turn long-context benchmark
- **Needle-in-a-haystack-conversation** — inject a specific fact at turn
  10, ask about it at turn 100; measure recall. Build locally, it's
  mechanical.
- Synthetic generator: chain of user turns that reference prior turns
  ("remember when I said X? now do Y with it")

### Metrics
- Recall@needle-turn — did the fact from turn 10 survive to turn 100?
- Cost-per-turn — tokens processed per turn, plotted vs. conversation length
- Quality — LLM-judged coherence vs. a full-context baseline up to its
  context limit

### Baselines
1. Planck 1.4 with hybrid retrieval (the thing)
2. Planck 1.4 with truncation-only (the naive small-model baseline)
3. A frontier model with full context (until it hits its own limit)
4. A frontier model + a MemGPT-style external memory wrapper

### Deliverable
A chart showing cost-per-turn flat vs. growing, and quality holding up
at 100+ turns. If both land, this is a paper or a loud LinkedIn post.

### Tradeoff
Needle benchmarks are easy to game. The real test is *actual* long chats
with a person, where the memory fails in subtle ways (wrong emotional
context, stale preferences). A second-phase eval with human raters on
~20 real long conversations is worth planning for once the synthetic
numbers are clean.

---

## Rollout

1. **1.4.0** per-turn blob writer, wire into the Raum demo's chat panel
   (or build a bare chat harness) — 3-5 days once 1.3 retrieval is stable
2. **1.4.1** hybrid retrieval with ablation switches — 3-5 days
3. **1.4.2a** needle-in-conversation benchmark — 1 week
4. **1.4.2b** real long-chat human eval (optional, gated on 1.4.2a)

Gate to ship: needle recall at turn ≥ 100 must be > 60% and cost-per-turn
slope must be < 10% of the full-context baseline's slope. Below that, the
pitch doesn't hold.
