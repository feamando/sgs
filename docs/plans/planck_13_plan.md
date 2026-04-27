# Planck 1.3 — Fresh-knowledge blobs from live news

*Status: stub plan. Written 2026-04-27. Depends on Planck 1.2 (acceleration
recipe) landing so 1.3 retrains are affordable.*

Planck 1.1 proved that blobs work: a small LM (~100M params) can be conditioned
on externally-built blobs and meaningfully use their content at inference time
without retraining. Planck 1.3 pushes the same concept to **freshness** —
instead of static curriculum blobs, the blob store is continuously rebuilt
from live news feeds, so a cheap base model can answer questions about events
that happened in the last hour.

The pitch: a 100M-param Planck + live RSS blob store should outperform a
frontier model (Gemini 3 / GPT-5) on 24-hour-fresh factual QA, because
frontier models can't retrain that fast and their public web tools are slow,
rate-limited, or both.

Everything below is a plan, not a commitment.

---

## 1.3.0 — Generic-grammar base retrain

### Goal
Replace the TinyStories-style training distribution with a corpus that gives
the model broad grammatical competence without domain-specific bias. The base
model should be a neutral "reasoner" that leans on blobs for facts.

### Candidate corpora
- **BabyLM 10M / 100M** — deliberately human-scale, grammar-first
- **C4 (filtered)** — broad web English, drop news-adjacent domains to
  avoid pretraining leakage into the freshness eval
- **WikiText-103** — clean encyclopedia English, too narrow alone but useful
  as a mix component

Cheap first pass: BabyLM-100M. Harder version: 50/50 BabyLM + filtered C4.

### Deliverable
A Planck checkpoint with comparable or better perplexity on held-out WikiText
vs. the TinyStories Planck 1.1, with no memorization of Reuters/AP content
(verify via n-gram overlap audit).

### Tradeoff
Grammar-first corpora produce a model that writes well but knows very little.
That's the point — the blob store fills in knowledge. But if the base is too
weak at reasoning, even good blobs won't rescue downstream QA. Plan to abort
if base MMLU-lite score drops more than ~20% vs. 1.1.

---

## 1.3.1 — Dynamic blob builder from RSS

### Goal
Continuously ingest live news and rebuild the blob store without model
retraining. Blobs age out or decay so stale coverage doesn't crowd out fresh.

### Architecture
- **Ingest**: pull Reuters + AP + BBC + Al Jazeera RSS every ~5 min
- **Chunk**: split each article into ~150-token spans, one blob per span,
  tagged with `{source, publish_ts, headline, url}`
- **Embed**: reuse the Planck blob-builder embedding head from 1.1
- **Store**: flat Faiss index with per-blob TTL (e.g. 72h hard cap) and
  exponential recency weight on retrieval score
- **Topic shard**: optional — politics / business / tech / world shards so
  retrieval at query time can pre-filter

### Deliverable
A long-running process that holds a ~50-200k blob index fresh, with ingest
latency < 10 min from article publish to retrievable blob.

### Tradeoff
Faiss + exponential decay is cheap but doesn't handle contradictions (same
event, two sources, conflicting details). A harder version does cross-source
deduplication + a "most-recent-wins" policy per entity. Leave that for 1.3.2
only if the eval exposes it.

---

## 1.3.2 — Freshness-scoped QA eval

### Goal
Prove the thing. Build a benchmark where the questions can only be answered
from the last 24 hours of news, run Planck 1.3 + blobs, compare against:
1. Planck 1.3 with no blobs (ablation)
2. A hosted frontier model with and without its web tool
3. A strong small baseline (e.g. Phi-3 / Llama-3.2 1B) with the same blob
   store

### Benchmark construction
- Daily, automated. Each day, sample ~100 articles published in the last
  24h, generate 3 factual QA pairs per article via a frontier model
  (GPT-5 as scorer/builder), filter for unambiguity, sanity-check by hand.
- Questions must be answerable from a single blob (1.3.2a) or require
  combining 2-3 (1.3.2b, harder).

### Metrics
Exact-match + LLM-judged factuality. Track latency too — the freshness
pitch dies if Planck + blob retrieval is slower than a frontier web call.

### Deliverable
A weekly leaderboard chart, checked into `docs/results/planck_13/`. If Planck
1.3 + blobs beats the best frontier model on same-day news QA at >10x lower
cost per query, write it up.

### Tradeoff
LLM-as-judge is noisy and leaks bias. For the first week, dual-score with
human spot checks (50 questions/week) so we know the judge isn't lying.

---

## Rollout

1. **1.3.0** base retrain on BabyLM-100M — 1-2 weeks assuming 1.2 accel lands
2. **1.3.1** RSS ingest + blob builder — 1 week of infra work, mostly plumbing
3. **1.3.2a** single-blob QA — first eval results, go / no-go gate
4. **1.3.2b** multi-blob QA — unlocks the "reasoner + knowledge" story
5. Writeup + LinkedIn post if the numbers land

Gate to Planck 1.4: only start 1.4 once 1.3.2a shows the blob retrieval path
is solid end-to-end. 1.4 reuses the same retrieval machinery for conversation
memory, so fixing retrieval bugs twice is wasteful.
