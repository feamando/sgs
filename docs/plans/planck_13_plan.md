# Planck 1.3 — Wikipedia base + fresh-knowledge blobs from live news

*Status: plan. Written 2026-04-27, updated 2026-05-01 to (a) adopt
Wikipedia as the primary base corpus and (b) ship a Wikipedia blob
index alongside the live-news one. Planck 1.2.x acceleration work
shelved 2026-05-01; 1.3 trains on plain AdamW, same regime as
Hertz 1.2.*

Planck 1.1 proved that blobs work: a small LM (~100M params) can be conditioned
on externally-built blobs and meaningfully use their content at inference time
without retraining. Planck 1.3 pushes the same concept in two directions:

1. **Stronger base.** Retrain Planck on Wikipedia instead of TinyStories.
   English Wikipedia is broad, encyclopedia-grammar-clean, and has
   repeated, consistently-phrased definitions of most concepts. It is a
   better base than TinyStories for a model whose pitch is "reasoner +
   knowledge via blobs".
2. **Freshness.** Continuously rebuild a second blob index from live
   news feeds, so a cheap base model can answer questions about events
   that happened in the last hour.

The pitch: a 100M-param Planck trained on Wikipedia + a Wikipedia-
derived static blob index + a live RSS blob index should outperform a
frontier model (Gemini 3 / GPT-5) on 24-hour-fresh factual QA, because
frontier models can't retrain that fast and their public web tools are
slow, rate-limited, or both. It should also outperform Planck 1.1 on
general knowledge QA because the base itself now carries encyclopedic
priors.

Everything below is a plan, not a commitment.

---

## 1.3.0 — Wikipedia base retrain

### Goal
Replace the TinyStories-style training distribution with Wikipedia so the
base model has broad grammatical competence **and** encyclopedic priors.
The same prose appears in both roles — as training data for the base, and
later (1.3.1b) as a static blob index — so the model's training
distribution and its retrieval distribution are the same thing. That
alignment is a feature, not a risk: the base learns "how concepts are
usually defined", and the blob store hands it "which specific concept is
involved in this prompt".

### Corpus: English Wikipedia
- **Source**: Wikimedia snapshot dumps (`enwiki-YYYYMMDD-pages-articles-multistream.xml.bz2`).
  Pick a specific snapshot and check it into `data/wikipedia/snapshot_id.txt`
  so the base-training run is reproducible.
- **Cleaning**: strip markup with `wikiextractor` (templates, references,
  tables that confuse the tokenizer); keep lead sections + body prose,
  drop disambiguation pages, lists, and redirects.
- **Size**: cleaned English Wikipedia is ~4B tokens at our tokenizer,
  comparable to Planck 1.1's token budget. No subsampling needed.
- **Why not mix in C4 / BabyLM**: keeping the base corpus to one source
  means the blob-retrieval distribution shift is zero. If we mix corpora,
  the base model has a mixed prior over phrasings and we lose the
  "encyclopedia voice" that makes Wikipedia blobs retrievable.

### Why not keep TinyStories or switch to BabyLM
- TinyStories is a narrative-fiction distribution. Planck 1.1 demonstrated
  that blobs work *in that regime*, but it can't answer "what is X" for
  encyclopedic X — it has no prior on the definitional voice.
- BabyLM is grammar-first but factually threadbare. If we're going to pay
  the retrain cost, we want the base to also contribute knowledge, not
  just grammar.
- Wikipedia gives us both: clean grammar + definitional priors + a corpus
  that naturally doubles as a blob index.

### Deliverable
A Planck-1.3 checkpoint with:
- Perplexity on held-out Wikipedia ≤ a stable baseline (we're training on
  Wikipedia, so this is fitness, not generalisation).
- Zero memorisation of held-out news (sanity: no 5-gram overlap with the
  Reuters/AP eval set).
- MMLU-lite (or a small internal knowledge benchmark, ~200 questions)
  score *without blobs* strictly above Planck 1.1's. This is the check
  that the base itself got smarter; blobs only add to it.

### Tradeoff
Wikipedia is ~4B tokens vs. TinyStories' ~1B, so base training is ~4×
longer wall-clock at matched throughput. We absorb this because:
(a) Planck 1.2.x accel is shelved — we are not waiting for a magic
    speedup, so "long training" is now the honest baseline,
(b) a stronger base is the single biggest quality lever for a 100M model,
(c) the same corpus powers the 1.3.1b blob index, so preprocessing
    amortises.

If base MMLU-lite score drops below Planck 1.1's despite the larger
corpus, abort the 1.3 track — something is wrong with the tokenizer or
training harness, not with the corpus choice.

---

## 1.3.1a — Static Wikipedia blob index

### Goal
Turn the same Wikipedia snapshot used for base training into a blob index
the Planck 1.3 runtime retrieves from at inference. This is the
"encyclopedic knowledge on demand" half of the pitch and it also powers
Satz 0.1.3 (Wikipedia blob bundle in the text-demo).

### Architecture
- **Ingest**: reuse the 1.3.0 cleaned Wikipedia corpus. No extra download.
- **Chunk**: one blob per lead section (~150-token summary for most
  articles), tagged with `{title, categories, pageid, revision_ts}`.
  Optionally add a second-tier index of body-paragraph blobs later, but
  the lead-section index covers "what is X" for most X at the demo scale.
- **Embed**: reuse the Planck blob-builder embedding head from 1.1.
- **Store**: flat Faiss index. No TTL — Wikipedia blobs are
  explicitly static. Keep the snapshot id on each blob so a future
  refresh can coexist with an older one during A/B eval.

### Deliverable
A Faiss index of ~6M article-lead blobs derived from the same snapshot
as the 1.3.0 training corpus. Loadable by `src/blob_store.py` via a new
`--bundle wikipedia` option.

### Tradeoff
Lead-section-only coverage misses long-tail article bodies (e.g. a
specific battle described in a history article's §3). That's a 1.3.1a.1
follow-up (paragraph-level blobs) and only worth doing if the eval
exposes it as the bottleneck.

---

## 1.3.1b — Dynamic blob builder from RSS

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

## 1.3.2 — QA eval (static + fresh)

### Goal
Prove the thing, on both axes: encyclopedic knowledge (static Wikipedia
blobs, 1.3.2a) and 24h-fresh news (RSS blobs, 1.3.2b). Run Planck 1.3
against itself with and without each blob index, and against external
baselines.

### Comparisons
1. Planck 1.3 with no blobs (ablation — did the base get smarter?)
2. Planck 1.3 + Wikipedia blobs only (how much does static retrieval help?)
3. Planck 1.3 + RSS blobs only (freshness pitch, 24h-news eval)
4. Planck 1.3 + both (the full stack)
5. A hosted frontier model with and without its web tool
6. A strong small baseline (e.g. Phi-3 / Llama-3.2 1B) with the same blob
   store

### Benchmark construction
- **1.3.2a** (static): ~500 knowledge questions from an off-the-shelf
  benchmark (MMLU subset, TriviaQA) plus ~100 hand-curated "what is X"
  questions where X is a Wikipedia article title. The latter is the
  direct test of the Wikipedia blob index.
- **1.3.2b** (fresh): daily, automated. Each day, sample ~100 articles
  published in the last 24h, generate 3 factual QA pairs per article
  via a frontier model (GPT-5 as scorer/builder), filter for
  unambiguity, sanity-check by hand.
- **1.3.2c** (multi-blob): questions that require combining 2-3 blobs,
  possibly one Wikipedia + one RSS (e.g. "X was appointed prime
  minister yesterday; what is X's political party's historical stance
  on Y?" — RSS tells you the appointment, Wikipedia tells you the
  party stance).

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

1. **1.3.0** base retrain on Wikipedia — ~2-3 weeks on plain AdamW (no
   accel recipe; ~4B tokens at 4090 baseline throughput)
2. **1.3.1a** static Wikipedia blob index — 3-5 days of infra, reuses the
   1.3.0 preprocessing pipeline
3. **1.3.1b** RSS ingest + dynamic blob builder — ~1 week of plumbing
4. **1.3.2a** single-blob QA on Wikipedia blobs — smoke test of the
   retrieval + conditioning path
5. **1.3.2b** single-blob QA on RSS blobs — first eval results, go / no-go
   gate for the freshness pitch
6. **1.3.2c** multi-blob QA (Wikipedia + RSS) — unlocks the full
   "reasoner + static knowledge + fresh knowledge" story
7. Writeup + LinkedIn post if the numbers land

Note on ordering: 1.3.1a ships before 1.3.1b so we can validate the
retrieval path end-to-end against a *static* corpus whose correctness we
can audit by hand. Adding the live-news index on top of a validated
retrieval stack is strictly less risky than debugging both at once.

Gate to Planck 1.4: only start 1.4 once 1.3.2a shows the blob retrieval
path is solid end-to-end on Wikipedia. 1.4 reuses the same retrieval
machinery for conversation memory, so fixing retrieval bugs twice is
wasteful.
