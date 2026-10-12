# Planck 3.0 round 6: a reader trained once, Brave with a relevance check, writers with a KV cache

Nikita Gorshkov · 2026-10-12 · Status: ready to run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261009.md` · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round6` (about 8-10 h, idempotent, pushes results). Round 5 showed the weighing works when the reader is good: with Gemma reading, the 9-10 band was right 93-94% of the time and the 7-8 band 77-79%. The small product lost 20 points to Gemma, and the loss was in **reading**, not writing. Round 6 fixes the three things that held it back:
1. **Search:** Brave becomes the product search, and every search is checked for relevance before its results count. Round 5's ddgs results for the 100 new questions included xhamster.com and calculator.net.
2. **Reading:** Planck 1.3 (100M) and Hertz 1.2 (640M) are each fine-tuned **once** as a cross-encoder that reads the question and the evidence together. They train on about 3,200 questions that are not in any benchmark: 2025 news, 2025 events and general information, plus round 3's encyclopedic set. Labels come from known answers, never from a model.
3. **Writing:** the faithfulness checker no longer mistakes "[1, 2]" for numbers. Every answer starts from a forced lead (the value, "Possibly …", "Unclear:"), the small writers get 3 tries, and they decode with a new **exact KV cache**, which writes about 12x faster on CPU.

Nothing is trained on the world: facts still arrive through search, and nothing from 2026 is in any training set. The rules below were written **before** the run (2026-10-12).

## 0. Why this round

- **Customer:** an answer the user can trust at a glance on their phone. The band has to mean what it says (9-10 right at least 9 times in 10), and the answer has to arrive in seconds without a GPU.
- **Commercial:** a 100M reader + 100M writer is about 40x smaller than Gemma 4 E4B, runs locally (no per-query model cost, no data leaving the phone), and never needs retraining for new facts. If the small reader closes most of the gap, Gemma leaves the runtime for good.

**What "retrain Planck" means here.** Planck and Hertz are retrained *as a reader* and *as a writer*, once each, on reading and writing examples. Planck's base model is **not** retrained on web text: the box's page cache holds pages fetched for the benchmark questions, so that would put benchmark answers into the weights. It would also teach the model facts, which breaks the "skills, not the world" rule.

## 1. What changes

**Search** (`src/planck3/web.py`).
- **On topic:** a search is on topic when at least 2 of its top 5 results (1 if fewer come back) mention enough of the question's key words in title + snippet. Key words are the content words without question-form, month and year words, and a 6+ letter word also matches its stem ("election" ~ "elected").
- **Chain:** off-topic results are treated like an empty search: primary → secondary → Wikipedia. If nothing is on topic, the first answer is served and counted as off topic.
- **Validity:** a run is valid only if ≥ 60% of its searches were on topic and ≤ 20% came back empty.
- **Product search:** `auto` is now Brave (key in `.env`) → ddgs → SearXNG → Wikipedia.
- **Brave quota:** read from the API's `X-RateLimit-Remaining` header. Training-data collection stops calling Brave once **400** searches are left in the month and continues on ddgs (relevance-checked), so the product always keeps a reserve.

**Reader** (`src/planck3/xreader.py`, `readtasks.py`). The cross-encoder reads the shared prefix "QUESTION … ANSWER TYPE …", then one continuation per candidate: "CANDIDATE … FROM … EVIDENCE … DOES THE CANDIDATE ANSWER THE QUESTION?" It reads the top 16 candidates of a pool. The hidden state at the last token, plus round 4's lexical features, gives two logits:
- **choice:** which candidate answers, a softmax over the pool, trained where a right one exists;
- **answer:** does this candidate answer, trained on every candidate.

The reader returns r = choice × g, where g is a Platt-scaled answer logit. Both are calibrated on held-out *questions*, and r feeds `consolidate.py` unchanged, so the weighing is round 5's.

**Year shift.** Half the training pools get every year shifted by the same random amount (−4 to +4). "2025" vs "2026" then carries no signal on its own, so the reader has to read. This is the fix for round 4's gate, which learned that only past-year questions were answerable.

**Writers** (`src/planck3/writer.py`).
- **Checker fix:** `faithful()` treats citation groups ("[1, 2]", "[1-3]", "[1 and 2]") as citations. Round 5's 1,200 Gemma examples are re-judged with the fix, not regenerated.
- **Forced lead:** every model answer starts from a lead (the value; "Possibly <value>" when low; "Unclear:" when sceptical), so it can't leave the answer out (84 Hertz failures in round 5). Gemma is told to begin with it too.
- **Three tries:** the small writers try greedy, then 2 seeded samples (T 0.7, top-k 20), and the first faithful answer is shown. Otherwise the template is shown, as before.
- **More data:** Gemma writes answers for the reader-training questions too (up to 1,500 more, news-style included). Only examples that start with the lead train the writers.

**KV cache** (`src/sgs_lm.py`, `SGSLanguageModel.forward_cached`).
- **Exact, not approximate:** a token's Gaussians at every pass depend only on tokens at or before it. Appending a token never changes earlier ones, so their per-pass μ, α and features are cached. Max difference from the full forward on the offline test: 0.0.
- **Measured on the Mac CPU (4 threads, Planck size):**
  - 48 tokens after a 400-token prompt take **0.31 s** with the cache, against about 3.7 s without.
  - A reader pool of 16 candidates shares one cached question prefix and takes **0.18 s**.

## 2. Training data (never a benchmark, nothing from 2026)

| Set | Questions | Built from | Labels |
|---|---|---|---|
| stable (round 3) | 1,455 | Wikidata, 2012-2024 events + popular encyclopedic facts; pools from round 4's ddgs collection (reused, no search) | gold (Wikidata) |
| recent | 557 | Wikidata, the fresh benchmark's templates on **2025**: winners 300, seasons 120, elections 77, heads of government 60 | gold (Wikidata) |
| info | 500 | Wikidata, popular items: airport IATA codes 120, cities served 100, stadium capacities 100, building floors 80, station opening dates 100 | gold (Wikidata) |
| news | ~700 (of 1,091 items) | Wikipedia Current Events portal **2025**, up to 3 sections a day; **Gemma** writes one question per item on the box, and checks drop any that fail (answer must be in the item, not in the question, year in the question, short answer) | gold = the item's own words |

**Disjoint from every benchmark:**
- by Wikidata QID;
- items whose distinctive name appears in any benchmark question are dropped (Dubai, Haneda and Schiphol airports, for example);
- Gemma's news answers that name a benchmark subject are dropped.

The question files are committed: `scripts/assets/planck3_read_train_tasks.json`, `scripts/assets/planck3_portal_items_2025.json`, and Gemma's `data/planck3/reader/news_tasks.json`, including every rejected question with its reason.

## 3. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke + doctor + Brave check | 79 offline tests; stops if Brave does not answer | `doctor.json` | ~5 min |
| **news questions** | Gemma writes questions from the 1,091 portal items (resumable) | `data/planck3/reader/news_tasks.json` | ~1 h |
| **reader data** | search + label pools for ~1,750 questions (Brave to the 400 reserve, then ddgs) | `data/planck3/reader/points.jsonl` | ~1.5-2 h |
| **readers, once** | x-planck (100M) and x-hertz (640M) cross-encoders | `checkpoints/xreader_*`, `xreader_*/train_log.json` | ~15 min + ~45 min |
| **reading only** | template writer with: heuristic, round 5's Planck reader, x-planck, x-hertz, Gemma | `r6_*_template` | ~1.5 h |
| **writer data** | round 5 data re-judged; Gemma writes up to 1,500 answers with the lead | `data/planck3/writer/train6.jsonl` | ~1-1.5 h |
| **writers, once more** | Planck and Hertz writers on the fixed data; KV cache check | `checkpoints/writer_*6`, `kvcheck_*` | ~15 min |
| **product candidates** | chosen reader + Planck writer, + Hertz writer; Gemma + Gemma | `r6_<reader>_{planck,hertz}`, `r6_gemma_gemma` | ~1.5 h |
| **product + checks** | pre-registered choice; worst-case personal profile; blind writer judge; CPU-only run (20 questions) | `r6_product.json`, `*_contrarian`, `judge.json`, `*_cpu` | ~40 min |
| report + push | REPORT.md gains the two round 6 tables | git | 1 min |

Every evaluation runs the same 243 questions (143 fresh + long-tail, 60 news, 40 general information) on Brave. Most searches are cache hits from round 5's Brave run, so they cost no quota. Training collection needs about 1,750 new searches. Brave answers them until the API reports 400 left in the month, and ddgs takes the rest. The mix is reported in `points.health.json`.

## 4. Step by step

1. `git pull`
2. Shakedown, about 45 min (`_quick` outputs, 40-60 examples per stage, 12 questions per run):
   ```powershell
   .\scripts\planck3.ps1 round6 -Quick -NoPush
   ```
   Check that REPORT.md has the two "Round 6" tables with `r6_*_quick` rows.
3. The real run:
   ```powershell
   .\scripts\planck3.ps1 round6
   ```
   Re-run the same command if it stops; finished stages are skipped.
4. To try the product afterwards: `.\scripts\planck3.ps1 serve`. It loads whatever `r6_product.json` chose, on Brave.

## 5. Pre-registered rules (2026-10-12, before the run)

**The product** is chosen by fixed rules, on Brave, over all 243 questions; paired comparisons use exact McNemar on the same questions.
- **Reader:** x-planck, unless x-hertz beats it by **≥ 3 points** of choice accuracy on the readers' own held-out test questions (a 6x larger model has to earn its size).
- **Writer:** the smallest small writer that passes rule W, else the template.

- **R, the reader closes the gap (primary):** the chosen reader with the template writer must pass both:
  - accuracy at or above the midpoint between round 5's Planck reader and the Gemma reader (both with the template writer, measured this round on the same 243 questions);
  - significantly better than round 5's Planck reader (p < 0.05).
- **C1, the confidence means something:**
  - the product's 9-10 band is right ≥ **90%** of the time;
  - the 7-8 band ≥ **70%**;
  - precision falls band by band.

  A band with fewer than 20 answers is "too few to judge"; bands under 5 are left out of the ordering check.
- **C3, small models instead of Gemma (stretch):** the product is not significantly worse than Gemma reader + Gemma writer (FAIL if Gemma wins with p < 0.05).
- **W, the writer is faithful:** the writer's own answers (any of its ≤ 3 tries) pass the check ≥ **95%** of the time. First-try faithfulness is reported. The shown answer is always faithful, because a failed one is replaced by the template.
- **K, the KV cache is exact:** **100%** of the greedy answers are token-identical with and without the cache, for both writers on GPU and on CPU. Speed-up reported.
- **L, phone proxy:** on CPU only (4 threads), the product's reading + writing take ≤ **5 s** per question, search and page fetches excluded.
- **E, no echo chamber:** under the worst-case personal profile (every source scored 10 − system), accuracy drops ≤ **10 points**, and every changed answer is flagged.
- **S, search is valid:** every round 6 run must meet all three:
  - Brave served ≥ 60% of its searches;
  - ≥ 60% of searches were on topic;
  - ≤ 20% came back empty.

  A run below the bar is redone once, then kept as DEGRADED.
- **Reported, no bar:**
  - x-hertz vs x-planck on the benchmark;
  - per-group test accuracy of the readers (stable / recent / news / info);
  - the blind writer judge (Gemma prefers the small writer, the template, or neither);
  - registry coverage, question-type accuracy, LLM tokens.

## 6. Disclosures (before any round 6 number exists)

1. **The relevance rule was designed after seeing round 5's off-topic results.** Its word thresholds were set by hand on those examples, never on benchmark accuracy.
2. **The reader-training sets were designed after round 5 showed the news gap.** The "recent" templates are the fresh benchmark's templates moved to 2025, the "info" templates were chosen to look like general-information questions, and the news questions come from 2025 portal items. None is from 2026, all are disjoint by QID, and any item named in a benchmark question is dropped.
3. **The searches for the 243 questions are mostly round 5's cached Brave results.** They differ only where the relevance check now falls back to Wikipedia.
4. **Gemma's writer prompt changed:** it is told to begin with the lead and may use 80 tokens (48 before). So `r6_gemma_gemma` is not round 5's writer exactly.
5. **The weighing constants are unchanged from round 5** and are not refit. If C1 fails with a good reader, they change only with a stated reason and a new pre-registration.
6. **Your spot-check of the 10 flagged benchmark answers (section 2 of round 5's doc) is still pending.** Fixes are applied with `planck3.py r5 --rescore`, without re-running.
7. **Label noise:** Wikidata answers can lag the web (capacities, floors), and a news item's wording is the only gold form. This is accepted for training; it would be fatal for evaluation, which is why the benchmarks keep 2-source gold.

## 7. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Relevance check + search chain + Brave quota | reuse `Web`; new `on_topic`, chain, header parsing | S |
| Reader data | reuse `benchgen` SPARQL + `g2.collect` + round 4's pools; new portal parser + Gemma question checks | M |
| Cross-encoder reader | reuse `SGSLanguageModel` + round 4 features + `consolidate.py`; new `xreader.py` | M |
| KV cache | new `forward_cached` + `expand_cache` (exact; tested against the full forward) | S |
| Writer fixes | reuse `writer.py` + `train_writer`; checker fix, lead, tries, rescore, kvcheck | S |
| Runner + report | reuse the r5 evaluator; new `round6` command, `r6_product.json`, report tables, judge | S |

## 8. Recommendations

1. Run `round6 -Quick -NoPush`, then `round6`.
2. **If R and C1 pass,** the small product is real: a 100M reader and a 100M (or template) writer with honest bands. Next: a real phone build (ONNX or Core ML export of the cached forward) and a latency measurement on a device.
3. **If R passes but C1 fails,** the reader reads well but its p is miscalibrated. Recalibrate on the held-out questions (still training data) before touching the weighing.
4. **If R fails,** the reading gap is not about data breadth. Next would be a larger context (more evidence per candidate) or reading pages instead of snippets, not a bigger model.
5. **Review the PROPOSED registry rows and the 10 spot-check answers** when you can. They still set the scores most answers rest on.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| The reader learns "the year decides" again (round 4's gate) | year-shift augmentation; no 2026 data; per-group test accuracy and the benchmark's news column reported separately |
| Gemma writes bad news questions | 7 mechanical checks per question; every rejected question is kept with its reason; labels come from the item's own words |
| Training data leaks benchmark facts | 2025 only; disjoint by QID; benchmark subjects dropped by name; question files committed for audit |
| Brave quota runs out mid-collection | the header-based reserve hands collection to ddgs at 400 left; benchmark searches are cached |
| The relevance check throws away good results | it only re-orders the chain (the first answer is still served if nothing is on topic); on-topic rate reported per run |
| The KV cache changes outputs | exactness test offline (diff 0.0) and the on-box `kvcheck` bar K (100% identical) |
| A forced lead reads awkwardly ("1943, …") | the blind writer judge reports whether the small writer reads better than the template |
