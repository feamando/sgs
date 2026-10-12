# Planck 3.0 round 6: a reader trained once, Brave with a relevance check, writers with a KV cache

Nikita Gorshkov · 2026-10-12 · Status: run 2026-10-12 (results in section 10: R FAILS); round 6b pre-registered (section 11) · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261009.md` · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round6` (about 8-10 h, idempotent, pushes results). Round 5 showed the weighing works when the reader is good: with Gemma reading, the 9-10 band was right 93-94% of the time and the 7-8 band 77-79%. The small product lost 20 points to Gemma, and the loss was in **reading**, not writing. Round 6 fixes the three things that held it back:
1. **Search:** Brave becomes the product search, and every search is checked for relevance before its results count. Round 5's ddgs results for the 100 new questions included xhamster.com and calculator.net.
2. **Reading:** Planck 1.3 (100M) and Hertz 1.2 (640M) are each fine-tuned **once** as a cross-encoder that reads the question and the evidence together. They train on about 3,200 questions that are not in any benchmark: 2025 news, 2025 events and general information, plus round 3's encyclopedic set. Labels come from known answers, never from a model.
3. **Writing:** the faithfulness checker no longer mistakes "[1, 2]" for numbers. Every answer starts from a forced lead (the value, "Possibly …", "Unclear:"), the small writers get 3 tries, and they decode with a new **exact KV cache**, which writes about 12x faster on CPU.

Nothing is trained on the world: facts still arrive through search, and nothing from 2026 is in any training set. The rules below were written **before** the run (2026-10-12).

**Result (2026-10-12, section 10).**
- **Reading, the primary rule (R): FAIL.** The new readers are significantly **worse** than round 5's reader: x-hertz 39.1% and x-planck 42.4%, against 51.4% for round 5's reader and 63.4% for Gemma.
- **Writing:** much better, but still short of W. The small writers' own answers are faithful 88% (Planck) and 86% (Hertz) of the time, up from 46-47% in round 5, against a bar of 95%. Gemma's writer went from 37% to 96%.
- **KV cache (K): PASS.** Answers are 100% identical with and without it, and it is about 9-10x faster on CPU.
- **Search: valid.** Every evaluation was served by Brave, 98% on topic.
- **Two confounds:**
  - A bug in the reader's training code: the last candidate slot was right far more often than it ever is when answering.
  - The month's Brave quota was already used up, so the reader trained on ddgs and Wikipedia pools (3 of 1,914 searches were Brave).
- **Round 6b** (section 11) retrains the readers with only the leak fixed, under the same rules.

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

## 10. Results of record (run 2026-10-12, 12:14-18:18 on the box)

**TL;DR.**
- **R fails, in the wrong direction.** Training a small SGS model as a reader made reading worse than round 5's frozen-feature reader. Neither new reader beats round 5's reader, and round 5's reader in turn is not distinguishable from the no-model floor on Brave.
- **The writer fixes worked for Gemma and halved the small writers' failures,** but neither small writer reaches W.
- **The KV cache is exact and ~9-10x faster on CPU.**
- **What ships stays round 5's Planck reader + the template writer.** `serve` now loads a new reader only if it passed R.

**Search (S): valid.**
- **Benchmark runs:** every run was served by Brave (cache hits from round 5's Brave run). 98.4% of searches were on topic and none came back empty.
- **Training-data collection was not on Brave.** The API reported **0** searches left in the month when collection started. Brave served 3 of 1,914 searches, ddgs 810 and Wikipedia 1,100 (fallback rate 50%, on topic 88%). The pre-run Brave check passed only because it accepts a cached Brave answer (a runner flaw, noted in Risks).

**The readers, trained once** (held-out questions of the training sets; 654 test pools, 431 with a right candidate):

| reader | test choice acc | lexical ranker | gate AUC | stable | recent | news | info |
|---|---|---|---|---|---|---|---|
| x-planck (101M) | 0.538 | 0.401 | 0.753 | 0.655 | 0.398 | 0.388 (ranker 0.418) | 0.455 |
| x-hertz (641M) | 0.568 (+3.0 pts) | 0.401 | 0.789 | 0.699 | 0.458 | 0.328 (ranker 0.418) | 0.491 |

- **The size rule picked Hertz by the narrowest margin** (+3.02 points against a bar of 3).
- **News is the real problem:** even on its own training distribution, neither reader beats the lexical ranker on news. Planck's news gate AUC is 0.50, i.e. chance.
- These test numbers are inflated by the last-slot leak (below).

**Reading only, template writer, 243 questions on Brave:**

| reader | right | fresh (60) | long tail (83) | news (60) | general info (40) | no answer | read ms (GPU) |
|---|---|---|---|---|---|---|---|
| heuristic (no model) | 0.498 | 0.22 | 0.55 | 0.60 | 0.65 | 1 | 0 |
| round 5 Planck reader | 0.514 | 0.40 | 0.66 | 0.40 | 0.55 | 1 | 39 |
| **x-planck** | 0.424 | 0.35 | 0.61 | 0.30 | 0.33 | 39 | 47 |
| **x-hertz** (chosen) | 0.391 | 0.28 | 0.65 | 0.33 | 0.10 | 54 | 63 |
| Gemma 4 E4B | 0.634 | 0.58 | 0.71 | 0.57 | 0.65 | 17 | 2,874 |

Paired, same 243 questions (exact McNemar):
- **x-hertz vs round 5's reader:** 18 vs 48 only-right, p 0.0003 (worse);
- **x-planck vs round 5's reader:** 27 vs 49, p 0.015 (worse);
- **x-hertz vs x-planck:** 22 vs 30, p 0.33;
- **round 5's reader vs the heuristic floor:** 37 vs 33, p 0.72 (no difference);
- **Gemma vs round 5's reader:** 61 vs 32, p 0.003.

**Where the new readers lose:**
- **They answer nothing too often.** r = choice × gate has a median of 0.24-0.30, against 0.66 for round 5's reader. The softmax temperature fitted on validation is 1.6 (Planck) and 4.1 (Hertz), so 39-54 questions fall under consolidation's MIN_R 0.05 and get no answer.
- **The abstentions are not justified:** on x-hertz's 54, Gemma was right on 33 and round 5's reader on 23.
- **They also choose worse where they do answer:** 0.503 against 0.540 for round 5's reader on the same 189 questions.

**The verdicts:**

| rule | bar | result | verdict |
|---|---|---|---|
| **R** (primary) | ≥ midpoint 0.574 and beat round 5's reader (p < 0.05) | 0.391; worse than round 5's reader, p 0.0003 | **FAIL** |
| C1 | 9-10 band ≥ 90%, 7-8 ≥ 70%, monotone | 9-10: 12 answers, 75% (too few to judge); 7-8: 67 answers, 68.7%; monotone | **FAIL** (7-8 band) |
| C3 (stretch) | not worse than Gemma + Gemma | 0.391 vs 0.634, p 4e-9 | **FAIL** |
| W | own answers faithful ≥ 95% within 3 tries | Planck 0.883 (first try 0.746, 1.42 tries); Hertz 0.858 (0.690, 1.49). Round 5: 0.472 / 0.460 | **FAIL** (much improved) |
| K | 100% identical | 100% for both writers on GPU (20) and CPU (10). CPU per answer: Planck 5.0 s → 0.55 s (9.1x), Hertz 41.5 s → 4.2 s (9.9x). GPU 1.0x / 2.1x | **PASS** |
| L | CPU reading + writing ≤ 5 s | x-hertz reading alone 10.6 s per question (20 questions, 4 threads) | **FAIL** (x-planck would be ~6x lighter; not measured) |
| E | worst-case profile drop ≤ 10 points, every change flagged | 0.399 vs 0.391; 154 of 154 changed answers flagged | **PASS** |
| S | Brave ≥ 60%, on topic ≥ 60%, empty ≤ 20% | 100% / 98.4% / 0% on every run | **PASS** |

**Reported, no bar:**
- **Gemma's writer** with the forced lead and the fixed checker: faithful 0.957, against 0.371 in round 5. It passes W, so the checker fix was most of round 5's "unfaithful".
- **Small-writer failures:** most are an inserted "2025" (the writer data is 2025 questions) or merged names ("Trondheim Trondheim Tramway").
- **Writer judge:** only the 10-question shakedown ran, because the product writer is the template (Gemma preferred the small writer 3, template 6, tie 1).
- **Training data:**
  - Gemma's news questions: 857 accepted, 234 rejected (kept with reasons).
  - Writer data: 1,412 examples that start with the lead.

**Diagnosis, in order of confidence:**
1. **A leak in round 6's code (bug).** At training time, `pool_cands` swapped a right candidate from beyond the top 16 into the **last** slot, and "position in the ranking" is one of the reader's features. So the reader learned "the 16th candidate is often right". When answering there are no labels and no swap, so that is never true. The held-out test used the same swap, which also inflates the table above.
2. **Search shift.** The readers trained on ddgs and Wikipedia snippets and were evaluated on Brave snippets (quota exhausted, above).
3. **A real limit.** Even in distribution, the 100M and 640M SGS models do not learn to read news from about 5,300 pools: choice accuracy is below the lexical ranker and the gate is at chance (Planck). The base models were pretrained on Wikipedia only, and 6x the size bought +3 points.

**Shipping:**
- `r6_product.json` names x-hertz + template, by the pre-registered choice rule.
- R failed, so `serve`/`chat` "auto" now loads a new reader only when its product passed R. Until then it loads **round 5's Planck reader + the template writer**, the best small product measured (0.514 on Brave).

## 11. Round 6b: the same readers, trained on the pools as served (pre-registered 2026-10-12, before the run)

**What changes (one thing).** `pool_cands` keeps the top 16 candidates exactly as the reader sees them when answering. A pool whose only right candidate is beyond the cut becomes a pool with no right candidate, as served. The train log now reports how many such pools each split has.

**Nothing else changes:**
- the same points and the same 80/10/10 split by question;
- the same hyperparameters, seeds and calibration;
- the same 243 questions on cached Brave results;
- the same consolidation (MIN_R 0.05);
- the template writer (no small writer passed W).

Round 6's runs are the comparators: `r6_planck_template`, `r6_gemma_template`, `r6_x-*_template`.

**Run:** `.\scripts\planck3.ps1 round6b -Quick -NoPush` (~15 min), then `.\scripts\planck3.ps1 round6b` (~1.5 h: the readers ~1 h, the runs ~10 min). It makes no new Brave searches.

**Rules (unchanged from section 5, applied to `r6b_*`):**
- **Reader:** x-planck unless x-hertz is ≥ 3 points better on test choice accuracy (now without the leak).
- **R:** the chosen reader + template reaches the midpoint between `r6_planck_template` and `r6_gemma_template` (0.574) and beats `r6_planck_template` with p < 0.05.
- **C1, L, E:** as in section 5, on `r6b_product.json`'s run, its `_contrarian` and its `_cpu` (20 questions).
- **Reported:** `r6b_x-* vs r6_x-*`, i.e. what the leak cost.

**Disclosures:**
- The bug was found **after** seeing round 6's results. Round 6's verdicts stand as the results of record, and round 6b is reported next to them, never in place of them.
- Expectation, stated before the run: R is **unlikely** to pass. Diagnosis 3 (no news reading even in distribution) is not touched by the fix. Round 6b separates the bug from the real result.

**If round 6b fails R as well**, fine-tuning a small SGS model on about 5,000 known-answer pools is not enough to make it read. The next levers, in order:
1. **Gemma as teacher** (allowed: teacher and reference only). Gemma reads the pools of the reader-training questions offline and its choices become soft labels: more reading examples from the same pools, with no new searches and no world data.
2. **Re-collect the reader pools on Brave** when the monthly quota resets, so training and serving see the same snippets.
3. Read pages, not snippets, for news (section 8, recommendation 4).

The small writers need a cleaner writer set (no "2025" in the targets) before another W attempt.

**New risk, with its mitigation:** the Brave check passes on a cached answer even at 0 quota. Next runner change: fail the check, or warn, when `brave_remaining` is 0 and collection needs new searches.
