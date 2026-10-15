# Planck 3.0 round 7: reading taught as a skill, then our pools (SGS only)

Nikita Gorshkov · 2026-10-14 · Status: run 2026-10-14/15 (section 10: R FAILS, N FAILS -> pre-registered outcome "rebuild"; Hertz stage A alone reaches the R bar, exploratory) · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261012.md` (rounds 6 and 6b) · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round7` (about 4-6 h, idempotent, pushes results). It makes no new searches.

**The problem.** Rounds 5, 6 and 6b measured three small-reader designs, and none reads better than the no-model floor on Brave. Gemma is 10-16 points ahead of all of them. Even on their own training distribution, the readers do worse than word matching on news. The likely reasons:
- the base models were pretrained on Wikipedia, not news;
- they were trained to predict the next token, not to read;
- they saw only about 5,300 reading examples.

**The change.** Before our own pools, Planck 1.3 and Hertz 1.2 are taught reading **as a skill** on about 150k examples from public reading-comprehension sets. Each set is turned into the reader's own pool format, through the same candidate extractor that answers questions:
- SQuAD;
- NewsQA (CNN news);
- TriviaQA;
- SearchQA (real search-result snippets);
- Natural Questions;
- SQuAD 2.0's unanswerable questions (teaches "the answer is not here").

Then round 6b's training runs on our pools, unchanged.

**SGS only.** No non-SGS control arm (owner's decision, 2026-10-14).

**What can come out.** The product reader is fixed before the run: x-planck, the phone-size model. A pre-registered **rebuild gate (N)** decides what round 8 is:
- **Planck still can't read news:** the Planck base is rebuilt from scratch.
- **Planck reads news but not our benchmark:** the problem is the gap between training searches and Brave, and there is no rebuild.

## 0. Why this round

- **Customer:** the phone product needs a reader that actually reads. Today's small product is right 51% of the time against Gemma's 63%, and its answers are mostly the lexical ranker's.
- **Commercial:** this round settles a week-long decision for a few hours of box time. Rebuilding Planck costs 3-7 days of the 4090 per attempt, so it should only happen if reading training on the current base provably fails on news.
- **Research:** it tests whether SGS models can learn to read at all once reading is what they train on. Either answer is a result for the SGS track.

## 1. What changes (one thing: reading training before our pools)

**Stage A, reading as a skill** (`src/planck3/rcdata.py`, `xreader rcdata`).
- **Sources:**
  - MRQA 2019 train (`mrqa-workshop/mrqa`): SQuAD, NewsQA, TriviaQA-web, SearchQA, NaturalQuestions;
  - SQuAD 2.0 train (`rajpurkar/squad_v2`), unanswerable questions only.
  - HotpotQA (multi-hop) is left out.
- **Same format as answering:**
  - Each passage goes through `span_candidates`, the extractor used on a fetched page.
  - SearchQA's documents go through `snippet_candidates`, as search results.
  - Labels come from `is_correct` against the dataset's answers.
  - The top 16 candidates are kept, as served (round 6b's fix).
- **Answer type:** taken from the question. A "text" question takes its gold answer's type; anything still untyped is dropped, because the benchmark has only entity / year / number / date questions.
- **Balance:** per set, at most 40% of pools have no right candidate on offer. The SQuAD 2.0 pools have none by design.
- **Training:** 1 epoch, the same learning rates and settings as round 6b. Validation and test are 6,000 pools each, split by question. Calibration is fitted on the held-out reading questions.

**Stage B, our pools.** Stage A's weights, then exactly round 6b's training: the same points, split, 3 epochs, settings and calibration.

**Unchanged:**
- the 243 questions on cached Brave results;
- consolidation (MIN_R 0.05);
- the template writer (no small writer passed W in round 6);
- the KV cache.

## 2. Training data

| Set | What it is | Pools (cap 40k each; expected) | Source of pools |
|---|---|---|---|
| SQuAD | Wikipedia paragraphs, crowd questions | ~26k | page extractor |
| NewsQA | **CNN news articles**, crowd questions | ~23k | page extractor |
| TriviaQA-web | trivia questions, web pages | ~15k (many answers untyped) | page extractor |
| SearchQA | Jeopardy clues, **search-result snippets** | ~40k | snippet extractor |
| NaturalQuestions | real Google queries, Wikipedia | ~33k | page extractor |
| SQuAD 2.0 unanswerable | the answer is not in the paragraph | ~12k (cap 20k) | page extractor |
| our pools (stage B) | round 6's 5,310 / 592 / 654 | unchanged | round 6 |

Per-set counts and drops are written to `data/planck3/reader/rc_points.health.json`, which is pushed.

**Benchmark disjointness.** A reading question is dropped if either:
- its subject (the longest capitalized span) is named in a benchmark question; or
- its answer is a benchmark answer (an entity of more than 4 letters).

This is deliberately conservative: about 6-7% are dropped, including generic subjects like "President".

## 3. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke | 82 offline tests | | ~2 min |
| reading data | download ~390 MB of parquet once; build the pools | `data/planck3/reader/rc_points.jsonl` | ~15-30 min |
| stage A | Planck, then Hertz: 1 epoch on the reading pools | `checkpoints/xreader_<b>_rc`, `xreader_<b>_rc/train_log.json` | ~25 min + ~3 h |
| stage B | from stage A, round 6b's training on our pools | `checkpoints/xreader_<b>_r7` | ~3 min + ~25 min |
| reading runs | stage A only and stage A+B, for both models, template writer, 243 questions | `r7_x-<b>-rc_template`, `r7_x-<b>_template` | ~10 min |
| product checks | x-planck r7: worst-case personal profile; CPU only (20 questions) | `r7_x-planck_template_contrarian`, `_cpu` | ~5 min |
| decide + report + push | the rules below → `r7_product.json`; REPORT.md gains a "Round 7" section | git | 1 min |

## 4. Step by step

1. `git pull`
2. Shakedown, about 15-20 min: 300 pools per set, small training samples, 12 questions per run.
   ```powershell
   .\scripts\planck3.ps1 round7 -Quick -NoPush
   ```
3. The real run:
   ```powershell
   .\scripts\planck3.ps1 round7
   ```
   Re-run the same command if it stops; finished stages are skipped. To rebuild the reading pools, delete `rc_points.jsonl`.

## 5. Pre-registered rules (2026-10-14, before the run)

**Product.** **x-planck r7 + template writer**, fixed. There is no selection between Planck and Hertz:
- Round 6b's size rule picked the worse model for the benchmark.
- The target is the phone.

Hertz is the scaling arm; it is reported and tested on the same rules as a secondary result.

- **R (primary):** `r7_x-planck_template` must pass both:
  - accuracy at or above the midpoint between `r6_planck_template` (round 5's reader) and `r6_gemma_template` (the bar is 0.574);
  - significantly better than `r6_planck_template` (exact McNemar, p < 0.05).
- **N, the rebuild gate:** does Planck read news?
  - On stage A's held-out **NewsQA** pools that have a right candidate, Planck's choice beats its own lexical ranker (the top candidate) by **≥ 10 points**, with exact McNemar p < 0.05.
  - The same measure on SearchQA (snippets) and for Hertz is reported.
- **H, reported:** Hertz r7 beats Planck r7 on the benchmark by ≥ 5 points (p < 0.05). That would mean broader pretraining matters, which also supports a rebuild with a broader corpus.
- **C1, L, E, S:** as in round 6 (section 5 of `SETUP_planck_20261012.md`), on the product.
  - C1: the 9-10 band right ≥ 90% of the time, the 7-8 band ≥ 70%, both judged at ≥ 20 answers; precision falls band by band.
  - L: reading + writing ≤ 5 s per question on CPU.
  - E: worst-case profile costs ≤ 10 points, and every change is flagged.
  - S: Brave search validity.
- **Outcome**, written to `r7_product.json` by `planck3.py r7 decide`:

| Result | Round 8 |
|---|---|
| R passes | Ship x-planck r7 (`serve` picks it up automatically). |
| R fails and N fails | **Rebuild Planck from scratch:** a reading-first training goal and a pre-2026 corpus that includes news. |
| R fails and N passes | Planck reads, but not our benchmark: the problem is the gap between training searches and Brave. Brave-collected pools with Gemma soft labels; no rebuild. |

- **Reported, no bar:**
  - stage A alone vs stage A+B on the benchmark (does reading training transfer by itself?);
  - r7 vs round 6b per model (what stage A added);
  - per-group held-out accuracy (all reading sets and our four groups);
  - the benchmark by regime (fresh, news, long tail, general information).

## 6. Disclosures (before any round 7 number exists)

1. **The reading sets carry old world facts** (2019 and earlier).
   - The reader learns to find answers in passages, but its weights may absorb some facts.
   - The benchmark's fresh and news questions are about 2026, so they cannot be helped by this.
   - Long-tail and general-information questions could be, which is why the benchmark is reported by regime.
   - The disjointness filter removes questions about benchmark subjects.
2. **This round was designed after rounds 6 and 6b.**
   - The choice of sets (NewsQA for news, SearchQA for snippets) follows round 6b's diagnosis.
   - No set was chosen by looking at benchmark accuracy.
3. **The fixed-product rule replaces round 6's size rule** because round 6b showed the size rule picked the worse reader. Fixing the product before the run is the conservative choice.
4. **The N margin (10 points)** is set from rounds 6b and 6: on our encyclopedic pools a reader that reads beats the ranker by 20-26 points.
5. **There is no non-SGS control** (owner's decision). So an N failure says "this SGS base cannot learn to read news this way". It cannot separate the SGS architecture from the Wikipedia pretraining. A rebuild changes the pretraining, and keeps SGS.
6. **The Brave quota and the pending spot-check** are unchanged from round 6:
   - no new searches are needed;
   - the 10-answer spot-check and the PROPOSED registry rows still await review.

## 7. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Reading sets → pools | reuse the candidate extractors, `is_correct`, answer-type inference; new `rcdata.py` (download, parse, type, filter, balance) | S |
| Two-stage reader | reuse `xreader.train`; new `--init`, `--max-eval`, lazy encoding, McNemar vs the ranker per group | XS |
| Rules + report | reuse `round6.rule_r` and the run table; new `round7.py` (N, H, C1, L, E, outcome) | XS |
| Runner | reuse `Invoke-R6`; new `Do-Round7` | XS |

## 8. Recommendations

1. Run `round7 -Quick -NoPush`, then `round7`.
2. Whatever the outcome, round 8 follows the pre-registered table in section 5, not a new judgment.
3. **If the outcome is "rebuild",** write the rebuild plan before any training:
   - the corpus: a pre-2026 mix including news, with a date cutoff;
   - the training goal: reading-first (answer from a passage, abstain when the answer is not in it);
   - the budget: 3-7 days on the 4090 per attempt.
4. Review the PROPOSED registry rows and the 10 spot-check answers when you can.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| The reading sets' format differs from our pools (long clean passages vs messy snippets) | the same extractor and pool format; SearchQA gives real snippets; stage B ends on our own pools |
| Reading training teaches facts that help long-tail questions | regime-level reporting; disjointness filter; fresh/news are 2026 |
| Stage A takes too long on Hertz | 1 epoch, lazy encoding, ~3 h measured from round 6b's throughput; Planck (the product) trains first |
| Memory: 150k pools | pools are trimmed to the top 16 at load and encoded lazily per batch |
| Hugging Face is down or slow | the download is cached in `data/planck3/rc_raw`; re-run resumes |
| Stage B undoes stage A (forgetting) | reported: stage A alone vs A+B on the benchmark |

## 10. Round 7 results of record (run 2026-10-14 22:16 to 2026-10-15 01:57, 37a641d)

**TL;DR.**
- **R FAILS:** the product (x-planck r7 + template) gets 54.3%, against the 57.4% bar; vs round 5's reader 51.4%, p 0.47.
- **N FAILS, so the pre-registered outcome is "rebuild":** after 137k reading pools, Planck does not beat its own lexical ranker on held-out NewsQA (46.4% vs 47.7%, p 0.37).
- **Planck does not read at all, not only news.**
  - It is below the ranker on SQuAD (56.0% vs 60.1%, p 0.0002), which is Wikipedia, its own pretraining text.
  - Its training loss is flat from step 2,000 to the end of the epoch (1.85 → 1.80).
- **Hertz reads everything.**
  - It is +10 to +30 points over the ranker on every reading set (NewsQA +10.2, p 3e-11).
  - Its loss falls steadily through the whole epoch (1.79 → 1.43).
- **The strongest result is exploratory:** Hertz with reading training only (stage A, no stage B) gets **60.1%** on the benchmark.
  - That is above the R bar, and p 0.015 vs round 5's reader.
  - It is not distinguishable from Gemma (63.4%, p 0.38), and it is the first small reader to beat the no-model floor clearly (p 0.0002).
  - It was one of four arms and not the pre-registered product, so it needs a confirmation run (with a ×4 correction, p ≈ 0.06).
- **Stage B (our pools) adds nothing for Planck and costs Hertz 4 points:** 60.1% → 56.0% (p 0.15). Hertz overfits our 5,310 pools (loss 0.18).
- **What ships is unchanged:** round 5's Planck reader + the template writer (`serve` loads a reader only if its product passed R).

**Stage A, the reading sets** (171,538 pools built; 137,581 train, 6,000 val and 6,000 test; held-out test pools that have a right candidate; choice vs the lexical ranker, exact McNemar):

| set | pools | ranker | x-planck rc | x-hertz rc |
|---|---|---|---|---|
| SQuAD (Wikipedia) | 813 | 0.601 | 0.560 (p 0.0002, **below**) | 0.654 (p 0.0003) |
| **NewsQA (CNN, gate N)** | 859 | 0.477 | **0.464 (p 0.37) FAIL** | 0.580 (p 3e-11) PASS |
| TriviaQA-web | 417 | 0.422 | 0.508 (p 3e-5) | 0.717 (p 4e-29) |
| SearchQA (snippets) | 833 | 0.462 | 0.522 (p 4e-5) | 0.677 (p 1e-34) |
| NaturalQuestions | 884 | 0.389 | 0.399 (p 0.24) | 0.628 (p 1e-41) |
| all | 3,806 | 0.474 | 0.487 | 0.643 |
| train loss (start → end of the epoch) | | | 1.85 → 1.80 (flat after step 2,000) | 1.79 → 1.43 (still falling) |

**Stage B, our pools** (held-out, the same 654 test pools as round 6b):

| reader | test choice | ranker | stable | recent | news | info | train loss, epochs 1-3 |
|---|---|---|---|---|---|---|---|
| x-planck v2 (6b) | 0.446 | 0.434 | 0.512 | 0.324 | 0.407 | 0.385 | 1.99 / 1.87 / 1.85 |
| x-planck r7 | 0.456 | 0.434 | 0.525 | 0.338 | 0.407 | 0.385 | 1.88 / 1.88 / 1.85 |
| x-hertz v2 (6b) | 0.609 | 0.434 | 0.737 | 0.423 | 0.458 | 0.500 | 1.51 / 0.69 / 0.21 |
| x-hertz r7 | 0.667 | 0.434 | 0.710 | 0.577 | 0.593 | 0.692 | 1.26 / 0.54 / 0.18 |

**The benchmark: reading only, template writer, the same 243 questions on Brave (cached; 251 searches, all Brave, 98.4% on topic):**

| reader | right | fresh (60) | long tail (83) | news (60) | general info (40) | no answer |
|---|---|---|---|---|---|---|
| heuristic (no model) | 0.498 | 0.22 | 0.55 | 0.60 | 0.65 | 1 |
| round 5 Planck reader (ships) | 0.514 | 0.40 | 0.66 | 0.40 | 0.55 | 1 |
| x-planck v2 (6b) | 0.535 | 0.25 | 0.59 | 0.63 | 0.70 | 5 |
| x-planck rc (stage A only) | 0.531 | 0.25 | 0.60 | 0.63 | 0.65 | 5 |
| **x-planck r7 (the product)** | **0.543** | 0.28 | 0.59 | 0.63 | 0.70 | 6 |
| x-hertz v2 (6b) | 0.473 | 0.32 | 0.70 | 0.33 | 0.45 | 31 |
| **x-hertz rc (stage A only)** | **0.601** | 0.47 | 0.66 | 0.60 | 0.68 | 6 |
| x-hertz r7 | 0.560 | 0.38 | 0.67 | 0.60 | 0.53 | 17 |
| Gemma 4 E4B | 0.634 | 0.58 | 0.71 | 0.57 | 0.65 | 17 |

Paired, the same 243 questions (exact McNemar, only-run-right vs only-other-right):
- **x-planck r7 (the product):**
  - vs round 5's reader: 38 vs 31, p 0.47;
  - vs the heuristic: 16 vs 5, p 0.027;
  - vs Gemma: 28 vs 50, p 0.017;
  - vs x-planck v2: 8 vs 6, p 0.79.
- **x-hertz rc (exploratory):**
  - vs round 5's reader: p 0.015;
  - vs the heuristic: 35 vs 10, p 0.0002;
  - vs Gemma: 28 vs 36, p 0.38;
  - vs x-planck r7: 25 vs 11, p 0.029;
  - vs x-hertz r7: 25 vs 15, p 0.15.
- **What stage A added to Hertz:** x-hertz r7 vs x-hertz v2, 35 vs 14, p 0.004.
- **H (reported):** x-hertz r7 vs x-planck r7, 26 vs 22, p 0.67 (+1.6 points): no flag.

**The verdicts (product = x-planck r7 + template):**

| rule | result | verdict |
|---|---|---|
| **R** | 0.543 vs midpoint 0.574; vs round 5's reader p 0.47 | **FAIL** |
| **N** (rebuild gate) | NewsQA 0.464 vs ranker 0.477 (-1.3 points, p 0.37) | **FAIL** |
| N, SearchQA (reported) | +6.0 points, p 4e-5 (under the 10-point margin) | fail |
| N, Hertz (reported) | NewsQA +10.2 points, p 3e-11 | pass |
| H (reported) | Hertz r7 +1.6 points, p 0.67 | no flag |
| C1 | 9-10: 26 answers, 80.8% (bar 90%); 7-8: 80, 67.5% (bar 70%); not monotone (sceptical 52.5% > low 39.6%) | **FAIL** |
| L | reading 1.05 s per question on CPU (20 questions) | **PASS** |
| E | 0.543 under the worst-case profile (no drop); 197 of 197 changed answers flagged | **PASS** |
| S | Brave served 100%, on topic 98.4%, empty 0% | **PASS** |

**Outcome (pre-registered, `r7_product.json`):** "rebuild: Planck does not read news even after reading training (round 8 rebuilds the base)".

**What the run says beyond the rules.**
1. **Planck's failure is general, not about news.**
   - It fails on Wikipedia (SQuAD) as well as news, and its loss stops moving after 2,000 of ~26,600 steps.
   - So the pre-registered rebuild rationale ("a corpus that includes news") names the wrong lever. The rebuild has to target reading itself.
2. **Size and pretraining data are confounded.** Hertz differs from Planck in both:
   - size: 640M vs 100M;
   - corpus: FineWeb-Edu + 30% Wikipedia, 9.7B tokens, vs Wikipedia only, ~4B tokens.
   
   This run cannot say which one matters.
3. **A recipe check is due before a 3-7 day rebuild.** Hertz learns under the same code. But Planck's flat loss from step 2,000 also fits an optimization problem, as well as a capacity one:
   - the learning rate (5e-5) was never swept for Planck;
   - the Planck reader was never shown to overfit a small set.
   
   If Planck can learn under another recipe, N was measuring the recipe, not the base. That would be a validity issue like round 6's leak, not a re-judgment.
4. **Reading training is the lever for Hertz, and our own pools hurt it.**
   - Stage A alone: 60.1%.
   - Stage A then stage B: 56.0%.
   - Stage B overfits 5,310 pools in 3 epochs.
5. **The best small reader is now Hertz rc,** within 3 points of Gemma. But:
   - it is 640M parameters;
   - round 6b's Hertz read at 11.4 s per question on CPU (L fails). Hertz rc was not timed on CPU in this run.
   - Its 9-10 band is right 72% of the time, so C1 fails as well.
6. **Housekeeping found in the logs:** the cosine schedule's step count is estimated from candidates, not from batches. So the last ~9% (Planck) and ~15% (Hertz) of stage A ran at learning rate 0. Both models were affected the same way. Planck's plateau starts long before that point.

**Recommendations (owner's call; not built):**
1. **Recipe check first (~30-45 min on the box), pre-registered before it runs:**
   - Planck overfits 500 reading pools (20 epochs)?
   - A learning-rate sweep (2e-5, 1e-4, 3e-4) on 20k reading pools, scored on the NewsQA held-out set.
   - **If Planck still cannot beat the ranker on NewsQA by 10 points,** the rebuild stands.
   - **If it can,** stage A is re-run for Planck with the better recipe, and N is judged on that run (disclosed as a re-run).
2. **If the rebuild stands, separate size from data:**
   - a 100M SGS base on Hertz's corpus mix, with reading in the training goal;
   - 3-7 days per attempt;
   - the plan is written before any training.
3. **Confirm x-hertz rc as a product candidate, pre-registered:**
   - fresh questions on Brave once the quota resets (the 243 questions are now used for selection);
   - CPU latency (L) with fewer candidates (top 8) and int8;
   - its C1 bands.
4. **Drop stage B in its current form.** Train on our pools only with early stopping on a benchmark-like slice, or with Gemma soft labels.
5. **Fix the schedule's step count** before the next training run.

