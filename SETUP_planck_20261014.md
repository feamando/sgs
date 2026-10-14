# Planck 3.0 round 7: reading taught as a skill, then our pools (SGS only)

Nikita Gorshkov · 2026-10-14 · Status: ready to run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261012.md` (rounds 6 and 6b) · Plan and results of record: `SETUP_092026_planck3.md`

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
