# Planck 3.0 round 9: Hertz's reader confirmed on fresh questions, as it would run on a phone

Nikita Gorshkov · 2026-10-19 · Status: run 2026-10-19 (section 9: **R9 PASSES → Hertz's int8 reader ships**; Q, L, E pass; C1 fails) · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261015.md` (round 8: Hertz int8 passes the phone proxy; Planck 1.5 reads like Planck 1.3) · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round9` (about 2 h). It is resumable and pushes results.

**The problem.** Hertz's reading reader, in int8 on CPU, got 60.5% in round 8, against 63.4% for Gemma and 51.4% for the shipped reader. But it was chosen out of four arms on those same 243 questions, so the number is selected.

**The fix: a fresh benchmark,** built once from questions nobody has looked at, and committed to git before any reader runs on it. About 220 questions:
- **~140 from Wikidata:** 2026 winners, elections, seasons and new heads of government, all items unused by any earlier benchmark or training set;
- **~80 news questions:** written by Gemma from Wikipedia's Current Events portal for 9-18 October 2026, after the old news benchmark ends.

**What runs on it:**
- **The product, fixed before the run:** Hertz's reader, int8 weights, CPU with 4 threads, top 16 candidates, template writer. This is exactly what passed round 8's phone proxy.
- **Against:** round 5's reader (shipped today), Gemma, the no-model heuristic, and Hertz on the GPU (the int8 check).

**Search:**
- It needs about 300 live Brave searches. Each question is searched once, and every reader then reads the same cached results.
- The Brave check is fixed: a cached answer no longer counts as "Brave is up". The run reads the month's remaining quota first and stops if it is short.

**If it passes, it ships:** `serve` picks it up and reads with int8 on CPU. If not, round 10 trains Hertz's reader on Brave pools with Gemma soft labels.

## 0. Why this round

- **Customer:** a reader that is right about 10 points more often than today's (60% vs 51%), within reach of Gemma, at 888 MB and about 1.3 s per question on a desktop CPU. This round decides whether that holds on questions it was not chosen on.
- **Commercial:** it is the cheapest possible ship decision. There is no training: one benchmark build, about 300 searches, about 2 h of the box.
- **Research:** a pre-registered confirmation on held-out questions, the step every selected result needs before it is believed.

## 1. What changes

**The fresh benchmark** (`src/planck3/freshbench.py`, `planck3.py fresh9`, → `scripts/assets/planck3_tasks_fresh9.json`):

| part | what | how many | gold |
|---|---|---|---|
| Wikidata 2026 | event winners (60), elections (20), season champions (30), new heads of government (30); the old fresh benchmark's templates | 140 (dry run on the Mac: 140 kept from pools of 840 / 65 / 436 / 239) | the Wikidata label + English aliases, no model |
| Portal news | Wikipedia Current Events, 2026-10-09 to yesterday; up to 15 items a day across sections (126 items for 9-18 October); Gemma writes one question per item, through round 6's checks (answer in the item, not in the question, year in the question, typed) | up to 80 | the item's own words |

**Disjointness:**
- Every Wikidata item already in a benchmark, round 3's training set or round 6's reader training set is dropped by QID, and so is any item named in a benchmark question. The dry run dropped 11 items as used before and 5 as named.
- The portal window starts the day after the old news benchmark's last event.
- The file joins the benchmark list, so no later training set can include it.

**Built once:**
- If the file exists, it is the benchmark of record and is never rebuilt.
- The full run commits and pushes it before the first reader runs on it, with its build manifest: seed 9, the pool sizes and every drop count.

**The Brave check** (`web.py`):
- `brave_alive` used to pass on a cached answer, which is how round 6 ran on ddgs and Wikipedia while believing it was on Brave.
- It now makes one live search and reads the monthly quota from the headers. It fails at 0 quota.
- `search-check --backend brave --need N` prints the searches left and fails if fewer than N remain.
- The runner asks for about 1.3 × the number of questions + 20 before any reader runs.

**Search once, read many:**
- The product runs first and searches live; the other arms read the same cached Brave results.
- So every arm is judged on identical evidence, at one Brave search per question (plus a few retries and fallbacks).

**Unchanged:** the readers (no training), consolidation, the template writer, the source registry.

## 2. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke | 85 offline tests | | ~2 min |
| 1. fresh benchmark | Wikidata (cached queries) + Gemma on the portal items; commit + push | `scripts/assets/planck3_tasks_fresh9.json` | ~20-40 min |
| 2. Brave check | live search, quota ≥ needed | log line | seconds |
| 3. the product | Hertz rc, int8, CPU 4 threads, top 16 (searches live) | `r9_x-hertz-rc_cpu-int8-top16` | ~15-25 min |
| 4. the arms | heuristic, round 5's reader, Hertz on GPU, Gemma, Planck 1.5 (reported) | `r9_*_template` | ~25 min |
| 5. rule E | the product under the worst-case personal profile (CPU) | `r9_x-hertz-rc_cpu-int8-top16_contrarian` | ~10 min |
| 6. decide + report + push | the rules below → `r9_product.json`; REPORT.md gains a "Round 9" section | git | 1 min |

## 3. Step by step

1. `git pull`
2. Shakedown, about 10-15 min. It builds a small separate benchmark (3 questions per template + 6 news questions from 2 days, written to `planck3_tasks_fresh9_quick.json`) and runs every arm on it:
   ```powershell
   .\scripts\planck3.ps1 round9 -Quick -NoPush
   ```
   - It spends about 20 Brave searches.
   - Check that the log line `Brave Search API: UP; searches left this month: N` shows N of at least ~310.
3. The real run:
   ```powershell
   .\scripts\planck3.ps1 round9
   ```
   - The fresh benchmark is pushed as soon as it is built, before any reader runs.
   - If anything stops, re-run the same command: the benchmark is kept as built, and finished runs are skipped.
   - If the quota is short, the run stops before the first reader. Re-run when the quota resets.

## 4. Pre-registered rules (2026-10-19, before the benchmark exists)

**Product.** **x-hertz rc, int8, CPU (4 threads), top 16 + template writer**, fixed: round 8's configuration, unchanged.

- **R9 (primary):** on the fresh questions, the product must pass both:
  - at or above the midpoint between round 5's reader and Gemma;
  - significantly better than round 5's reader (exact McNemar, p < 0.05).
  
  This is round 6's rule R on new questions; the midpoint is computed on the fresh questions.
- **Q:** int8 on CPU loses ≤ 2 points against the same reader in fp32 on the GPU, on the fresh questions.
- **L:** reading + writing ≤ 5 s per question, measured on CPU over every fresh question (not a 20-question sample).
- **E:** under the worst-case personal profile, ≤ 10 points drop, and every changed answer is flagged.
- **C1, reported** (it has failed for every small reader so far):
  - as before: 9-10 ≥ 90%, 7-8 ≥ 70%, judged at ≥ 20 answers, monotone;
  - **recalibrated:** the band thresholds are fitted on the product's round 8 answers to the 243 old questions (the lowest confidence whose answers at or above it are right ≥ 90% / ≥ 70%, at ≥ 10 answers), then judged on the fresh questions.
- **Reported, no bar:**
  - the product vs the heuristic and vs Gemma;
  - Planck 1.5's reader vs round 5's reader;
  - accuracy by regime (Wikidata fresh vs portal news).
- **Outcome,** written to `r9_product.json` by `planck3.py r9 decide`:

| Result | Next |
|---|---|
| R9 passes | **Ship Hertz's reader** (int8 on CPU, top 16). `serve` / `chat` pick it up from `r9_product.json` automatically. Then: an on-device export (ONNX Runtime Mobile or ExecuTorch) and a measurement on a handset. |
| R9 fails | No ship. Round 10 trains Hertz's reader on Brave-collected pools with Gemma soft labels (the lever not yet tried). |

Q, L and E are product checks. If R9 passes but one of them fails, the result is reported, and shipping waits for that check to be fixed.

## 5. Disclosures (before any round 9 number exists)

1. **The news questions are written by Gemma, and Gemma is one of the arms.**
   - Gemma only writes the question from a summary. The answer is the summary's words, checked by rule, and Gemma does not see the search results when writing.
   - Gemma's own reading may still be helped by question phrasing in its own style. R9's midpoint uses Gemma, so this could raise the bar the product has to reach. That is conservative for the product.
2. **The Wikidata half reuses the old fresh templates** ("Who won the …?"). The new questions share that style but not the items. That is the right control for "the same task, new questions", but it does not test new kinds of question.
3. **The portal window is short** (9 October to the run date). News questions cover about ten days of events, mostly conflicts, disasters and law.
4. **Wikidata golds are labels and aliases as of the build date.** Some 2026 items may be edited later. The file records `gold_as_of`.
5. **The product on CPU searches live; the other arms read its cached results.** All arms see the same evidence. A search that fails for the product and falls back to Wikipedia is also what the others get.
6. **The phone proxy is still the box's desktop CPU** (AMD Zen 4, 4 threads). A phone is slower (my assumption: 2-4×). L passing here is necessary for the phone, not sufficient.
7. **The Brave check change affects every later run:** each run now spends 1 search to confirm the API is live with quota left.
8. **Found in the shakedown (2026-10-19):** the check failed with "searches left this month: 0" while the Brave dashboard showed $1.91 of $5 credit used.
   - The likely cause: a credit-based plan has no monthly search quota, so the rate-limit header's monthly part reads 0 even while searches succeed.
   - **Confirmed by the headers** (2026-10-19 shakedown): `X-RateLimit-Limit: 50, 0`, `X-RateLimit-Remaining: 49, 0`, `X-RateLimit-Policy: 50;w=1, 0;w=2678400`. The monthly limit is 0: there is no monthly quota on this plan.
   - The fix: a monthly remainder is believed only when the plan reports a monthly limit, and never on an HTTP 200 with results. The check now prints the raw rate-limit headers, so the cause is confirmed on the next run, not assumed.
   - **This very likely also explains round 6.** Round 6's training collection paused Brave at once ("0 searches left this month (reserve 400)") and gathered its pools on ddgs and Wikipedia. That was recorded as "the Brave quota was 0". It was more likely the same header misread, while the benchmark runs (no reserve) were served by Brave.

## 6. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Fresh Wikidata questions | reuse `benchgen`'s 2026 queries, `_golds`, `_task`, and round 6's disjointness helpers; new QID filter across benchmarks and training sets | XS |
| Fresh news questions | reuse round 6's portal parser, Gemma prompt and checks; new window (after 2026-10-08) and several items a day | XS |
| Brave check fix | reuse `_search_brave` and its quota header; new live `brave_quota`, `search-check --need` | XS |
| Rules + report | reuse `rule_r`'s logic, `product_checks`; new `round9.py` (R9, Q, recalibrated C1, outcome) | XS |
| Runner | reuse `Invoke-R6` (now takes a task file), `Invoke-OnCpu`; new `Do-Round9` | XS |

## 7. Recommendations

1. Run `round9 -Quick -NoPush`, check the Brave quota line, then run `round9`.
2. If R9 passes, ship the product and build the on-device export next. Rule M was a desktop proxy; a handset number is the real one.
3. If R9 fails, round 10 trains Hertz's reader on Brave pools with Gemma soft labels, and is re-judged on this fresh benchmark only if it is still unused for selection. Otherwise a new one is built.
4. Still open: the 10-answer spot-check, and the PROPOSED source-registry rows.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| The Brave quota is short | checked live before the first reader; the run stops instead of falling back to ddgs |
| Gemma writes fewer than 80 usable news questions | round 6's checks keep about 78%; 126 items gives ~95; the Wikidata half (140) carries R9 on its own |
| The fresh benchmark leaks into training later | the file is in the benchmark list every training-set builder filters against |
| A Wikidata answer is wrong or out of date | the Wikidata label and aliases are the gold, as for the old fresh benchmark; the file records the date |
| Re-running rebuilds a different benchmark | built once: an existing file is never rebuilt, and the full run commits it before any reader runs |

## 9. Round 9 results of record (run 2026-10-19, benchmark 58b6a74, results f32deb0)

**TL;DR.**
- **R9 PASSES, so Hertz's reader ships** (int8 on CPU, top 16, template writer). On 220 fresh questions:
  - it is right **49.1%** of the time, against **31.4%** for round 5's reader (53 vs 14 only-right, **p 2e-6**) and 35.5% for the no-model heuristic (p 0.0002);
  - the bar was the midpoint to Gemma, 46.1%.
- **Q, L and E pass:**
  - int8 on CPU is within 1 point of the same reader on the GPU (49.1% vs 50.0%, 1 vs 3 only-right);
  - it reads in 3.35 s per question on CPU;
  - the worst-case personal profile costs nothing, with all 152 changes flagged.
- **Gemma is still 12 points ahead** (60.9%, 23 vs 49 only-right, p 0.003). The gap is all on the Wikidata questions; on the news questions they are level (51% vs 54%, p 0.82).
- **The bar is relative, and it fell.**
  - The midpoint is lower than round 6's (46.1% vs 57.4%) because round 5's reader does much worse on fresh questions (31.4%, vs 51.4% on the old 243).
  - Hertz's own accuracy also fell, from 60.5% to 49.1%: these questions are harder.
  - It passes because the gap to the shipped reader held (+18 points, vs +9 on the old set), not because it got better.
- **Confidence bands still fail C1.** The recalibration, fitted on the old questions, did not carry over to the new ones.
- **What ships changes:** `serve` and `chat` now load Hertz's reader from `r9_product.json` (int8 on a machine without a GPU).

**The fresh benchmark** (built and committed at 17:23, before any reader ran; unchanged since):
- **140 Wikidata 2026 items:** winners 60, elections 20, seasons 30, heads of government 30. Drops: 10 used before, 6 named in a benchmark question, 19 without a proper-name answer, 4 not a place.
- **80 portal news questions,** 9-18 October 2026: Gemma wrote 100 usable from 126 items, and 80 were drawn by seed.
- **Search:** 218 of 220 questions served by Brave and 2 by Wikipedia, 97.7% on topic, no empty searches. Every arm read the same results.

**The benchmark (220 fresh questions; reading only, template writer):**

| reader | right | Wikidata (140) | news (80) | 9-10: n / right | 7-8: n / right | no answer |
|---|---|---|---|---|---|---|
| heuristic (no model) | 0.355 | 0.29 | 0.48 | 5 / 0.60 | 35 / 0.46 | 0 |
| round 5 Planck reader (shipped until now) | 0.314 | 0.28 | 0.38 | 6 / 0.83 | 37 / 0.59 | 0 |
| Planck 1.5 rc (reported) | 0.386 | 0.34 | 0.48 | 1 / 1.00 | 38 / 0.55 | 5 |
| Hertz rc, GPU fp32 | 0.500 | 0.49 | 0.51 | 12 / 0.92 | 51 / 0.73 | 2 |
| **Hertz rc, CPU int8, top 16 (the product)** | **0.491** | 0.48 | 0.51 | 13 / 0.92 | 49 / 0.67 | 1 |
| Gemma 4 E4B | 0.609 | 0.65 | 0.54 | 17 / 0.94 | 107 / 0.70 | 19 |

**By template (Wikidata), right out of n:**

| template | n | product | Gemma | round 5's reader |
|---|---|---|---|---|
| event winners | 60 | 32 | 42 | 17 |
| elections | 20 | 14 | 16 | 5 |
| season champions | 30 | 12 | 14 | 12 |
| new heads of government | 30 | 9 | 19 | 5 |

Paired, the same 220 questions (exact McNemar, only-product-right vs only-other-right):
- vs round 5's reader: 53 vs 14, p 2e-6. By part:
  - Wikidata: 40 vs 12, p 0.0001;
  - news: 13 vs 2, p 0.007.
- vs the heuristic: 46 vs 16, p 0.0002. By part:
  - Wikidata: p 0.0001;
  - news: 9 vs 6, p 0.61.
- vs Gemma: 23 vs 49, p 0.003. By part:
  - Wikidata: 14 vs 38, p 0.001;
  - news: 9 vs 11, p 0.82.
- vs itself on the GPU (Q): 1 vs 3, p 0.63.
- Planck 1.5 rc vs round 5's reader (reported): 38 vs 22, p 0.052.

**The verdicts (product = Hertz rc, int8, CPU, top 16 + template):**

| rule | result | verdict |
|---|---|---|
| **R9** | 0.491 vs midpoint 0.461 (round 5's reader 0.314, Gemma 0.609); 53 vs 14, p 2e-6 | **PASS** |
| Q | 0.491 vs 0.500 on the GPU (-0.9 points; bar 2) | **PASS** |
| L | 3,353 ms reading per question, over all 220 questions (median 3.7 s, p90 4.7 s, max 5.5 s) | **PASS** |
| E | no drop; 152 of 152 changed answers flagged | **PASS** |
| C1 (as before) | 9-10: 13 answers (too few to judge), 92% right; 7-8: 49, 67.3% (bar 70%); not monotone | **FAIL** |
| C1 (recalibrated) | on the old answers no threshold reaches 90% (the high band is never shown); "good" from 6: 102 answers, 58.8% right | **FAIL** |

**Outcome (pre-registered, `r9_product.json`):** "ship: Hertz's reader (int8, CPU) passed R9 on fresh questions".

**What the run says beyond the rules.**
1. **A real, confirmed improvement over what shipped, not Gemma-level reading.**
   - The product beats round 5's reader by 18 points on questions nobody chose it on.
   - It matches Gemma on news, but trails it by 17 points on the Wikidata questions, mostly new heads of government (9 vs 19 of 30) and event winners (32 vs 42 of 60).
2. **Reading got slower than in round 8: 3.35 s vs 1.33 s per question,** on the same machine and settings.
   - The worst-case profile run read from the cache and took 3.7 s, so live searching is not the cause.
   - It is not diagnosed. The likely cause is fuller candidate pools on fresh pages: reading time grows with the number of candidates, capped at 16.
   - **On a phone 2-4× slower this is about 7-13 s per question,** so the top-8 configuration (round 8: half the time, -1.2 points) becomes the phone default unless the handset measurement says otherwise.
3. **Confidence is informative but mislabelled.** On the fresh questions the product's answers at 8 or above are right 22 of 24 times (92%), and at 7 or above 45 of 62 (73%). The bands are cut at 9 and 7. Thresholds fitted on the old questions do not transfer, because the score's meaning shifts with the question mix. C1 needs a calibrator that does not depend on one question set.
4. **The heuristic and round 5's reader collapse on fresh Wikidata questions** (28-29%), while the readers trained to read hold up (Hertz 48-49%). Reading, not ranking, is what carries to new questions.
5. **Planck 1.5's reader is ahead of Planck 1.3's lineage on fresh questions** (38.6% vs 31.4% for round 5's reader, p 0.052), not significant. Reported only.

**Recommendations (owner's call; not built):**
1. **Measure on a handset now.** Export the int8 reader (ONNX Runtime Mobile or ExecuTorch), and time top 16 and top 8 on a real phone. The desktop proxy says 3.35 s; a phone decides.
2. **Calibration (C1) as its own small round:** fit the bands on a held-out slice of fresh-style questions (or isotonic on reader_p + features), and judge on a separate fresh set. Today's 9-10 label overstates certainty on 7-8 answers.
3. **Round 10 (the Gemma gap):** train Hertz's reader on Brave-collected pools with Gemma soft labels. Brave now works (the quota misread is fixed), so round 6's planned training data can finally be collected on Brave. Judge it on a new fresh benchmark; this one is now used.
4. **Retire this benchmark for selection.** It decided this round; a later round needs its own.

