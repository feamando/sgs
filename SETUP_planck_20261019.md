# Planck 3.0 round 9: Hertz's reader confirmed on fresh questions, as it would run on a phone

Nikita Gorshkov · 2026-10-19 · Status: ready to run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261015.md` (round 8: Hertz int8 passes the phone proxy; Planck 1.5 reads like Planck 1.3) · Plan and results of record: `SETUP_092026_planck3.md`

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
