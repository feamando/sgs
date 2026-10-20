# Planck 3.0 round 10: closing the Gemma gap (Hertz's reader on Brave pools, Gemma soft labels)

Nikita Gorshkov · 2026-10-20 · Status: built; owner bought credit (2026-10-20, $11.81): run now with `-Limit 2000`, judge from 29 October · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261019.md` (round 9: Hertz's int8 reader ships, 49.1% vs Gemma 60.9%) · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round10`, run once in November and once in December. It uses free Brave credit only (owner's call, 2026-10-20): at most 950 live searches a month, $4.75 of the $5 credit. It resumes where it stopped.

**The gap.** On round 9's fresh questions, Gemma beat the shipped reader by 12 points. All of the gap is on Wikidata-style questions; on news they were level.

**The lever.** Round 6 meant to train the reader on Brave-collected pools. A header misread paused Brave, so the pools came from ddgs and Wikipedia. Round 10 does it properly:
1. **November:** collect round 6's training questions on Brave (557 Wikidata 2025 items like the gap, then 857 portal news questions from 2025).
2. **December:**
   - finish the collection;
   - Gemma reads every Brave pool once, giving soft labels;
   - retrain Hertz's reader from the shipped one (our old pools + the Brave pools, Gemma's labels at weight 0.5, early stopping);
   - build a **new** fresh benchmark, committed before any reader runs;
   - judge.

**Ship rule:** the new reader must beat the shipped one by at least 5 points on the new fresh questions (p < 0.05), and pass the phone checks.

## 0. Why this round

- **Customer:** the shipped reader is right on 49% of fresh questions, Gemma on 61%. Closing even half the gap is about 6 points more right answers, at the same phone cost: the same model, int8, 888 MB.
- **Commercial:** it costs nothing in cash (free credit only), a few box-hours, and about six weeks of calendar time.
- **Research:** it tests the one lever not yet tried cleanly: training data from the product's own search engine, plus a teacher's soft labels.

## 1. What changes

**Brave collection** (`xreader collect`, `points_brave.jsonl`):
- **Questions:** round 6's training questions, unchanged and never in a benchmark.
  - first, the 557 Wikidata 2025 "recent" items: winners, elections, seasons, heads of government, the gap's question kinds;
  - then the 857 portal news questions from 2025.
- **Brave only** (`--secondary none`): an empty Brave answer stays empty and is never filled from another engine.
- **A monthly credit ledger** (`results/planck3/r10_brave_ledger.json`): one budget per calendar month (950 live searches), shared by every step of the run.
  - When it is spent, collection stops cleanly. It never falls back and never marks an unsearched question as done.
  - The next month's run carries on (`points_brave.progress.json` records how far it got).
  - **Before 1 November the runner refuses to spend,** unless `-Limit N` is given.

**Gemma soft labels** (`xreader teach`):
- Gemma reads every Brave pool once, with exactly the reader prompt it uses on the benchmark (top 20 candidates, pick one or none, with a probability).
- Its pick is stored on the pool, and the step is resumable.
- The share of pools where Gemma picks a right candidate is logged.

**Training** (`xreader train --teacher-w 0.5 --early-stop 1`):
- **Starts from** round 7's Hertz reading reader: the shipped product.
- **Trains on** our round 6 pools plus the Brave pools.
- **Loss:**
  - the known-answer loss, unchanged;
  - plus 0.5 × a soft-label term: Gemma's chosen candidate gets its stated probability and the rest share the remainder; "none" pushes every answer logit down.
- **Early stopping** after one epoch without a better validation score. In round 7, Hertz overfit our pools (loss 0.18).
- Calibration on held-out questions, as before.

**A new fresh benchmark** (`fresh9 --prefix f10 --after 2026-10-18`, → `planck3_tasks_fresh10.json`):
- Built exactly like round 9's: about 140 Wikidata 2026 items unused by rounds 1-9 (round 9's benchmark now counts as used), and up to 80 Gemma-written questions from portal items after 18 October.
- Built once, committed and pushed before any reader runs on it.

## 2. What runs, by month

| When | What | Brave searches | Box time |
|---|---|---|---|
| now (optional) | `round10 -Quick -NoPush` shakedown: 24 collection questions, 40 pools taught, a tiny training run, a tiny benchmark | ~50 (October credit) | ~20 min |
| November (from the 1st) | collect: ~1 search per question (plus a retry on empty), until 950 | ≤ 950 | ~30 min |
| December (from the 1st) | finish collecting (~500); Gemma labels every Brave pool (~2,500 pools, ~1 h); train (~30-45 min); build the new benchmark (~30 min); judge the five arms (~1.5 h) | ~800 | ~4 h |

If December's credit runs short before the judging, the run stops before the first reader and judges in January.

**Update 2026-10-20: the owner bought credit ($11.81 ≈ 2,360 searches),** so the monthly wait is gone. Round 10 needs about 1,750: ~1,450 for the collection, ~300 for the judging, and a few for checks.
- **Now:** `round10 -Limit 2000` collects, teaches and trains in one run (~3 h), then stops.
  - The new benchmark's news half needs the same 10-day portal window as round 9 (19-28 October). Built today, it would hold about 2 days of news.
- **On or after 29 October:** `round10 -Limit 2000` again builds the benchmark (pushed before any reader runs) and judges (~1.5 h).
  - Both runs share October's ledger, so the cap of 2,000 searches ($10) covers both.

## 3. Step by step

1. `git pull`
2. Optional, now:
   ```powershell
   .\scripts\planck3.ps1 round10 -Quick -NoPush
   ```
   It spends about 50 searches of October's remaining credit.
3. On or after 1 November:
   ```powershell
   .\scripts\planck3.ps1 round10
   ```
   It stops at 950 searches and logs `N of 1414 collection questions through`.
4. On or after 1 December, the same command. It finishes the collection, then teaches, trains, builds the benchmark (pushed at once) and judges.

## 4. Pre-registered rules (2026-10-20, before any data exists)

**Product.** **x-hertz r10, int8 on CPU (4 threads), top 16 + template writer**, fixed before the run.

- **G (primary):** on the new fresh questions, the product beats **the shipped reader** (round 9's product, the same configuration, on the same questions):
  - by **≥ 5 points**;
  - exact McNemar p < 0.05.
- **GAP, reported:** the share of the shipped reader's gap to Gemma that the product closes on the new questions: (product − shipped) / (Gemma − shipped).
- **Q:** int8 on CPU loses ≤ 2 points vs the same reader in fp32 on the GPU.
- **L:** ≤ 5 s per question on CPU, measured over every question.
- **E:** worst-case personal profile, ≤ 10 points drop, every change flagged.
- **C1:** reported, as before.
- **Outcome,** written to `r10_product.json` by `planck3.py r10 decide`:

| Result | Next |
|---|---|
| G, Q, L, E pass | **Ship x-hertz r10.** `serve` / `chat` pick it up automatically. |
| G passes, a product check fails | Keep round 9's reader; fix the check first. |
| G fails | Keep round 9's reader. Round 11's question becomes how big the Gemma gap is for a 640M SGS reader: a larger base, or Gemma's evidence view (top 20, longer contexts). |

## 5. Disclosures (before any round 10 number exists)

1. **Gemma both labels the training pools and is the reference arm.** The labels come from 2025 questions, not the benchmark, and Gemma does not see the benchmark when labelling. GAP measures how close the student gets to its teacher, so read it with that in mind.
2. **The training questions overlap round 6's.** They are the same questions on a different engine (Brave instead of ddgs/Wikipedia), with the same task ids, so they keep their train / validation / test split.
3. **The 950-a-month cap is mine,** not Brave's: the $5 monthly credit at $5 per 1,000 searches, less 50 for the checks. Retries on empty answers count against it.
4. **The 5-point margin is set from round 9.** It is the smallest gain that is both detectable on ~220 questions and worth shipping a new checkpoint for.
5. **Six weeks of calendar time:** the world moves. The new benchmark's news comes from after 18 October, and the training questions are from 2025, so there is no overlap.

## 6. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Brave collection | reuse round 6's `collect`, questions and pool format; new monthly ledger, Brave-only mode, credit stop + progress file | S |
| Gemma soft labels | reuse the Gemma reader prompt (`LLMReader`); new `xreader teach` (resumable) | XS |
| Training | reuse `xreader.train`; new soft-label loss term, early stopping, extra pool files | XS |
| New fresh benchmark | reuse round 9's builder; new id prefix and news window | XS |
| Rules + runner | reuse round 9's paired tests and product checks; new `round10.py` (G, GAP), `Do-Round10` | XS |

## 7. Recommendations

1. Optional now: the shakedown (`round10 -Quick -NoPush`), to catch a box-side problem before November.
2. Run `round10` once after 1 November and once after 1 December.
3. In parallel, no Brave needed: the handset timing of round 9's reader (top 16 vs top 8). It decides the phone default whatever round 10 finds.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| The credit runs out mid-run | the ledger caps each month at 950; collection stops cleanly and resumes; the judging checks its need first |
| A Brave failure looks like a credit stop | a streak of 15 empty answers stops the run (round 3's breaker), not a silent fallback |
| Gemma's labels are wrong on many pools | the known-answer loss stays at full weight; Gemma's agreement with the gold is logged |
| Training on Brave pools overfits | early stopping on held-out questions; it starts from the shipped reader, not from scratch |
| The new benchmark leaks into training | it is in the benchmark list every training-set builder filters against; built after the collection |
