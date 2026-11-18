# Planck 3.0 round 12: distillation at scale (can Hertz's reader learn Gemma's reading?)

Nikita Gorshkov · 2026-11-13 · Status: built + pre-registered, awaiting the box run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261109.md` (round 11: diagnosis **size**, the gap is the reader) · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** Round 11 put the whole gap on reading: with the same evidence, Gemma gets 60.9% and the shipped reader 48.2% (p 0.0005). A bigger base is the obvious answer, but our own data says it is the wrong next spend. Going from 100M to 640M on the same reading recipe gained about 11 points. At that slope, closing 12 more points takes a base the 4090 cannot pretrain. The owner chose (2026-11-13) the cheap lever round 10 under-tested: **the recipe.** Round 10 distilled Gemma on only 2,710 pools at weight 0.5. Round 12 does it on about 37,000 at weight 1.0.
- **One command:** `.\scripts\planck3.ps1 round12`.
  - **Teach:** about 13 h, resumable.
  - **Train:** about 2 h.
  - **Pilot:** about 15 min, **0 Brave searches**.
- **Rule P:** a pilot on round 11's cached fresh10 must show +3 points before anything is spent.
- **Only a passing pilot** builds fresh12 and spends about 300 searches on the ship rule.

## 0. Why this round

- **Customer:** fresh questions are where the product has to win. Today it is right on 49% of them, Gemma on 61%.
- **Commercial:**
  - **Pilot:** about 15 box hours and $0 (Gemma is local, the pilot replays the cache).
  - **Judging (only after a passing pilot):** about $1.50 of Brave.
  - **A larger base (the alternative):** multi-day pretraining at best, for a predicted +2-3 points.
- **Research:** round 11 showed what Gemma does better, which is reading the same evidence. Distillation is the direct way to copy that skill. Round 10 is the only test of it so far, and it was small.
  - **Round 10's numbers:** 2,710 of 9,266 pools labelled. Gemma's labels moved held-out choice accuracy, but the gain did not carry to fresh10.
  - **What round 12 settles:** whether that was the method failing or the dose being too small.

## 1. What we already know (before any new run)

| Fact | Evidence |
|---|---|
| The gap is reading, not the view | round 11: Gemma top 20 vs top 16 = +0.9 pts (p 0.73). At identical evidence, Gemma +12.7 (45 vs 17). |
| Size bought about 11 points per 6.4x | round 7: Planck 1.3 rc vs Hertz rc on the same reading recipe; round 8: Planck 1.5 (same data as Hertz) ≈ Planck 1.3 |
| Size beyond 640M is expensive on this box | 1.04B ran at about 2k tok/s vs about 20k for 640M (`SETUP_hertz_202606_1.md` 3.2). 10B tokens would take weeks. |
| Round 10's distillation was small | 2,710 Brave pools taught (29% of training), weight 0.5, from round 7's reader; val choice 0.613 at epoch 0, then overfit |
| Gemma is a useful teacher | right on 70.7% of round 10's Brave pools where a right candidate exists |
| Teach speed | 1.28 s per pool (round 10 log) |

## 2. What runs

| Step | What | Size | Time |
|---|---|---|---|
| sample | seeded copies (`xreader sample`, seed 12): 30,000 of round 7's 137k public reading pools; our pools (round 6's `points.jsonl`); round 4's stable pools. The source files are never modified (`teach` rewrites its file in place). | about 36.5k pools | minutes |
| teach | Gemma reads every copied pool once (the benchmark's reader prompt, top 16). Round 10's 2,710 Brave pools are already taught. | about 36.5k + 2.7k | about 13 h, resumable |
| train | **from the shipped reader** (x-hertz rc); every pool above; Gemma's soft labels at **1.0**; 2 epochs, early stop 1, validation and test capped at 6,000 | about 45k pools | about 2 h |
| pilot | the new reader on round 11's cached fresh10, the shipped configuration (int8 CPU, top 16, pages below 7). Ledger budget 0, so no Brave call can happen. | 220 q | about 15 min |
| judge (only if P) | **fresh12**, built like fresh9-11: about 140 unused Wikidata 2026 items plus up to 80 Gemma-written questions from portal news after 3 November. Committed before any reader runs. Then the product, shipped, GPU, Gemma, heuristic and contrarian arms. | about 300 searches | about 2 h |

New pieces: `xreader sample`; `test_teacher_agreement` in every reader's `train_log.json`; `round12.py` (`pilot`, `decide`, `plan`, report); `Do-Round12`. Fresh12 joins the never-train list.

## 3. Pre-registered rules (2026-11-13, before any round 12 number exists)

| Rule | Bar | Fails → |
|---|---|---|
| **P** (pilot, cached fresh10, `r12 pilot` → `r12_pilot.json`) | the product beats round 11's `D_ship` by **≥ 3 points** with more only-right than only-wrong | **stop.** Distillation at this scale does not move the reader at identical evidence. The recipe is spent at 640M, and a larger base is the only lever left (owner's call; round 13 not built). |
| **G12** (fresh12, `r12 decide` → `r12_product.json`) | the product beats the shipped reader by **≥ 3 points**, exact McNemar p < 0.05, more only-right | keep round 9's reader |
| **Q** | int8 CPU within 2 points of the same reader on the GPU | ship blocked |
| **L** | read + write ≤ 5 s per question, CPU, 4 threads | ship blocked |
| **E** | worst-case personal profile: drop ≤ 10 points, every change flagged | ship blocked |
| GAP, D, C1 | reported only: share of the Gemma gap closed (pilot and fresh12); how often the reader picks Gemma's candidate on held-out pools; calibration | |

**Outcome:** P, G12, Q, L and E all pass → ship x-hertz r12. `serve` and `chat` pick it up from `r12_product.json`.

## 4. Disclosures (before any round 12 number exists)

1. **The pilot reuses fresh10,** which rounds 10 and 11 already read. It only gates spending. Nothing ships on fresh10.
2. **3 points on 220 questions** is about 7 questions: a direction, not proof. That is why G12 repeats the bar on new questions with a p-value.
3. **Weight 1.0 and 2 epochs are fixed now, not tuned.** One setting, one run. A miss says "distillation at this dose", not "distillation never".
4. **The reading pools are public sets** (SQuAD, NewsQA, TriviaQA, SearchQA, NQ, SQuAD 2.0). Gemma's labels on them say nothing about 2026. Benchmark questions were already dropped by `rcdata` (round 7).
5. **The teacher is wrong about 30% of the time** where a right candidate exists. The known-answer losses stay on, so a pool with a label still pulls toward the truth.
6. **Fresh12's news window** (after 3 November) overlaps fresh11's planned window. Fresh11 was never built (round 11 chose "size"), so no question is reused.

## 5. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Soft-label loss, teach, early stop | reuse round 10 (`_losses`, `teach`, `--teacher-w`, `--early-stop`) | none |
| Pools to teach | reuse round 7's `rc_points.jsonl` and round 6's points; new seeded streaming sampler that leaves sources untouched | XS |
| Mechanism metric | new `teacher_agreement` in `train_log.json` | XS |
| Pilot on cached evidence | reuse round 11's 0-search ledger and its `D_ship` / `D_gem` runs | XS |
| Rules and report | new `round12.py`, reusing round 9's paired test and round 7's product checks | S |
| Fresh12 | reuse round 9's builder (`fresh9 --prefix f12`) | XS |
| Runner | new `Do-Round12` (`planck3.ps1 round12`) | XS |

## 6. Step by step

1. `git pull`
2. Shakedown (300 reading pools, 40 taught per file, 1 epoch, 12 questions per pilot arm; about 20-30 min; no push):
   ```powershell
   .\scripts\planck3.ps1 round12 -Quick -NoPush
   ```
3. The run (keep the box awake; an interrupted teach resumes on re-run):
   ```powershell
   .\scripts\planck3.ps1 round12
   ```
   It pushes after teaching and again at the end. If the pilot passes, the same run builds fresh12 (pushed at once) and judges, provided Brave has about 300 searches left this month. Otherwise re-run after the credit resets: teach, train and pilot are kept.

## 7. Recommendations

1. Run it now. It is the last cheap test of the reader before a size decision, and the pilot costs no cash.
2. If P fails, write the size case from rounds 7, 8, 11 and 12 together, then decide on a larger base. Model it on the measured slope, not on hope.
3. Whatever P says, read **D**. If the reader picks Gemma's candidate on held-out pools far more often than before but the benchmark does not move, the skill is learned on public sets and not carried over. That points to the pools, not the size.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Teach takes longer than 13 h, or the box sleeps | resumable per pool (saved about every 500); re-running the command carries on |
| `teach` corrupts a training file | it only rewrites round 12's copies in `data/planck3/reader/r12/`; the sources are read only |
| Training on 45k pools overfits like round 7 | early stop 1 on held-out questions; the shipped reader as the start point |
| The pilot passes by noise | G12 on fresh12 with p < 0.05 |
| Brave credit runs out before judging | the check runs before any search; re-run next month keeps everything before it |

## 9. Results of record

*(empty until the box run)*
