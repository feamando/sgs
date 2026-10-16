# Planck 3.0 round 8: Planck rebuilt on the right data (Planck 1.5), and Hertz on a phone proxy

Nikita Gorshkov · 2026-10-15 · Status: shakedown clean 2026-10-16 (pretraining 92-118k tok/s, ~25 h); ready for the real run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261014.md` (round 7: R and N fail → the pre-registered rebuild) · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round8`. It is resumable, pushes results, and makes no new searches. It does two things.

**1. Planck rebuilt as "Planck 1.5"** (a working name; the version number is the owner's call).
- **What it is:** the same 100M SGS architecture as Planck 1.3, pretrained from scratch on **Hertz 1.2's corpus**. That is FineWeb-Edu + 30% Wikipedia, 9.69B tokens, with Hertz's tokenizer, reused as is.
- **Then:** round 7's reading stage, and the benchmark.
- **Why this design:** Round 7 showed that Planck 1.3 does not read at all, while Hertz reads every set. The owner's diagnosis is that Planck 1.3 was trained on the wrong data (Wikipedia only).
  - Giving Planck Hertz's data at Planck's size tests that directly.
  - It also separates size from data, which round 7 could not do.
- **Time:** about **1 day** on the 4090. The shakedown measured 92-118k tokens/s (eager mode; compile fails on the box's triton-windows), so 9.69B tokens take about 25 h.

**2. Hertz on a phone proxy, first** (about 1 hour, pushed before the pretraining starts).
- **What it checks:** round 7's best small reader, Hertz with reading training (60.1% on the benchmark, exploratory). It runs on CPU with 4 threads, in fp32 and int8, reading the top 16 or the top 8 candidates.
- **The question:** does it fit a phone's time and memory without losing accuracy?

## 0. Why this round

- **Customer:** the phone product needs a small reader that actually reads. Today's shipped reader (round 5's) is right 51% of the time, against Gemma's 63%.
- **Commercial:** there are two routes to a phone reader, and this round prices both at once:
  - **Planck 1.5:** phone-size by construction. It costs one pretraining run.
  - **Hertz:** already reads (60.1%), but it is 640M parameters and was 11 s per question on CPU in round 6b.
- **Research:** the round answers the open SGS question from round 7. Is a 100M SGS model unable to read, or was it trained on the wrong text? The comparison is clean: same data as Hertz, Planck's size.

## 1. What changes

**Planck 1.5, the base** (`scripts/train_hertz.py` with Planck's sizes; nothing new in the trainer except a step cap for the shakedown and a `final_metrics.json`):

| | Planck 1.3 (round 7) | **Planck 1.5 (round 8)** | Hertz 1.2 |
|---|---|---|---|
| architecture | d_s 128, d_f 1000, 3 passes, 4 heads, 512 context | **the same** | d_s 256, d_f 3700, 3 passes, 4 heads, 512 context |
| parameters | ~101M | **~101M** | ~641M |
| corpus | Wikipedia (20231101), ~4B tokens | **Hertz's: FineWeb-Edu sample-10BT + 30% Wikipedia, 9.69B tokens** (`data/hertz12_data`, reused) | the same as Planck 1.5 |
| tokenizer | `data/wikipedia/tokenizer.model` | **Hertz's** 32K (`data/hertz12_data/tokenizer.model`) | Hertz's |
| training | | **Hertz's settings:** 1 epoch, AdamW 3e-4 (betas 0.9/0.95, weight decay 0.1), warmup 2,000, cosine, bf16, 64 × 512 tokens per step | the same |
| run time on the 4090 | | **~25 h (measured 92-118k tok/s)** | ~5 days (~20k tok/s) |

**The reader:** round 7's stage A on Planck 1.5.
- The same 137,581 reading pools, split and settings: 1 epoch, calibrated on held-out reading questions.
- **No stage B.** In round 7 our pools added nothing for Planck and cost Hertz 4 points.
- **One fix:** the learning-rate schedule is now sized in batches. Round 7 sized it in candidates, so the last 9-15% of its reading epoch ran at learning rate 0.

**The Hertz phone proxy:** round 7's Hertz stage A reader (`checkpoints/xreader_hertz_rc/reader.pt`), on CPU only with 4 threads, template writer, the same 243 questions:
- **fp32, top 16:** 20 questions, the latency reference.
- **int8 weights** (dynamic quantization of every linear layer), top 16: all questions.
- **int8, top 8 candidates:** all questions.

The reader's weight size is measured as it sits in memory, without the LM head, which a reader never uses.

**Unchanged:** the 243 questions on cached Brave results, consolidation, the template writer, the KV cache, and rules C1, L and E.

## 2. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke | 83 offline tests | | ~2 min |
| 1. Hertz phone proxy | 3 CPU runs (above), then `r8 decide` (the outcome reads "pending"), report, **push** | `r8_x-hertz-rc_cpu-*`, `r8_product.json` | ~1 h |
| 2. Planck 1.5 pretraining | 1 epoch over Hertz's corpus | `checkpoints/planck15/` (best.pt, final.pt, `final_metrics.json`, pushed) | ~25 h |
| 3. reading stage | round 7's stage A on Planck 1.5 | `checkpoints/xreader_planck_p15rc`, `results/planck3/xreader_planck_p15rc/train_log.json` | ~25 min |
| 4. the product | benchmark; worst-case personal profile; CPU (20 questions) | `r8_x-planck15-rc_template`, `_contrarian`, `_cpu` | ~10 min |
| 5. decide + report + push | the rules below → `r8_product.json`; REPORT.md gains a "Round 8" section | git | 1 min |

## 3. Step by step

1. `git pull`
2. Shakedown, about 20-30 min. It runs 12 questions per run, 300 pretraining steps (which prints the real speed and ETA for Planck 1.5), and 1,500 reading pools:
   ```powershell
   .\scripts\planck3.ps1 round8 -Quick -NoPush
   ```
   Read the `tok/s` and ETA lines from the pretraining shakedown. If the ETA is far beyond 3 days, stop and tell me before the real run.
3. Keep the box awake: Settings → System → Power → Sleep: Never, for the next 2 days.
4. The real run:
   ```powershell
   .\scripts\planck3.ps1 round8
   ```
   - The Hertz phone result is pushed after about an hour; the pretraining then starts.
   - If anything stops, re-run the same command.
     - Finished stages are skipped.
     - The pretraining resumes from its last `step_*.pt` (saved every 2,000 steps, about every 10 minutes at the measured speed).
     - On resume the data restarts at its first batch while the step count and the schedule carry on (see disclosure 5).

## 4. Pre-registered rules (2026-10-15, before the run)

**Product.** **x-planck15 rc + template writer**, fixed: the phone-size model. Hertz is measured on the phone proxy (rule M), not selected against Planck.

- **R (primary):** `r8_x-planck15-rc_template` must pass both:
  - at or above the midpoint between round 5's reader and Gemma (0.574);
  - significantly better than round 5's reader (exact McNemar, p < 0.05). This is `round6.rule_r`, unchanged.
- **N:** does Planck 1.5 read news? On its held-out NewsQA reading pools, its choice beats its own lexical ranker by **≥ 10 points**, exact McNemar p < 0.05. This is the round 7 gate, unchanged. SearchQA is reported.
- **M (Hertz on the phone proxy):** passes if at least one **int8** configuration meets all four bars:
  - reads in **≤ 5 s** per question on CPU with 4 threads;
  - loses **≤ 2 points** against its own GPU run on the same questions;
  - is judged on at least 200 questions;
  - needs **≤ 1.5 GB** of weights.
- **C1, L, E** on the product, as in round 7:
  - C1: the 9-10 band right ≥ 90%, the 7-8 band ≥ 70%, judged at ≥ 20 answers, monotone;
  - L: ≤ 5 s on CPU;
  - E: ≤ 10 points drop under the worst-case profile, with every change flagged.
- **Reported, no bar:**
  - **size at equal data:** Planck 1.5 rc vs Hertz rc, on the benchmark and on NewsQA;
  - **data at equal size:** Planck 1.5 rc vs Planck 1.3 rc, the same two ways;
  - Planck 1.5's validation loss on Hertz's validation set (Hertz 1.2: 3.18);
  - per-group held-out accuracy, and the benchmark by regime.
- **Outcome,** written to `r8_product.json` by `planck3.py r8 decide`:

| Result | Next |
|---|---|
| R passes | **Ship Planck 1.5's reader** (`serve` picks it up automatically). |
| R fails, N passes | Planck 1.5 reads, but not our benchmark: the gap between training searches and Brave. Round 9: Brave-collected pools with Gemma soft labels on Planck 1.5. No new base. |
| R fails, N fails | A 100M SGS base does not read even on Hertz's data, so size is the lever. Next: Hertz, if M passes (confirmed on fresh questions); otherwise a mid-size base or a faster Hertz. |

**M is judged on its own:**
- **If M passes,** Hertz is a phone candidate whatever Planck 1.5 does, pending a confirmation on fresh questions once the Brave quota resets.
- **If both pass,** the phone-size Planck 1.5 ships first.

## 5. Disclosures (before any round 8 number exists)

1. **The recipe check was skipped** (owner's decision, 2026-10-15). If Planck 1.5 fails N, this round cannot tell "100M is too small" from "the reading fine-tune does not suit a 100M SGS model". The reader recipe is round 7's, with only the schedule fix.
2. **The 243 benchmark questions have been looked at often.**
   - Rounds 6-7 measured eight readers on them, and Hertz rc was picked out of four arms.
   - M compares Hertz with itself (int8 vs fp32 on the same questions), so it is not affected.
   - Any claim that Hertz rc is *better* still needs the fresh-question confirmation.
3. **The phone proxy is the box's desktop CPU with 4 threads, not a phone.** A phone CPU is slower; my rough assumption is 2-4×. Passing M is necessary for the phone, not sufficient. The next step after a pass is an on-device export and a measurement on an actual handset.
4. **The reader's learning-rate schedule changed** (sized in batches). Planck 1.5's reader is trained slightly differently from round 7's readers. The effect is small: round 7's Planck plateaued long before its learning rate reached 0.
5. **Resuming the pretraining** restarts the data at the first batch while the step budget carries on. An interruption at x% means the first x% of the data is seen twice and the last x% never. The log shows every resume.
6. **The corpus dates:** FineWeb-Edu sample-10BT is CommonCrawl up to 2024, and Wikipedia is pinned to 2023-11-01. Nothing is from 2026, so the fresh and news questions cannot leak into the base.
   - The corpus is not filtered against the benchmark, which Hertz 1.2 was not either.
   - Long-tail and general-information facts may appear in it, the same exposure as Hertz.
7. **Planck 1.5's optimizer settings are Hertz's,** not tuned for 100M. That is deliberate: the comparison with Hertz changes only the size.

## 6. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Corpus | reuse Hertz 1.2's tokenized corpus and tokenizer (`data/hertz12_data`), as is | none |
| Pretraining | reuse `train_hertz.py` with Planck's sizes; new `--max-opt-steps` (shakedown) and `final_metrics.json` | XS (code); L (box time) |
| Reading stage | reuse round 7's pools and `xreader train`; fix the schedule's step count | XS |
| Phone proxy | reuse the round 6 CPU runs; new int8 + top-k options on the reader, weight size in the summary | XS |
| Rules + report | reuse `rule_r`, `rule_n`, the product checks; new `round8.py` (M, size and data comparisons, outcome) | XS |
| Runner | reuse `Invoke-R6`; new `Do-Round8`, `Invoke-OnCpu`, logged pretraining | XS |

## 7. Recommendations

1. Run `round8 -Quick -NoPush`, check the pretraining ETA, then run `round8`.
2. Read the Hertz phone result when it lands (after about 1 hour). Don't wait for Planck 1.5.
3. **If M passes, start the fresh-question confirmation for Hertz rc** as soon as the Brave quota allows: new 2026 questions, pre-registered, the same rules. It is the fastest route to a better shipped reader.
4. Whatever the outcome, the next round follows the table in section 4.
5. Still open from earlier rounds: the 10-answer spot-check, the PROPOSED registry rows, and the runner's Brave check passing on a cached answer at 0 quota.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Pretraining is much slower than measured | the shakedown measured 92-118k tok/s (~25 h); the log prints the speed and ETA every 50 steps |
| The box sleeps, or Windows updates and restarts | keep-awake step; resumable from `step_*.pt` every 2,000 steps; re-run the same command |
| A training collapse (SGS history: Planck 1.2.x) | Hertz's proven settings and plain AdamW (no SGS-native acceleration); validation loss every 2,000 steps in the log |
| `data/hertz12_data` is missing or partial | the runner checks the files and that train.bin holds ≥ 9B tokens, and stops rather than preparing a different corpus |
| int8 quantization fails on the box's PyTorch | the engine is picked from what PyTorch supports; the fp32 run gives the latency reference regardless |
| Planck 1.5 reads but the benchmark does not move | pre-registered: round 9 = Brave pools + Gemma soft labels on Planck 1.5, no new base |
