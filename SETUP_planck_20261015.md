# Planck 3.0 round 8: Planck rebuilt on the right data (Planck 1.5), and Hertz on a phone proxy

Nikita Gorshkov · 2026-10-15 · Status: run 2026-10-17/18 (section 9: R FAILS, N FAILS → pre-registered outcome "size"; M PASSES: Hertz int8 reads in 1.3 s on CPU at no accuracy cost) · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261014.md` (round 7: R and N fail → the pre-registered rebuild) · Plan and results of record: `SETUP_092026_planck3.md`

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

## 9. Round 8 results of record (run 2026-10-17 15:24 to 2026-10-18 15:07, 4a293a2)

**TL;DR.**
- **Hertz fits the phone proxy (M PASSES).** int8 weights, reading the top 16 candidates:
  - 1.33 s per question on CPU with 4 threads (fp32 takes 6.46 s);
  - 888 MB of weights (fp32: 1,997 MB);
  - **60.5% right, vs 60.1% for the same reader on the GPU** (3 vs 2 only-right; no loss).
  - With the top 8 candidates: 0.75 s per question, 58.8% (-1.2 points, inside the 2-point bar).
- **Planck 1.5 trained cleanly but reads no better than Planck 1.3.**
  - The base: 9.67B tokens, validation loss 3.449 on Hertz's validation set (Hertz 1.2: 3.183).
  - Its reader:
    - NewsQA 46.1% vs the ranker's 47.7%, so **N FAILS** (Planck 1.3: 46.4%);
    - the benchmark 52.3%, so **R FAILS** (vs round 5's reader p 0.90).
- **The pre-registered outcome is "size":** a 100M SGS base does not read even on Hertz's data. The reader path is Hertz, and M passes, so the next step is confirming it on fresh questions.
- **The data diagnosis did not hold, for the reader.** Planck 1.5's reader gives the **same answer as Planck 1.3's on 235 of 243 questions.** On the reading sets it is within 1 point of Planck 1.3 on every group.
- **What ships is unchanged:** round 5's Planck reader + the template writer. No pre-registered product has passed R.

**Planck 1.5, the base:** 100.9M parameters, 295,143 steps, 9.67B tokens, eager mode (torch.compile fails on the box's triton-windows), no interruptions, ~22 h.

| step | val loss (Hertz's validation set) |
|---|---|
| first eval | 4.079 |
| final | **3.449 (ppl 31.5)** |
| Hertz 1.2 (640M, same data) | 3.183 (ppl 24.1) |

At equal data, 6.4× the parameters buys 0.27 nats.

**The reading stage** (round 7's stage A; held-out test pools with a right candidate; choice vs the lexical ranker):

| set | ranker | Planck 1.3 rc | **Planck 1.5 rc** | Hertz rc |
|---|---|---|---|---|
| SQuAD | 0.601 | 0.560 | **0.562** (below the ranker, p 0.0004) | 0.654 |
| **NewsQA (gate N)** | 0.477 | 0.464 | **0.461 (p 0.24) FAIL** | 0.580 |
| TriviaQA-web | 0.422 | 0.508 | **0.513** | 0.717 |
| SearchQA | 0.462 | 0.522 | **0.533** (+7.1, p 2e-6) | 0.677 |
| NaturalQuestions | 0.389 | 0.399 | **0.398** | 0.628 |
| all | 0.474 | 0.487 | **0.489** | 0.643 |
| train loss (start → end) | | 1.85 → 1.80 | **1.84 → 1.80** | 1.79 → 1.43 |

The schedule fix ran: 25,669 steps sized in batches, with the learning rate decaying to the end.

**The benchmark: reading only, template writer, the same 243 questions on Brave (cached):**

| reader | right | fresh (60) | long tail (83) | news (60) | general info (40) | 9-10: n / right | 7-8: n / right |
|---|---|---|---|---|---|---|---|
| heuristic (no model) | 0.498 | 0.22 | 0.55 | 0.60 | 0.65 | 33 / 0.76 | 72 / 0.67 |
| round 5 Planck reader (ships) | 0.514 | 0.40 | 0.66 | 0.40 | 0.55 | 39 / 0.64 | 66 / 0.62 |
| x-planck rc (Planck 1.3, round 7) | 0.531 | 0.25 | 0.60 | 0.63 | 0.65 | 14 / 0.71 | 77 / 0.66 |
| **x-planck15 rc (the product)** | **0.523** | 0.25 | 0.59 | 0.62 | 0.65 | 14 / 0.71 | 77 / 0.65 |
| x-hertz rc, GPU (round 7) | 0.601 | 0.47 | 0.66 | 0.60 | 0.68 | 29 / 0.72 | 88 / 0.74 |
| **x-hertz rc, CPU int8, top 16** | **0.605** | 0.47 | 0.66 | 0.63 | 0.65 | 27 / 0.70 | 87 / 0.75 |
| x-hertz rc, CPU int8, top 8 | 0.588 | 0.45 | 0.63 | 0.63 | 0.65 | 27 / 0.70 | 90 / 0.72 |
| Gemma 4 E4B | 0.634 | 0.58 | 0.71 | 0.57 | 0.65 | 44 / 0.70 | 128 / 0.70 |

Paired, the same 243 questions (exact McNemar):
- **Planck 1.5 rc:**
  - vs Planck 1.3 rc: 0 vs 2, p 0.5;
  - vs round 5's reader: 34 vs 32, p 0.90;
  - vs the heuristic: 14 vs 8, p 0.29;
  - vs Hertz rc: 8 vs 27, p 0.002;
  - vs Gemma: 26 vs 53, p 0.003.
- **Hertz rc, int8, top 16:**
  - vs its own GPU run: 3 vs 2, p 1.0;
  - vs round 5's reader: 46 vs 24, p 0.012;
  - vs the heuristic: 37 vs 11, p 0.0002;
  - vs Gemma: 28 vs 35, p 0.45.
- **Hertz rc, int8, top 8:** vs its GPU run 2 vs 5, p 0.45; vs Gemma 28 vs 39, p 0.22.

**The verdicts:**

| rule | result | verdict |
|---|---|---|
| **R** (product: x-planck15 rc) | 0.523 vs midpoint 0.574; vs round 5's reader p 0.90 | **FAIL** |
| **N** | NewsQA 0.461 vs ranker 0.477 (-1.6 points, p 0.24) | **FAIL** |
| N, SearchQA (reported) | +7.1 points, p 2e-6 (under the 10-point margin) | fail |
| **M** (Hertz, phone proxy; CPU: AMD Zen 4, 4 threads) | int8 top 16: 1,331 ms, 888 MB, -0.4 points (n 243). int8 top 8: 752 ms, 888 MB, -1.2 points (n 243). fp32 top 16: 6,460 ms, 1,997 MB (20 questions, reference) | **PASS** (both int8 configurations) |
| C1 (product) | 9-10: 14 answers (too few to judge); 7-8: 77, 64.9% (bar 70%); not monotone | **FAIL** |
| L (product) | 0.91 s per question on CPU | **PASS** |
| E (product) | no drop under the worst-case profile; 199 of 199 changed answers flagged | **PASS** |

**Outcome (pre-registered, `r8_product.json`):** "size: a 100M SGS base does not read even on Hertz's data; the reader path is Hertz (M passes: confirm it on fresh questions)".

**What the run says beyond the rules.**
1. **The base is not what holds Planck's reader back.** Two 100M bases with different corpora, tokenizers and loss levels produce readers that agree on 97% of the benchmark and are within 1 point on every reading set. Their training losses stall at the same value (1.80). When changing the whole backbone changes nothing, the reader is not using the backbone much. Its answers are carried by the head's lexical features: Planck 1.5's reader matches the heuristic on 182 of 243 questions, against 148 for Hertz.
2. **So "100M is too small" and "the reader recipe does not suit a 100M SGS model" are still not separated** (disclosure 1). This run makes the recipe explanation more likely, not less. The cross-encoder head reads one hidden state of d_f = 1,000 at Planck's size, against 3,700 for Hertz, and the learning rates were never tuned for Planck.
3. **Hertz is now a real phone candidate.**
   - int8 costs it nothing measurable on the benchmark.
   - 888 MB fits an 8 GB phone.
   - At 1.33 s on a desktop CPU, a phone 2-4× slower would read in about 2.7-5.3 s (top 16) or 1.5-3 s (top 8).
   - The size is mostly the embedding and output tables, which dynamic int8 leaves in fp32. The reader never uses the LM head (already excluded from the 888 MB). The input embeddings could be quantized too.
4. **Hertz rc on CPU int8 would itself pass rule R** (60.5%; p 0.012 vs round 5's reader). But it was selected on these 243 questions, so this is not a ship decision; the fresh-question confirmation is.
5. **Confidence bands are still not honest** for any small reader: C1 fails everywhere, and the 9-10 band is right 70-72% of the time, not 90%. Gemma's bands are no better on Brave here (70%). C1 needs its own work whichever reader ships.

**Recommendations (owner's call; not built):**
1. **Round 9 = the Hertz path, as pre-registered:**
   - confirm x-hertz rc (int8, top 16) on **fresh 2026 questions** collected on Brave, with the same rules and the product fixed before the run;
   - first fix the runner's Brave check, which passes on a cached answer at 0 quota;
   - check the Brave quota first.
2. **Measure it on a handset:** export the int8 reader (ExecuTorch or ONNX Runtime Mobile) and time it on a real phone. Rule M was a desktop proxy.
3. **Calibration (C1)** as its own small piece on the chosen reader: refit the bands on held-out benchmark-like questions.
4. **Planck stays the research arm, not the product path.** The one cheap test that would tell recipe from size is the 30-45 min check from round 7:
   - a features-only head (does the backbone add anything?);
   - a learning-rate sweep on Planck 1.5's reader.
   
   Run it only if a phone-size SGS reader still matters after Hertz is confirmed.
5. **Name:** "Planck 1.5" is a working name for the new base. It is a better language model than Planck 1.3 on mixed text, and may be the better Planck base for other SGS work. Keep or rename.

