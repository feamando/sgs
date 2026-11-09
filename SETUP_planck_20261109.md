# Planck 3.0 round 11: where is the Gemma gap? (evidence view vs the reader)

Nikita Gorshkov · 2026-11-09 · Status: built + pre-registered, awaiting the box run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261020.md` (round 10: rule G failed, round 9's reader keeps shipping) · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** Round 10 closed none of the gap: the shipped reader got 48.6% on fresh10, Gemma 60.9%. Its pre-registered next question was whether the gap is **size** (the reader) or Gemma's **evidence view**. Round 11 answers it before spending anything.
- **Phase 1, the diagnosis:** one command, `.\scripts\planck3.ps1 round11`, about 2-3 h and **zero Brave searches**. It re-reads round 10's 220 fresh questions from the cache with the view and the pages varied one at a time, then splits the 12-point gap into view, pages and reading.
- **The pre-registered diagnosis then picks one path:**
  - **config:** ship a cheaper configuration. It is judged on a new benchmark, with no training.
  - **view:** train on a wider view (round 12).
  - **size:** the reader itself is the limit, so a bigger base is the owner's call (round 12).
  - **split:** no single lever; stop.
- **Phase 2 runs only for "config":** a new fresh benchmark (`fresh11`, news after 3 November, buildable from **14 November**), about 300 searches, then the ship rule.

## 0. Why this round

- **Customer:** the shipped reader is right on 49% of fresh questions, Gemma on 61%. Three rounds of training (6b, 7, 10) moved that by at most a point. Before a fourth, we need to know whether the reader is the problem at all.
- **Commercial:**
  - **Phase 1:** no cash and one box evening.
  - **Phase 2:** about $1.50 of Brave.
  - **A "size" answer** would justify a multi-day pretraining run, the only expensive lever left. A "config" answer ships a gain with no training.
- **Research:** round 10 compared readers that did not see the same evidence. This round separates the two.

## 1. What we already know (from round 10's records, before any new run)

| Fact | Evidence |
|---|---|
| Gemma reads the top **20** candidates, the shipped reader the top **16** | `readers.py` `LLMReader`, `xreader.py` `MAX_CANDS` |
| The arms read **different pages** | Pages open only when the snippets weigh in under 7/10, and that weighing uses each reader's own scores. Gemma fetched 1.15 pages per question on fresh10, the shipped reader 1.50. In 62 of 220 questions the two fetched a different number. |
| "Longer contexts" is **not** a Gemma advantage | Gemma sees 200 characters per candidate, the small reader 240 (all a candidate carries) |
| The gap is mostly Gemma-only wins | Gemma right / shipped wrong: 48. Shipped right / Gemma wrong: 21. Both right: 86. Both wrong: 65. |
| The latency "regression" (round 8 1.33 s → round 9 3.35 s) tracks pages read | Round 8: 78 of 243 questions opened no page (median 0.7 s) vs round 9: 42 of 220. With 2 pages: 1.8 s (round 8) vs 4.0 s (round 9). So likely more and longer pools, not a slower model. Phase 1 measures milliseconds per candidate to confirm. |

So the round 10 doc's option "longer contexts" is dropped. The view question becomes **top 16 vs 20 vs 32, and which pages get read**.

## 2. Phase 1: the diagnosis (no searches)

All arms read the **same cached Brave answers** for fresh10. A ledger with a budget of 0 makes any live Brave call impossible. A page that some arm newly opens is fetched from the web (free) if it is not cached, and counted. Every arm also writes what it read (`pools.jsonl.gz`, pushed).

| Arm | Reader | View | Pages | What it isolates |
|---|---|---|---|---|
| `D_ship` | shipped (x-hertz rc, int8 CPU) | top 16 | when < 7 | replication of round 10 |
| `D_ship32` | shipped | **top 32** | when < 7 | a wider view, no training (untrained beyond 16) |
| `D_shipP` | shipped | top 16 | **always 2** | the pages it skips today |
| `D_ship32P` | shipped | top 32 | always 2 | both; its dump gives the **reach** (is a right candidate in view at all) |
| `D_gem` | Gemma | top 20 | when < 7 | replication of round 10 |
| `D_gem16` | Gemma | **top 16** | when < 7 | Gemma's view advantage |
| `D_gem16P` | Gemma | top 16 | always 2 | the same evidence as `D_shipP`: pure reading |
| `D_old` | shipped | top 16 | when < 7 | round 8's 243 questions: ms per candidate (the latency) |

**The gap decomposes as:** Gemma's view (`D_gem − D_gem16`), the small reader's missing pages (`D_shipP − D_ship`), and reading at identical evidence (`D_gem16P − D_shipP`). The shares are reported; they need not add up exactly (the page rule interacts with each reader's confidence).

## 3. Phase 2: the judging (only if the diagnosis says "config")

- **A new fresh benchmark,** built like rounds 9-10 (`fresh9 --prefix f11 --after 2026-11-03 --seed 11` → `planck3_tasks_fresh11.json`):
  - about 140 Wikidata 2026 items unused by rounds 1-10 (fresh9 and fresh10 now count as used);
  - up to 80 Gemma-written questions from portal news after 3 November.
  - It is committed and pushed before any reader runs. The 10-day news window means the runner refuses to build it before **14 November**.
- **Arms:**
  - the product: round 9's reader with the diagnosed configuration, int8 CPU;
  - the shipped configuration;
  - Gemma;
  - the heuristic;
  - the worst-case personal profile.
- **Searches:** about 300 live Brave searches.

## 4. Pre-registered rules (2026-11-09, before any round 11 number exists)

**Diagnosis** (`planck3.py r11 diagnose` → `r11_plan.json`; checked in order, the first that holds wins):

| Check | Bar | Outcome → next |
|---|---|---|
| **Valid** | `D_ship` shows round 10's value on ≥ 90% of the questions | else **invalid**: not a replay (pages changed); nothing is decided |
| **Config** | the best of `D_ship32` / `D_shipP` / `D_ship32P` beats `D_ship` by **≥ 3 points**, with more only-right than only-wrong | **config** → phase 2 judges that configuration, no training |
| **View** | Gemma's view is worth **≥ 3 points** (`D_gem − D_gem16`) **and** the top 32 hold a right candidate on ≥ 3 points more questions than the top 16 (reach) | **view** → round 12 (not built): train the reader on a top-32 view |
| **Size** | at identical evidence Gemma reads **≥ 8 points** better (`D_gem16P − D_shipP`) | **size** → round 12 (not built): the owner's call on a larger SGS base |
| none | | **split** → no single lever; report and stop |

**Phase 2** (`planck3.py r11 decide` → `r11_product.json`):
- **G11:** the product beats the shipped configuration on fresh11 by **≥ 3 points**, exact McNemar p < 0.05.
- **L:** read + write ≤ 5 s per question on CPU (as rounds 8-10); page fetch time reported.
- **E:** as before.
- **C1:** reported.
- **Outcome:** G11 + L + E → ship. `serve` / `chat` read `reader_cands` and `open_below` from `r11_product.json`.

## 5. Disclosures (before any round 11 number exists)

1. **Phase 1 reuses fresh10, which round 10 already used.** Fresh10 is used here only to diagnose, never to ship. Anything that ships is judged on fresh11.
2. **The 3-point bars** are about 7 questions out of 220: a direction, not proof. That is why "config" must pass again on new questions. The 8-point size bar is two thirds of the gap.
3. **Reading the top 32 is outside the training distribution.** The reader was trained on 16, and its position feature never saw slots 17-32. A loss at top 32 says "untrained", not "a wider view does not help". The reach number answers the second question.
4. **Pages newly opened** by the always-read arms are fetched live (no Brave cost) and can differ from what existed on 4 November. The replay check covers the arms that open the same pages as round 10. The new fetches are counted in each run's summary.
5. **Order matters in the rules.** "Config" is checked first because it is the only outcome that ships without training. It can mask a view or size effect, which the report still shows.

## 6. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Replay on cached evidence | reuse round 10's cache, `--net live` with a 0-search ledger (round 10's ledger code) | XS |
| View knob | reuse `--reader-cands`; new: Gemma accepts it too, the cross-reader reads up to 32 | XS |
| Always read pages | new `--open-below` on `r5` (the answer pipeline's threshold, default 7 unchanged) | XS |
| What the reader saw | new `--dump-pools` → `pools.jsonl.gz`; `read_pools` / `read_cands` on every record | XS |
| Diagnosis + rules | new `round11.py` (`diagnose`, `decide`, report), reusing round 9's paired tests | S |
| Fresh11 | reuse round 9's builder (`fresh9 --prefix f11`); fresh11 joins the never-train list | XS |
| Runner | new `Do-Round11` (`planck3.ps1 round11`) | XS |

## 7. Step by step

1. `git pull`
2. Optional shakedown (12 questions per arm, no push):
   ```powershell
   .\scripts\planck3.ps1 round11 -Quick -NoPush
   ```
3. The diagnosis (about 2-3 h, 0 searches, pushes `r11_plan.json` and the dumps):
   ```powershell
   .\scripts\planck3.ps1 round11
   ```
4. Only if `r11_plan.json` says `config`: on or after 14 November, the same command. It keeps phase 1, builds fresh11 (pushed at once) and judges (about 1.5 h, about 300 searches).

## 8. Recommendations

1. Run phase 1 now. It costs nothing, and every later choice (train on a wider view, pretrain a bigger base, or ship a configuration) depends on it.
2. Hold any new training or pretraining until the diagnosis is in.
3. Read the latency numbers in the same report. If milliseconds per candidate are flat across rounds, the "regression" is evidence size, and the phone default (top 16 vs top 8) should be set on candidates read, not on the model.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| The replay does not reproduce round 10 (cache misses, changed pages) | the 90% replay check gates every outcome; a 0-search ledger means a cache miss is empty, never a new Brave answer |
| A configuration wins on fresh10 by noise | it must pass G11 on fresh11, built after the diagnosis and committed before any reader runs |
| Always reading pages breaks the 5 s phone budget | rule L on the product run; fetch time reported separately |
| Gemma's top-16 arm changes its prompt in a way that is not just "fewer candidates" | the prompt is identical, only the candidate list is cut |
| "Split" leaves no next step | then the decomposition table itself is the result, and round 12 is the owner's call with the numbers in hand |

## 10. Results of record

*(empty until the box run)*
