# Planck 3.0 round 3: fresh + long-tail benchmark, G2 from known answers, G1 confirmation

Nikita Gorshkov · 2026-10-07 · Status: ready to run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous rounds: `SETUP_planck_20260901.md` · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round3` (about 2-2.5 h, idempotent, pushes results), answers the three questions round 2 left open:
1. **Does the 93.6% hold** on questions nobody tuned against, and does Planck beat base chat where its memory ends? A new **fresh + long-tail benchmark**: 143 questions generated from Wikidata. Fresh means 2026 facts; long-tail means obscure items with ≤3 Wikipedia language editions. Each question carries a dated, sourced answer.
2. **Can Planck learn the G0 decisions from known answers** instead of imitating a teacher that is now worse than the heuristic? This is **G2, redesigned**: 1,455 disjoint training questions → labelled decision points → a candidate scorer with a "none of these" option → the `planck` policy.
3. **Does the `rank` arm hold up on races it was not picked on?** The **G1 confirmation**: new task seed (new targets, new test races), new model seeds 3-5, `planck-rank` declared primary, plus the Hertz arm.

The pass rules below were written **before** the run (2026-10-07).

## 0. Why this round

- **Customer:** the product is a direct answer + retrieval in depth, in chat. Round 2 showed it already matches base chat on famous, stable facts (93.6% vs 95.7%) with sources attached. A user asks about what happened this year and about things no model memorized, so that is the test that matters.
- **Commercial:** the no-model path costs ~$0 per answer. If a ~100M Planck policy keeps or improves quality on fresh/long-tail questions, the product is "better than base chat where it counts, at a fraction of the cost". If it does not, the deterministic tools are the product and Planck is a latency play only.

## 1. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke + doctor | as before (49 offline tests) | `doctor.json` | ~3 min |
| **G0 fresh** | 143 questions: Gemma closed-book (the base-chat rival), Gemma on our tools (snippet-first), heuristic (snippet-first) | `results/planck3/g0f_*` | ~20 min |
| **G2 collect** | 1,455 training questions: 1 search + 1 page each, every candidate labelled right/wrong against the known answer. Resumable | `data/planck3/g2/points.jsonl` | ~45-60 min |
| **G2 train** | candidate scorers on hash (control) and Planck features | `results/planck3/g2_head_{hash,planck}_s0/` | ~5 min |
| **G2 eval** | the `planck` policy on the fresh benchmark and on the 47-question seed benchmark | `results/planck3/g0f_planck-g2-*`, `g0_planck-g2-*` | ~10 min |
| **G1 confirm** | task seed 1, model seeds 3-5 × {hash, planck, planck-rank, hertz, hertz-rank}, Gemma teacher on 300 new races | `results/planck3/g1_*_t1/`, `g1_aggregate_t1/` | ~40 min |
| report + push | `REPORT.md` gains "By regime", "G2 heads", "G1 across seeds (t1)" | git | 1 min |

## 2. Step by step

1. `git pull`
2. Start **Docker Desktop** (SearXNG matters more this round: G2 collect makes ~1,400 searches).
3. Shakedown first, about 15 min. It skips G1 and uses 12 benchmark questions plus 100 training questions:
   ```powershell
   .\scripts\planck3.ps1 round3 -Quick -NoPush
   ```
   Check that `REPORT.md` has `g0f_*_quick` rows, a "By regime" table, and a "G2 heads" row for hash and planck.
4. The real run:
   ```powershell
   .\scripts\planck3.ps1 round3
   ```
   If it stops, run it again; finished stages are skipped and G2 collection resumes where it stopped.
5. Nothing to send: it pushes. I `git pull` and read `REPORT.md`.

## 3. Pre-registered rules (2026-10-07, before the run)

**A. Fresh + long-tail benchmark (the product's case against base chat).**
- **A1:** On **each** regime (fresh, long_tail), the snippet-first heuristic's success is ≥ Gemma closed-book's success.
- **A2:** Gemma closed-book is expected to collapse on `fresh` (2026 facts are after its training data). If it does not (≥50% on fresh), the benchmark is not testing what it claims, and I report that instead of a verdict.
- **A3 (transfer check, not a product verdict):** if the heuristic scores ≥ 80% on the fresh benchmark, round 2's 93.6% transfers to questions nobody tuned against. A lower score is **ambiguous**: these questions are also harder (obscure items, thinner snippets), so a drop can mean difficulty, seed-set tuning, or both. It is reported per template, not read as proof of either. The clean test of over-tuning is section B: a scorer learned on disjoint questions vs the hand-tuned ranker, both on unseen questions.

**B. G2, redesigned (learned from known answers).** The original G2 bars, unchanged. G2 PASSES if the `planck` policy (Planck encoder) meets all of:
- seed benchmark success ≥ 0.9× closed-book (≥ 86.1%);
- fresh benchmark success ≥ closed-book **and** ≥ the heuristic − 2 pts (a learned scorer must not lose to the deterministic ranker it is trained over);
- wrong-when-answered < 5% and ECE < 0.05 on the fresh benchmark;
- **it beats the hash-feature control** (`planck-g2-hash`) on the fresh benchmark. If not, Planck features add nothing to G2.

Cost and depth are reported (the `planck` policy uses 0 LLM tokens by construction).

The policy itself is fixed before the run:
- EXTRACT the top candidate when its calibrated p ≥ 0.5, else read a page;
- read **at most 2 pages per question**, then abstain;
- ANSWER under the 0.3 confidence gate.

**C. G1 confirmation.** **Primary arm: `head:planck-rank`** (promoted from round 2's secondary arm). It is tested on task seed 1 (new targets and test races, so picking the arm on seeds 0-2 cannot leak) with model seeds 3, 4, 5 and the Gemma teacher on the first 300 new test races. **PASS** if all three hold:
- the mean paired ratio to the teacher is ≥ 0.8;
- it beats `head:hash` in every seed;
- the warm decision takes < 100 ms.

Secondary, exploratory: `head:hertz-rank`, `head:hertz` (the capacity arm), `head:planck`. A secondary-only pass needs its own confirmation round.

## 4. The new benchmark, in brief

- **File:** `scripts/assets/planck3_tasks_fresh.json`, generated by `src/planck3/benchgen.py` from Wikidata SPARQL, cached.
- **Fresh (2026):**
  - event winners ("Who won the 2026 UEFA Europa League final?")
  - election winners
  - champions of seasons that ended in 2026
  - heads of government who took office in 2026
- **Long-tail (≤3 Wikipedia language editions):**
  - founding year of small businesses
  - birth year of lesser-known athletes, politicians, actors, writers and musicians
  - elevation of minor mountains
  - founders and authors
  - two-turn chat conversations over the same kind of subjects
- **Gold** is the Wikidata label + English aliases, stamped `gold_as_of`. Descriptions that leak the answer ("born 1987") are dropped.
- **Not tuned against any policy:** the generator was fixed for query timeouts and malformed responses only, never after looking at a policy's answers.
- **Known noise:** Wikidata can disagree with the web (founding vs launch year, renamed teams). That noise is the same for every policy, closed-book included, so the comparisons stay fair even where an absolute number is a little low.

### What the Mac check found (2026-10-07, before the box run; not results of record)

The Mac has no SearXNG, so it searches through the Wikipedia API (all words must match, short highlight snippets), and its numbers mostly measure that.

- **Heuristic on this benchmark (Mac, Wikipedia API):** 14.0% overall (fresh 5.0%, long-tail 20.5%).
- **The bottleneck is reachability, not choice:** for "Who won …" questions the right name was among the candidates in only 6/30 tasks (wrong pages: group stages instead of the final). Where the answer was reachable (birth years 12/20), the heuristic mostly got it (9/20).
- On fresh "who won" questions the answer sits in a table cell ("Champions: Spain"), which is exactly where a learned scorer (G2) has room to beat the hand-written ranker.

**Three pipeline fixes were made after looking at this benchmark on the Mac.** All are bug fixes, not ranking changes, and are disclosed so the box numbers can be read with that in mind:
1. **Wikipedia search retries with the question's subject when the full question finds nothing.** The API requires every word to match; this mattered for 27/143 questions. SearXNG is unaffected; this protects the fallback.
2. **A month or weekday is never offered as an entity answer.** Type correctness: "June" is not who won the World Cup.
3. **Empty search results are no longer cached.** A rate-limited engine during G2's ~1,400 searches would otherwise store "nothing found" for good.

No candidate scoring or ranking was changed after seeing this benchmark.

### Attempt 1 on the box (2026-10-07, commit 343e0e2): search blocked, half the round invalid

- **What happened:**
  - The shakedown worked: the heuristic answered 8 of 12 fresh questions, against 0 of 12 for base chat.
  - Minutes into the full run, SearXNG's upstream engines blocked it. SearXNG kept answering HTTP 200 with empty result lists. The old health check only tested that a results field existed, so every stage ran blind.
  - G2 collection got an empty result for nearly every search (699 of the first 700), and G2 trained on 84 points instead of thousands.
- **Invalid, to be re-run:** every `g0f_*` run that searches (`gemma_snip`, `heuristic_snip`, `planck-g2-*`), `g0_planck-g2-*`, and both G2 heads.
- **Valid (no web search involved):**
  - **Base chat on the new benchmark:** 4.9% (fresh **0.0%**, long-tail 8.4%), answering 98.6% of the time with 95% of those answers wrong. Rule **A2 holds**: base chat collapses where its memory ends.
  - **G1 confirmation (task seed 1, seeds 3-5): FAIL as pre-registered.**
    - `head:planck-rank` 23.5% ± 0.5, paired ratio to the teacher **0.71** (0.67 / 0.80 / 0.66); the teacher scored 33.7% on 300 new races (no overlap with round 2's races).
    - It beats `head:hash` in every seed (p ≤ 1e-18) and lexical (p ≤ 1e-9).
    - Its gain over `head:planck` shrank to +1.7 pts (p = 0.45 / 0.06 / 0.19), so round 2's +4.3 was partly selection on those seeds.
    - The Hertz arm did not run (no checkpoint found).
- **Fixed before attempt 2:**
  1. The SearXNG health check requires real results.
  2. Every empty SearXNG search waits, retries once, then falls back to Wikipedia for that query (logged as `fallback_rate`).
  3. ≥1.5 s between searches.
  4. A circuit breaker stops a run after 15 empty searches in a row.
  5. Every run records `search_health`; a run with >20% empty searches is labelled **INVALID**, and `round3` re-runs invalid stages automatically.
  6. The runner waits up to 10 min for a blocked SearXNG to recover.
  7. Hertz is auto-detected in `checkpoints/hertz/`, `checkpoints/hertz12/` (incl. the latest `milestone_*.pt`).

**Attempt 2:** `git pull`, start Docker Desktop. Let SearXNG rest an hour, or recreate it with `docker rm -f planck3-searxng`; the runner makes a new one. Then run `.\scripts\planck3.ps1 round3`. Only the invalid stages re-run; closed-book and G1 are kept, and G1 adds the Hertz arm if it finds the checkpoint.

### Attempt 2 on the box (2026-10-07, commit a77e502): valid, but on Wikipedia search. Results of record.

- **Search:**
  - All runs are valid (0.7% empty searches).
  - The recreated SearXNG container took longer than the runner's 60 s wait to come up, so **every tools run used the Wikipedia API**, the weaker retrieval layer. On this benchmark it put the answer in the top-3 snippets only 18.9% of the time.
  - Comparisons between policies are fair (same retrieval); absolute numbers understate SearXNG. Fixed: the runner now waits up to 5 min for a fresh container, prints its log if it never comes up, and once SearXNG answers it keeps these runs as `*_wikipedia` and redoes them.

**A. Fresh + long-tail benchmark (143 questions):**

| Policy | Overall | Fresh (2026) | Long-tail | Answered | Wrong when answered |
|---|---|---|---|---|---|
| Base chat (Gemma closed-book) | 4.9% | **0.0%** (0/60) | 8.4% | 98.6% | 95.0% |
| Heuristic, snippet-first | **15.4%** | 8.3% (5/60) | 20.5% | 79.7% | 80.7% |
| Gemma on our tools | 14.0% | 3.3% | 21.7% | 21.0% | 33.3% |
| `planck` policy (G2, Planck) | 9.8% | 1.7% | 15.7% | 18.2% | 46.2% |
| `planck` policy (G2, hash control) | 0.7% | 0.0% | 1.2% | 4.2% | 83.3% |

- **A1 holds.** The heuristic ≥ base chat on both regimes: fresh 5 vs 0 (McNemar p = 0.063), long-tail 17 vs 7 (16 vs 6 discordant, p = 0.053). Overall 21 vs 6 discordant, **p = 0.006**.
- **A2 holds.** Base chat answers almost every 2026 question and gets all 60 wrong, confidently and without sources.
- **A3: ambiguous, as pre-registered.** The heuristic's 15.4% is far below 80%, but retrieval (Wikipedia search) is the dominant confound. The heuristic also answers far too often (wrong 81% when it answers), where Gemma-on-tools abstains much more (wrong 33%).

**B. G2: FAIL** (seed benchmark 2.1% vs the ≥86.1% bar; fresh 9.8% vs heuristic − 2 = 13.4%; wrong-when-answered 46%; ECE 0.24).
- **What worked:**
  - Offline, the scorer learned real signal: choice accuracy **59.4% vs the deterministic ranker's 39.8%** on 133 held-out decision points with a right candidate. The hash control scored 51.9%.
  - Planck features beat hash end to end (9.8% vs 0.7%).
- **Why it failed:** one softmax over candidates + "none of these" answers two questions at once (which candidate? is any candidate right?).
  - The training points came from Wikipedia search, where 52% had no right candidate, so "none of these" absorbs the probability. On the easier seed questions the policy is badly under-confident: 137 of 194 decisions were below 0.1, including the right answer ranked **first** (LEGO 1932 at p = 0.33, Nintendo 1889 at 0.05, Sony 1946 at 0.12).
  - It then reads two pages and abstains: a design flaw (choice and answerability conflated, plus a train/deploy shift), not a capacity limit.

**C. G1 confirmation: FAIL confirmed; capacity is not the lever.**

| Arm (task seed 1, seeds 3-5) | Rollout success | Paired ratio to teacher |
|---|---|---|
| `head:planck-rank` (primary) | 23.5% ± 0.5 | **0.71** |
| `head:hertz-rank` (640M) | 23.7% ± 1.1 | 0.70 |
| `head:planck` | 21.8% ± 0.9 | 0.63 |
| `head:hertz` | 19.8% ± 0.9 | 0.47 |
| `head:hash` | 10.7% ± 0.5 | 0.25 |

- Hertz-rank = Planck-rank in every seed (p = 0.39 / 0.51 / 0.27).
- Hertz with the original objective is worse than Planck (p = 0.007, 1.0, 0.046).
- Both are ~0.7× the teacher at ~0.5-0.7 ms per decision (~350-500x faster).

### Round 3b: re-run on SearXNG + G2 v2 (built 2026-10-08, PRE-REGISTERED before the run)

**Why G2 v2.** v1 conflated "which candidate?" with "is any candidate right?" in one softmax with a "none of these" option. Trained where 52% of points had no right candidate, it ranked the right answer first yet gave it p 0.05-0.33, and abstained. v2 separates the two:
- **Choice head:** a softmax over candidates only, trained only on decision points that have a right candidate, temperature-calibrated.
- **Answerability gate:** a calibrated logistic model of P(the chosen candidate is right). Its inputs are the choice head's top-1 probability, margin and entropy, the candidate's deterministic features and the answer type. It is fit on **validation questions** the choice head never trained on.
- **Threshold τ, fixed by rule now:** the lowest gate score whose validation precision is ≥ **0.90**. If no threshold reaches 0.90, τ = 1.01 and the policy never answers; that is reported as coverage 0, not hidden. Coverage and precision at τ are reported on a separate **test split** (10% of questions).
- **Policy:** EXTRACT the choice head's pick when gate ≥ τ; otherwise read a page (max 2) or abstain. The 0.3 ANSWER gate stays.
- **Every decision logs** `argmax`, its value, the top-3 choice probabilities, the gate score and τ.
- **Training data:** collected with the deployment search. With SearXNG up, the runner collects into `points_searxng.jsonl`; if SearXNG is down it reuses the Wikipedia points and says so.

**Pass rules (G2 v2):** rule B unchanged, applied to `planck-g2v2-planck`:
- seed benchmark ≥ 86.1%;
- fresh ≥ base chat **and** ≥ the heuristic − 2 pts (same search backend);
- wrong-when-answered < 5% and ECE < 0.05 on fresh;
- beats `planck-g2v2-hash` on fresh.

**Required diagnostics:** the v2 choice head must beat the deterministic ranker on the test split (`test_choice_acc` > `test_ranker_acc`); gate AUC; coverage at τ.

**Run it:**
1. `git pull`
2. Start Docker Desktop and leave the existing `planck3-searxng` container in place; the runner now waits up to 5 min for it.
3. `.\scripts\planck3.ps1 round3`

What that does:
- The Wikipedia-backed G0-fresh runs are kept as `*_wikipedia` and redone on SearXNG.
- G2 v2 collects ~1,455 questions on SearXNG. That takes about 60-75 min with the 1.5 s pacing; it is resumable, and the circuit breaker stops it if SearXNG gets blocked.
- G2 v2 then trains and is evaluated.
- G1 is kept (its evals re-run quickly from the cache).

### Round 3b, attempt 1 (2026-10-07 21:38, commit 59befff): partial

- **G2 v2 did not run.** The run started at 21:38 on the previous commit; the v2 code landed at 21:47. It re-ran the old v1 instead.
- **SearXNG blocked again** after ~150 searches (the Gemma run).
  - Root cause: an engine that answers with a CAPTCHA is **suspended by SearXNG for 24 h (7 days for Google reCAPTCHA)**. Waiting minutes could never help.
  - Recreating the container clears the suspensions; that is why the first ~150 searches worked.
  - A runner bug made it worse: it moved runs aside "to redo on SearXNG" and then redid them on Wikipedia, because it trusted a SearXNG check made at the start of the round.
- **Result of record (the one SearXNG-backed run; 0% empty, 0% fallback):**
  - **Gemma on our tools 28/143 (19.6%) vs base chat 7/143 (4.9%):** 26 vs 5 discordant, **McNemar p = 0.0002**. Fresh 5/60 vs 0/60; long-tail 23/83 vs 7/83 (p = 0.002).
  - The same policy on SearXNG beats itself on Wikipedia search, 8 vs 0 discordant (p = 0.008): **retrieval is the lever on this benchmark**.
  - Even on SearXNG the answer is in the top-3 snippets only 27% of the time (19% on Wikipedia).
- **Fixed (respecting the engines, not evading them):**
  - `config/searxng/settings.yml` removes Google and enables Bing, Brave, DuckDuckGo, Mojeek, Qwant, Startpage and Wikipedia (names verified against SearXNG's defaults). Suspension times are kept at their defaults.
  - The runner recreates the container when `settings.yml` changes.
  - 3 s between searches.
  - A live SearXNG check before moving any run aside.
  - Every G2 decision point records which backend answered it.
- **Product note:** a local SearXNG scraping public engines is a research harness, not a production search layer. A deployed Planck would sit on a licensed search API; that cost line belongs in the cost model when this gets real.

**Run 3b again:**
1. `git pull` (it must show commit `0799af5` or later).
2. Docker Desktop running.
3. `.\scripts\planck3.ps1 round3`

The changed `settings.yml` makes the runner recreate SearXNG, which also clears today's suspensions; one recreate per config change is not a retry loop. G2 v2 collection takes ~75-90 min at 3 s/search. If engines still block, empty searches fall back to Wikipedia per question and each point records which backend answered it.

### CORRECTION + Round 3b attempt 2 (2026-10-07 22:13, commit 9d58062)

**Correction to "Round 3b attempt 1" above.** The "Gemma on our tools with SearXNG: 19.6%" result was **mostly Wikipedia search**:
- Until this fix, a SearXNG run read the **Wikipedia cache first**, so results stored by earlier Wikipedia runs were served silently and counted as SearXNG.
- In that run 131 of 151 result sets were Wikipedia-only; only ~20 questions really hit SearXNG.
- The paired result (26 vs 5 against base chat, p = 0.0002) stands as a **mixed-search** result, not a SearXNG one.
- The "SearXNG beats Wikipedia, 8 vs 0" comparison came from those ~20 real SearXNG questions: suggestive, not established.
- Fixed:
  - The fallback cache is read only after the primary search failed.
  - Every run records which backend **served** each search (cache hits included) as `search_health.served` / `primary_share`.
  - A "searxng" run with `primary_share` < 0.8 is labelled **DEGRADED** and redone (kept as `*_mixed`); G2 points collected the same way are re-collected.

**Attempt 2 results (valid as mixed / mostly-Wikipedia search; training and evaluation saw the same mix):**

| Policy (fresh + long-tail, 143) | Success | Answered | Wrong when answered | ECE |
|---|---|---|---|---|
| Heuristic | 21.0% | 81.8% | 74.4% | 0.47 |
| Gemma on our tools | 19.6% | 29.4% | 33.3% | 0.26 |
| **G2 v2, Planck** | 8.4% | 11.9% | 29.4% | 0.09 |
| G2 v2, hash control | 3.5% | 5.6% | 37.5% | 0.35 |
| Base chat (closed-book) | 4.9% | 98.6% | 95.0% | 0.95 |

**G2 v2: FAIL under rule B.** It misses the seed bar (63.8% vs ≥ 86.1%), the fresh bar (8.4% vs the heuristic − 2 = 19.0%), wrong-when-answered < 5% (29.4%) and ECE < 0.05 (0.094). It does beat the hash control (8.4 vs 3.5; seed 63.8 vs 61.7).

- **Much better than v1** (seed 2.1% → 63.8%). The design diagnosis was right.
- **The choice head is real:** 49.0% vs the deterministic ranker's 36.4% on 143 held-out decision points (hash: 33.6%, below the ranker).
- **The gate is calibrated** (AUC 0.78, ECE 0.05). At the pre-registered 90% precision target it answers ~5% of decision points (precision 1.0 on test).
- **End to end it is a strict subset of the heuristic on fresh** (0 vs 18 discordant, p < 1e-4): never right where the heuristic is wrong. It just declines far more often, trading coverage for reliability (wrong 29% vs 74% when answering).

**What the round says.** On fresh and long-tail questions **retrieval is the ceiling**: the right answer reaches the top-3 snippets only ~19-27% of the time, so every decision policy is capped near there. The learned decision layer is already better than the hand ranker at choosing; the next gain must come from **search**.

## 5. Reading the results

| Question | Where | Good looks like |
|---|---|---|
| Does base chat collapse on fresh facts? | "By regime" for `g0f_gemma_closedbook` | low `fresh` success (A2) |
| Does the product beat base chat where it counts? | "By regime": heuristic vs closed-book | heuristic ≥ closed-book on both regimes (A1) |
| Was 93.6% real? | `g0f_heuristic_snip` success | ≥ 80% (A3) |
| Does Planck learn the decisions? | "G2 heads": learned choice accuracy vs the deterministic ranker (same validation points, where a right candidate exists); `g0f_planck-g2-planck` vs `g0f_heuristic_snip` | learned top-1 above the ranker's; G2 bars in section 3B |
| Do Planck features matter for G2? | `g0f_planck-g2-planck` vs `g0f_planck-g2-hash` | planck > hash |
| Does `rank` hold on new races? | "G1 across seeds (g1_aggregate_t1)" verdict | PASS (section 3C) |

## 6. Troubleshooting (new this round)

| Symptom | Cause | Fix |
|---|---|---|
| `! N/M searches came back empty` during G2 collect | upstream engines rate-limit SearXNG | wait, then re-run `round3` (collection resumes); or collect via Wikipedia: `.\scripts\planck3.ps1 py g2 collect --search wikipedia` |
| `missing scripts/assets/planck3_g2_train_tasks.json` | not pulled | `git pull` (the file is generated on the Mac and committed) |
| Hertz arm skipped | `checkpoints\hertz\best.pt` or `data\hertz12_data\tokenizer.model` missing | copy them; the doctor checks their vocab match |
| G2 eval slow | the `planck` policy encodes candidates with Planck per decision | expected (GPU); CPU-only would be ~10x slower |

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Wikidata gold disagrees with the web | Same noise for every policy; aliases accepted; `gold_as_of` stamped; per-template breakdown in `results.jsonl` (`template` field) |
| Benchmark accidentally tuned | Generator fixed for query failures only; pass rules written before the run |
| SearXNG blocked during 1,400 searches | Resumable collection, empty-search warning, Wikipedia fallback |
| G2 learns the deterministic ranker's biases | It is trained against gold labels, not the ranker; A/B against the ranker is a pass condition |
| Picking `rank` on seeds 0-2 leaks into its confirmation | New task seed (new targets, new races) + new model seeds |
| G2 learns to abstain on everything. On the Mac's thin data only 30% of decision points had a right candidate, and the learned policy answered nothing | It is a calibrated outcome, not a crash. Rule B requires G2 to reach the heuristic − 2 pts, so an abstain-everything policy FAILS visibly. The box's retrieval (SearXNG) and 10x more questions should give it real positives |
