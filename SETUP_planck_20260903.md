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
