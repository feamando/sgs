# Planck 3.0 round 4: ddgs search, answers that show their work, three-layer source trust

Nikita Gorshkov · 2026-10-08 · Status: ready to run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20260903.md` · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round4` (about 3-3.5 h, idempotent, pushes results, **no Docker needed**). It tests the one lever three rounds point to, retrieval, and ships the product behaviour you asked for:
1. **Search: ddgs** (the `duckduckgo_search` library, renamed). It is the search of record from this round on; SearXNG and Wikipedia search remain as fallbacks. Rule S2 measures it against round 3's pure-Wikipedia runs on the same 143 fresh + long-tail questions: snippet recall @3, then success.
2. **Answers that show their work.** Every answer carries: the direct answer → how it was collated (the steps, in words) → its confidence and what kind it is (calibrated or not) → the weights behind the value and behind the confidence → the trust of every source. Three tiers: **Confident**; **"Low confidence in my results"**, which still shows the best guess, its confidence and weights; and **No answer found**, which shows evidence only. Rules P1-P2 judge whether the tiers tell the truth.
3. **Three-layer source trust.**
   - **System**, common to all users: measured on the 1,455 G2 training questions, not hand-picked.
   - **User**: the "trust more / trust less" buttons, plus what your questions corroborated.
   - **Session**: the "more results from here" button, for this chat only.

   They add in log-odds, so each layer shows as a number. Rule T measures whether the system layer helps.
4. **G2 v2 retrained on ddgs**, so training data matches what deployment sees. It is judged on rule B again, and on the tiers.

The rules below were written **before** the run (2026-10-08).

## 0. Why this round

- **Customer:** round 3 showed the answer is in the top-3 snippets only 19-27% of the time on fresh/long-tail questions. No decision model can pick an answer search never surfaced. And when Planck is unsure, a blank "couldn't verify" wastes what it did find. A clearly labelled best guess with its evidence and weights is more useful, and more honest.
- **Commercial:** ddgs costs $0 and needs no Docker container. If it lifts recall, it is the research backend until a licensed search API is chosen for the product. The explanation and trust layers are what separates "Brain for the web" from base chat: base chat shows no sources, no confidence and no way to say "more from here".

## 1. Your two questions, answered

**1. ddgs, or a browser scraper?** I built **ddgs**. A browser scraper reaches the same engines; what it adds is looking human to bot detection, which is the line we said we would not cross with SearXNG either. Facts to know about ddgs:
- **It scrapes.** DuckDuckGo has no official API, and the library presents itself as a browser to get results. That is fine for a personal research PoC at 1 query per 3 s. It is **not** a product backend: the product needs a licensed search API.
- **Blocks are fast, and they hit one engine at a time.** In the Mac check, about 15 test queries were enough for DuckDuckGo to answer HTTP 202 (its rate-limit page), Brave 429 and Mojeek 403, while Bing and Yahoo kept answering. So the engines are tried in a fixed order: `duckduckgo, bing, brave, yahoo, mojeek`. A blocked engine is skipped; when every engine is empty, the query is retried once, then falls back to Wikipedia.
- **Every run records which backend answered each search.** A run where ddgs served under 80% of searches is **DEGRADED**, set aside once, and redone.
- **Google is excluded.** No CAPTCHA solving, no identity rotation.

**2. Direct answer + traceability + confidence and weights, with clear low-confidence messaging.** Built, in the chat page (`serve`), the terminal chat (`chat`, type `why`) and the benchmark cards:

| Tier | What the user sees |
|---|---|
| **Confident** (answered, p ≥ 0.5) | **the value**, the sentence it came from, the source and who agrees, a confidence bar with the bar line drawn on it, the kind of confidence ("calibrated on held-out questions" for G2 v2; "rule-based score, not calibrated" for the heuristic; "the model's own estimate" for an LLM). |
| **Low confidence in my results** (an answer below 0.5, or a best guess the policy would not commit to) | the same, labelled up front, with "How I got this" open: the best guess, its confidence vs the bar, the runner-up, why the confidence is low. |
| **No answer found** | the evidence pack, open. No value is invented: an unchecked top candidate is shown only if its lexical score is ≥ 0.4. |

"How I got this" contains:
- **Steps, in words:** "Searched (ddgs) for … : 7 results from 7 sites" → "Read 29 candidate values off the snippets" → "Picked Tadej Pogačar from procyclinguk.com" → "Cross-checked: worldinsport.com gives the same value" → "Answered: 81%, above my bar of 50%".
- **The candidates it chose between,** with their probabilities (G2 v2) or scores (heuristic).
- **Why this value:** the additive score terms. Each term is a weight that pushes this candidate up or down: uses the question's words, sits next to them, the source is about the subject, other sources agree, the unit matches, the value repeats the subject.
- **Why this confidence:**
  - For G2 v2: each gate feature's push in log-odds (exact, because the gate is a logistic regression), plus the baseline.
  - For the heuristic: lexical match, lead over the runner-up, and whether a second source agreed.
- **Sources:** each with its system / you / this chat / combined trust and the three buttons.

**2a. Three-layer source confidence.**

| Layer | Who it is for | Where it comes from | Where it lives |
|---|---|---|---|
| **System** | every user | `planck3.py trust build`: on the G2 training questions where *some* snippet carried the right answer, did *this domain's* snippet carry it? Beta counts, capped at 20 observations so a user can outvote it | `config/planck3/source_trust.json`, in git |
| **User** | you | "trust more" / "trust less" (weight 2 each) + implicit (a value from it was corroborated: +1; a page with nothing usable: -0.5) | your private store (`personal_store.sqlite`) |
| **Session** | this chat | "more results from here": +1.0 log-odds per click (max +2), runs `site:<domain> <question>` now and on later questions in the chat | memory; gone on "New chat" |

`logit(trust) = logit(system) + user_shift + session_shift`. Trust decides what gets read and verified first, how depth passages rank, and what "For you" explores. It does **not** move the answer value yet; rule T measures what it does before it is allowed to.

## 2. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke + doctor | installs `ddgs` + `scikit-learn`; 63 offline tests; doctor checks ddgs answers | `doctor.json` | ~3 min |
| ddgs check | must answer; re-checks every 2 min for 10 min, then **stops** (round 4 measures ddgs, not the fallback) | | <1-10 min |
| **S2 / A1 / P** | fresh benchmark on ddgs, **system trust off** (the search change alone): heuristic, Gemma on tools. Gemma closed-book is reused from round 3 (no search involved) | `g0f_heuristic_snip_ddgs`, `g0f_gemma_snip_ddgs` | ~35 min |
| **G2 collect** | 1,455 training questions on ddgs (resumable) | `data/planck3/g2/points_ddgs.jsonl` (+ `.health.json`, pushed) | ~2-2.5 h |
| **system trust** | built from those points | `config/planck3/source_trust.json` (pushed) | <1 min |
| **G2 v2 train + eval** | hash + Planck heads on ddgs points; eval on the fresh and the 47-question seed benchmark | `g2v2_head_*_s0_ddgs`, `g0f_planck-g2v2-*_ddgs`, `g0_planck-g2v2-*_ddgs` | ~20 min |
| **T** | the heuristic again, **with** system trust | `g0f_heuristic_snip_ddgs_trust` | ~5 min (cached) |
| report + push | REPORT.md gains "Answer tiers" and "Round 4 paired comparisons" | git | 1 min |

## 3. Step by step

1. `git pull` (≥ the commit that adds this file).
2. **No Docker needed.** You can leave Docker Desktop off.
3. Shakedown (~20 min: 12 benchmark questions, 100 training questions, `_quick` outputs):
   ```powershell
   .\scripts\planck3.ps1 round4 -Quick -NoPush
   ```
   Check `REPORT.md`: there should be `g0f_*_ddgs_quick` rows, an "Answer tiers" table, and a `ROUND 4 (rules S/P)` verdict, not DEGRADED.
4. The real run:
   ```powershell
   .\scripts\planck3.ps1 round4
   ```
   If ddgs gets rate-limited partway, the run keeps going on the per-query Wikipedia fallback and records it. A run below 80% ddgs is set aside once and redone on the next `round4`. If it stops, run it again: finished work is kept and G2 collection resumes.
5. Optional, to try the product: `.\scripts\planck3.ps1 serve`, then open http://127.0.0.1:8010. Ask something, open "How I got this", click "trust more" or "more from here" on a source.

### Brave Search API key (added 2026-10-08, optional)

The key lives in `.env` at the repo root. `.env` is gitignored, because the repo is public; `.env.example` is the committed template. `planck3.py` and `planck3.ps1` both read `.env`, and a variable already set in the environment wins. On the box, once:

```powershell
.\scripts\planck3.ps1 setkey        # paste the key at the hidden prompt; it writes .env, then checks the API answers
```

Brave is **opt-in** (`--search brave`, e.g. `.\scripts\planck3.ps1 ask "..." --search brave`). Round 4's pre-registered runs stay on ddgs, so the rules below are unchanged. If S2 passes, the next step is the same paired comparison with Brave against ddgs, before Brave is used anywhere user-facing. `config/planck3_prices.json` still carries a placeholder Brave price: check your plan's price before reading any cost line.

## 4. Pre-registered rules (2026-10-08, before the run)

All rules are judged on the 143-question fresh + long-tail benchmark, paired by question (exact McNemar), and only on **valid** ddgs runs (≤ 20% empty, ddgs served ≥ 80%).

**S. Search is the lever.**
- **S2 (primary):** heuristic on ddgs vs the same heuristic on pure Wikipedia search (`g0f_heuristic_snip_wikipedia`, round 3 attempt 2). **PASS if snippet recall @3 is higher with p < 0.05.** The baseline is 18.9%. Success is reported paired too (baseline 15.4%).
- **A1 (repeat):** Gemma on tools (ddgs) beats Gemma closed-book (4.9%) with p < 0.05.

**P. The tiers tell the truth.** Judged for the heuristic and for G2 v2 (Planck).
- **P1:** precision of the **Confident** tier ≥ 0.80.
- **P2:** Confident precision − Low-confidence precision ≥ 0.20. Otherwise the label carries no information and must change.
- Reported, no bar: each tier's share, and for **No answer found**, how often its evidence pack contains the answer (that is the "retrieval in depth" promise).

**B. G2 v2 on ddgs.** The round-3b bars, unchanged (seed ≥ 86.1%; fresh ≥ closed-book and ≥ heuristic − 2 pts, here the `_trust` heuristic, which reads in the same trust order; wrong-when-answered < 5%; ECE < 0.05; beats the hash control). Reported alongside: v2's tiers vs the heuristic's. A calibrated gate should win P1/P2 even if it loses B.

**T. System trust.** `_ddgs_trust` vs `_ddgs`, paired success. Reported, no bar. The layer stays on in the product only if it does not lose (p < 0.05 the wrong way turns it off by default).

## 5. What the Mac check found (disclosed fixes, before the run)

1. **ddgs blocks fast, one engine at a time** (see section 1). The engine chain and fallback come from this.
2. **Bing-style snippets open with a publish stamp** ("Sep 13, 2026 · Who won …"). The heuristic picked "Sep" as the winner of a 2026 final. Two fixes:
   - The stamp is now split into its own sentence, so its month and year are not read as part of the answer sentence.
   - Month abbreviations are rejected as entity answers (full month names already were).

   This changes extraction for every backend; it was fixed before any round-4 number exists.
3. **The first version of the system trust measure was biased.** "Was the domain's top-ranked candidate right" punished long pages with many numbers (Wikipedia came out at 0.33). It now uses snippet points only and asks whether the domain carried the right answer when some snippet did.
4. `infer_answer_type` now reads "elevation / altitude" as a number question in chat.
5. Live Mac mini-run (6 questions, heuristic, ddgs): ddgs served 8 of 9 searches (1 Wikipedia fallback), snippet recall @3 5/6, tiers populated. That sample is far too small to read; it is a shakedown only.

## 6. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| ddgs backend | reuse the search cache, retry, fallback, circuit breaker, health; new `_search_ddgs`, `search_site`, `ddgs_alive` | S |
| Answer tiers + explanation | reuse trajectories, decision meta, candidate scores (now kept per term); new `explain` record, `render_why`, chat page | M |
| Gate weights | reuse the logistic gate; contributions = coef × standardized feature | XS |
| Three-layer trust | reuse the per-domain Beta store; new `trust.py` (system table + log-odds layers), session boost on the chat harness, `/api/more` | M |
| Runner | reuse `planck3.ps1`; new `round4`, `Test-Round4Run` (degraded → aside once → redo) | S |
| Tests | 8 new offline tests (63 total) | S |

## 7. Recommendations

1. Run `round4 -Quick -NoPush`, then `round4`.
2. If **S2 passes**, ddgs is the research backend. Price a licensed API (Brave Search API or similar; verify current prices) for anything user-facing, and run it through the same S2 rule before switching.
3. If **P1/P2 pass for G2 v2**, the product policy is "v2 decides, the card shows the tier". Retune τ for coverage only with a new pre-registration.
4. If **S2 fails** (ddgs no better than Wikipedia), retrieval is not fixed by changing scrapers. The next step is query rewriting (the question → 2-3 search queries), which is a System One decision Planck can learn.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ddgs rate-limited during the 1,455-question collect | engine chain + per-query Wikipedia fallback + health recorded per run; DEGRADED runs set aside and redone; collect is resumable |
| Scraping terms of service | personal research volume (1 query / 3 s), no identity rotation or CAPTCHA solving, google excluded; not a product backend (section 7.2) |
| Low-confidence guesses read as answers | label first ("Low confidence in my results"), bar drawn on the meter, no guess below lexical score 0.4; P2 checks that the label carries information |
| The system trust layer shifts the round-3 comparison | S2 runs with system trust **off**; T measures the layer separately |
| Trust table overfits to Wikipedia-heavy training questions | capped at 20 observations, min 3; users outvote it; measured on disjoint questions |

## 9. Results of record (round 4, 2026-10-08/09, e28d6aa)

All valid ddgs runs (ddgs served 85-100% of searches, 0-0.7% empty). G2 collect: 1,455 searches, ddgs 86%, Wikipedia fallback 14%.

| Run (143 fresh + long-tail) | Success | Snippet recall @3 | Confident: share / precision | Low confidence: share / precision | Value shown is right | ECE |
|---|---|---|---|---|---|---|
| Base chat (Gemma closed-book, round 3) | 4.9% | n/a | n/a | n/a | n/a | n/a |
| Heuristic, Wikipedia search (round 3) | 15.4% | 18.9% | n/a | n/a | n/a | n/a |
| **Heuristic, ddgs** | **51.0%** | **73.4%** | 0.909 / 0.562 | 0.084 / 0.000 | 51.0% | 0.319 |
| Heuristic, ddgs + system trust | 56.6% | 84.6% | 0.979 / 0.571 | 0.021 / 0.333 | 56.6% | 0.297 |
| **Gemma on tools, ddgs** | **60.1%** (fresh 56.7%) | 84.6% | 0.797 / 0.754 | 0.203 / 0.379 | 67.8% | 0.177 |
| **G2 v2 Planck, ddgs-trained** | 29.4% (fresh 0/60) | 85.3% | 0.322 / **0.913** | 0.678 / 0.495 | 62.9% | **0.022** |
| G2 v2 hash control | 24.5% | 85.3% | 0.294 / 0.833 | 0.706 / 0.287 | 42.0% | 0.102 |

**Verdicts (section 4 rules):**
- **S2 PASS, by a wide margin.**
  - Heuristic snippet recall @3: 18.9% → 73.4% (78 vs 0 discordant, p = 7e-24). Success: 15.4% → 51.0% (53 vs 2, p = 9e-14).
  - Gemma on tools: 14.0% → 60.1% (70 vs 4, p = 1e-16).
  - Retrieval was the ceiling, as round 3 said.
  - Conservative: 21 of the heuristic's 143 questions still ran on the Wikipedia fallback.
- **A1 PASS:** Gemma on tools 60.1% vs base chat 4.9% (83 vs 4, p = 3e-20). On 2026 facts: 56.7% vs 0%.
- **P1 (confident precision ≥ 0.80):**
  - **PASS for G2 v2** (0.913; hash 0.833).
  - **FAIL for the heuristic** (0.562): its rule-based score is uncalibrated (ECE 0.32), and it labels 91-98% of answers confident.
- **P2 (confident − low ≥ 0.20):**
  - **PASS for G2 v2** (0.913 vs 0.495: +0.42).
  - The heuristic passes formally (+0.56), but its low tier is 3-12 questions, so the label rarely appears.
- **B FAIL** for G2 v2 on ddgs:
  - Seed benchmark 29.8% (bar 86.1%).
  - Fresh 29.4% vs heuristic 56.6% (4 vs 43 discordant).
  - Wrong-when-answered 8.7% (bar 5%); ECE 0.022 passes.
  - vs hash: 12 vs 5, p = 0.14 (n.s.).
- **T, reported:** the raw +5.6 pts (9 vs 1, p = 0.02) is a **search artifact**. The no-trust run hit ddgs first and got the Wikipedia fallback on 21 questions; the trust run came later and got none. On the 120 questions where both runs got web results: **0 vs 0**, no effect. Per the rule, the layer stays on, since it does not lose.

**What it means:**
1. **The product case holds on the questions that matter.** On fresh + long-tail questions, tools + ddgs reach 51-60% against base chat's 4.9%, with sources, and the no-model path costs ~$0. Gemma driving the tools costs 3,635 LLM tokens/task ($0.0065 per correct at Haiku prices).
2. **G2 v2 is the honest one, not the accurate one.**
   - It is the only policy whose labels tell the truth: confident is 91% right and ECE is 0.02.
   - Its *displayed* value (any tier) is right 62.9% of the time, between the heuristic (56.6%) and Gemma (67.8%), neither difference significant (p = 0.22, 0.35). This metric was reported, not a pre-registered rule.
   - But it commits too rarely. On 2026 facts it never clears τ = 0.81, although its best guess is right 24/60 times (more than the heuristic's 20/60). The training questions are past events, so the gate is under-confident on the present.
3. **The fix is new training data, not tuning on this benchmark:** G2 training questions about recent events (2025-2026, disjoint from the benchmark), so the gate sees the regime it will be used in.
4. **Caveats:**
   - **The benchmark is built from Wikidata.** The domains ddgs surfaces most (Wikipedia, grokipedia, wikiwand, wikidata.org) mirror it. A non-Wikidata benchmark (news questions with human-checked answers) is the next generalization test.
   - **The system trust table reflects the encyclopedic training mix:** bbc.com scores 0.26 and en.wikipedia.org 0.85. That is wrong for news questions. Trust needs to be per question type before it moves anything.
