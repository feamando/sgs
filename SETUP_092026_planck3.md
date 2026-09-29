# Planck 3.0: a "System One" SGS model for web retrieval (PoC plan)

Nikita Gorshkov · 2026-09-29 · Status: in progress (G0 + G1 pipelines built, awaiting first 4090 run) · Swimlane 1 (Planck) · Roadmap id `1-planck-3-0` · Consumer surface: Satz

**TL;DR.** The bet is not that search is broken (it works, and has for decades). It is that users are moving from search boxes to **chat**, and that a chat answer can be delivered **far more cheaply** than an LLM-with-search product. Planck 3.0 keeps knowledge out of the weights. The model is a small (~100M), frozen-size **policy**. It never writes free text. It emits **typed decisions** from a closed action vocabulary (search, open result k, follow link k, extract span k, answer, abstain), each with a **calibrated probability**. Deterministic tools do the heavy lifting: search, fetch, main-content extraction, answer rendering. Everything it extracts goes into a local, growing **knowledge store**: typed facts plus SGS blobs, with source, timestamp and a per-domain trust prior. That store is what "improves with use". Loop: `query → store lookup → (miss) search → pick result → extract → verify → store → answer card with citations`. This is the Jev idea (typed, probabilistic, fast decisions) plus the Muse idea (a personal agent with memory), built SGS-native, local and read-only. The PoC is gated. **G0** proves the tool harness with a teacher policy, before any Planck training. **G1** distills on offline Wikiracing, where labels are free. **G2** covers three consumer task families. **G3** proves the store compounds. **G4** is the SGS research claim: does an alpha-compositing pointer beat a softmax pointer at candidate selection?

## 0. Why (objective, set 2026-09-29)

- **Customer:** behaviour is shifting from semantic/keyword search to chat. A chat turn gives a **direct answer** instead of ten links, carries **follow-ups** ("and H&M?", "when was it founded?"), and cites its evidence. Search is the tool underneath, not the rival.
- **Commercial:** LLM chat with search pays for a frontier model to read pages on every turn. Planck 3.0 moves the reading into deterministic tools and the choosing into a ~100M local policy, so the marginal cost per answer is search calls plus CPU milliseconds. Target: **≥100× cheaper per correct answer** than an LLM chat on the same tools, at comparable quality.

**What "better than search" means (measured, every G0/G2 run):**

| Comparator | What it is | Metric |
|---|---|---|
| **Search only** | The user reads the top-3 result snippets | `serp_answer_rate_at3`: is the gold value visible in them? (generous to search) |
| **LLM chat** | Teacher (Gemma, or Haiku) driving the same tools, tokens priced at Haiku API rates | success, `usd_per_correct` |
| **Planck 3.0** | The distilled local policy | success, `usd_per_correct`, ms/decision |

Planck 3.0 is "better than search" when its **direct-answer success ≥ the search-only snippet rate**, and "cheap" when its **cost per correct answer ≤ 1% of the LLM-chat comparator**. The LLM comparator only sees compact candidate lists, so real LLM chat products (which read whole pages) cost more: the ratio we report is conservative. Prices live in `config/planck3_prices.json` (Haiku 4.5 $1/$5 per MTok, Anthropic list price).

## 1. Source notes (cleaned from voice memo, 2026-09-29)

- Next Planck step: train a **very small model for a narrow task set**: web access, web search, data extraction, post-processing of extracted data, and answering user questions.
- Stop retraining the model to hold data. That needs lots of data and still leaves a knowledge cutoff. Instead, put a **logical engine** behind the model that pulls data on the fly and answers from web search.
- Search has to be good at **identifying valuable websites and data pieces**, not just returning links.
- Extract on the fly, **store** it, and let that become the model's **knowledge base**. It keeps improving the more the person uses it, **without growing model size or complexity**.
- Target: runs on **pretty much anything** (commodity CPU, low-end devices).
- Inspiration: TypeSafe **Jev** and Meta **Muse**. Build a "micro Muse" on SGS.

## 2. Research: Jev and Muse (what to borrow, what to skip)

**Jev (TypeSafe AI, "System One Models", early access 2026-09-15)** · [source](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- **Does not generate text.** The developer defines the output options; Jev returns **typed values with calibrated probabilities**. It is pitched as "smart if-statements": classify, route, score, extract, branch.
- Up to **255 options per choice**. For larger sets it **scores each option independently**, then makes a second selection.
- All outputs come from **one query**, not token by token. Latency is **70-500ms** vs 3-329s for frontier LLMs. Input price is **$0.042/MTok**, output is free.
- Trained with **RLCD** (RL for Calibrated Decisions): the target is honest probabilities, not preference (RLHF) or verifiable reward (RLVR).
- Demos: a Doom bot at 10 queries/s (~$7/h), and **Wikiracing** (choosing among hundreds to thousands of links per step, finishing in fewer steps than LLMs).
- Not disclosed: size, architecture, weights. It is closed and cloud-only.
- **Borrow:** closed option sets, independent candidate scoring, calibration as a first-class metric, Wikiracing as the benchmark.

**Muse (Meta Superintelligence Labs, launched 2026-09-08)** · [launch](https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/) · [Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) · [Small Business](https://about.fb.com/news/2026/09/introducing-muse-small-business/)
- A personal agent that "doesn't just answer questions, it actually does the work": email, travel, browser forms, negotiation, shopping via Stripe Link single-use cards. It works in the **background**, checks back for approval, and suggests unprompted.
- **Memory:** it remembers one-off details, learns from conversations, and supports "forget X".
- **Infra:** it runs on the **Muse Spark** frontier model (1M context, multi-agent orchestration, computer use). Every user gets a **dedicated cloud VM** with a browser. A separate **Sentinel** agent must approve anything that leaves for the internet. There is a full audit trail. Credentials are stored where the agent can use them but never see them.
- It is the opposite end of the spectrum from Jev: huge model, cloud, side-effecting actions.
- **Borrow:** persistent personal memory, background "watch" tasks, approval-gated actions, audit trail. **Skip for the PoC:** side-effecting actions (purchases, forms, sending) and a frontier-sized model.

**Where Planck 3.0 sits**

| | Jev | Muse | Planck 3.0 (PoC) |
|---|---|---|---|
| Output | Typed decisions + p | Free text + actions | Typed decisions + p, extractive answer cards |
| Model | Undisclosed, cloud | Muse Spark (frontier), cloud | ~100M SGS, **local CPU** |
| Knowledge | In weights | In weights + memory | **In a local store**; weights hold only policy + language |
| Actions | Returns decisions to your code | Browser, email, payments | **Read-only** web; handoff links for "doing" |
| Safety model | Type safety | Sentinel + approvals | Closed action vocab (page text can only become a candidate, never a command) |

## 3. Challenge the requirement (what the SGS history already tells us)

1. **"No training data needed" holds for knowledge, not for the policy.** We still need (a) a pretrained language base, which we have (Planck 1.3, 100M, Wikipedia; Hertz 1.2, 640M), and (b) one-time trajectory data to teach the decisions. What we avoid is **retraining for freshness**. That is the real win, so state it that way.
2. **A 100M model must not write answers.** Planck 1.4 needle benchmark: retrieval found the fact in **40%** of trials, but generation recall was **0%**. The base LM could not use context for QA. So answers are **extractive** (selected spans), rendered through **templates** into cards. Free-text synthesis is an optional escalation to a larger "System Two" model (Gemma 4 E4B locally), outside the PoC gates.
3. **Raum already taught the pattern.** Parametric primitives worked because the model **composes structure** and a **deterministic rasterizer** renders it ("the render is grammar, not the model"). Same here: the model emits an action sequence, and deterministic executors do search/fetch/extract/render. The model's intelligence goes into *choosing*, not *producing*.
4. **The 512-token context is a hard constraint.** Pages will not fit. Use Jev's trick: **encode each candidate independently** (result snippet, link + anchor text, page chunk) and score it against the query/state. The window only ever holds `query + state summary + one candidate`.
5. **The web is adversarial.** DuckDuckGo served a bot wall to our own fetch during this research. Pages carry prompt injection. Mitigation is structural: page text only ever becomes a **scored candidate**, never an instruction, and the action vocabulary has no free-text arguments sourced from pages.

## 4. PoC scope

**In scope (four task families, all read-only; everything is reachable from a chat turn):**
1. **Fact card.** `entity + attribute → value + source + retrieved_at`. Examples: opening hours, current price, release date, address, "who runs X". Output: a card with the value, 1-2 cited spans, a freshness stamp, and a confidence.
2. **Compare table.** `N items × M attributes`, e.g. "3 robot vacuums under €300: price, battery, rating". Built by composing fact-card extractions, so no new capability is needed.
3. **Watch.** A stored fact-card query re-runs on a schedule and notifies when the value changes or crosses a threshold ("tell me when X drops below €Y"). This is the Muse-style background agent, still read-only.
4. **Chat.** Multi-turn conversations: the answer type is inferred from the question and follow-ups are rewritten into standalone questions (`SWAP_ENTITY`: "and H&M?"; `PRONOUN`: "when was it founded?"). Both are deterministic rules now and small classification decisions for the Planck head later. Surfaces: terminal `chat` and a local web chat (`serve`, stdlib only).

**Actionable output without side effects:** deterministic renderers export a card or table as a checklist, shopping list, `.ics` event, or a **handoff deep link** to the page where the user completes the action. Muse does the action itself. We hand off.

**Out of scope for the PoC:** purchases, form filling, sending messages, logins, free-text synthesis as a gated capability, multimodal pages (images/PDF), JS-heavy pages that need a headless browser (flag them as a failure class, don't solve them).

## 5. Architecture

```
user query
  → LOOKUP store (typed facts + blobs, freshness TTL)
      hit & fresh & p ≥ τ → ANSWER card
      miss/stale → SEARCH (SearXNG, local) → K results
  → Planck 3.0 policy: score K candidates independently → OPEN k   (or FOLLOW k on a page)
  → fetch + main-content extract (trafilatura) → chunks
  → EXTRACT: score chunks/spans per schema field → value + span + p
  → VERIFY: 2nd source agrees? (optional step, costs 1 more OPEN)
  → STORE fact + source + domain-prior update
  → ANSWER (template card) | ABSTAIN ("couldn't verify", p < τ)
```

**Action vocabulary v0 (closed, 8 actions):** `LOOKUP · SEARCH · OPEN(k) · FOLLOW(k) · EXTRACT(field, k) · VERIFY · ANSWER(template) · ABSTAIN`. Step cap: 8. The search query in v0 is the user query verbatim, optionally prefixed with `site:<domain>` from the top of the domain prior. There is no generated query rewriting, because that is generation.

**Model (Planck 3.0 = frozen Planck 1.3 encoder + small heads):**
- **Encoder:** Planck 1.3 (100M), frozen or LoRA. It encodes `[query ; state]` and each candidate separately.
- **Action head:** 8-way classifier over the next action.
- **Pointer head:** scores K candidates for OPEN/FOLLOW/EXTRACT. Two variants, compared in G4:
  - *Softmax pointer* (baseline): dot product → softmax.
  - *SGS render pointer:* candidates as Gaussians in `d_s`, the query renders over them front-to-back, and transmittance-weighted opacity gives the selection. **Hypothesis tied to the JMLR theorem (alpha-compositing ⊋ softmax):** transmittance suppresses near-duplicate candidates behind a dominant one (mirror pages, syndicated copies, SEO clones), and softmax cannot do this. That is testable on duplicate-heavy result lists.
- **Calibration:** log-loss training, temperature scaling on held-out data, reported as **ECE** and selective accuracy curves. `ABSTAIN` fires when the answer p < τ. This is our anti-hallucination mechanism, the Jev/RLCD lesson in its cheap form. RL for calibration comes later.

**Knowledge store (reuse `src/blob_store.py` + `src/conversation_memory.py`):**
- **Facts table** (SQLite): `entity, attribute, value, unit, source_url, domain, span, retrieved_at, ttl, p`.
- **Blob index:** each stored fact/chunk is also a DynamicBlobStore blob, for fuzzy LOOKUP. Planck 1.4 retrieval worked at 40% with zero tuning; fix the recency decay first (the 0.05 decay crushed old hits, so use ≤0.01).
- **Domain prior:** per-domain Beta(successes, failures) over "extraction succeeded and was corroborated". It biases OPEN scores and `site:` choice. This is how "gets better at identifying valuable websites" happens **with zero weight updates**: a bandit in the store, not the model.
- **Freshness:** per-attribute TTL (price: 1 day, hours: 7 days, founding date: never). Watch tasks are TTL-triggered re-runs.

**Runtime target:** int8 Planck 3.0 is roughly 100MB. The target is **<100ms per decision on a laptop CPU** (to be measured in G1, not assumed). A typical task is 3-6 decisions, and network I/O dominates total latency.

## 6. Training data (one-time, cheap)

- **Wikiracing (free, perfect labels).** Build the link graph from the Wikipedia dump we already have (Planck 1.3 corpus). BFS gives shortest paths, so every step has a gold `FOLLOW(k)`. Fully offline, no web, no teacher cost. This is also Jev's own demo, so it gives a direct comparison story.
- **Consumer tasks.** Generate ~2k tasks across the three families with Claude Haiku, the same pattern as the 3,450 Raum trees (~$5). Gold answers are verified on a fixed date and snapshotted: we cache every fetched page so the benchmark is reproducible after the web changes.
- **Teacher trajectories.** Run the G0 harness with a teacher policy picking from the **same typed action space**. Log `(state, candidates, action, choice)`, keep **only successful trajectories** (answer matches gold), and distill. Teacher options: local **Gemma 4 E4B** ($0, already on the 4090) or Haiku via Bedrock (low cost, stronger).

## 7. Gates (gate-and-kill; thresholds are proposals, set before running)

| Gate | What | Pass | Kill / fallback |
|---|---|---|---|
| **G0 Harness + teacher ceiling** | Tools + store + typed actions with **Gemma/Haiku as the policy**, no Planck | Teacher ≥60% task success on the consumer set; cached replay deterministic | Teacher <40% → the action space/tools are wrong; fix them before training anything |
| **G1 Wikiracing distill** | Planck 3.0 policy on held-out Wikipedia start/target pairs (held-out targets) | Rollout success (within 2× optimal steps) ≥0.8× the Gemma teacher's; beats head:hash control; <100ms/decision warm on CPU | Fail on 100M → rerun with Hertz 1.2 (640M) encoder as capacity ablation |
| **G2 Consumer end-to-end** | Four task families on cached benchmark (fact, compare, watch, chat) | ≥0.7× the LLM-chat teacher's success **and** ≥ the search-only snippet rate; **cost per correct answer ≤1% of the teacher's**; wrong-answer rate on answered items <5% (abstain instead); ECE <0.05 | Large gap on extraction only → move extraction to deterministic schema parsers (JSON-LD/microdata first), keep the model for choosing |
| **G3 Store compounding** | Replay a second batch of related queries against the warm store | Web calls/task −40% at equal accuracy; domain prior raises OPEN precision@1 | No gain → store/lookup design problem, not the model |
| **G4 SGS pointer (research)** | Render pointer vs softmax pointer on OPEN/EXTRACT, incl. duplicate-heavy lists | Significant gain, **≥3 seeds, held-out, BH-corrected** | Parity → ship softmax; the negative result is still a paper footnote to the theorem |

Discipline carried from the VSP work: reseed before believing any delta, pick λ/τ on held-out data only, and pre-register the thresholds above.

## 8. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Search + fetch + extract tools (SearXNG, httpx, trafilatura, page cache) | New, off-the-shelf libs | S |
| Typed action DSL + validator + executor | New, same pattern as Raum `validate_tree` / `CompositionNode` | S |
| Knowledge store (facts SQLite + blob index + domain prior + TTL) | Reuse `blob_store.py`, `conversation_memory.py`; new facts table + Beta prior | M |
| Teacher harness (Gemma/Haiku as policy) = G0 | Reuse `gemma_decomposer.py` loading + Haiku tree-gen pattern | S |
| Wikiracing graph + BFS labels | New, from existing Wikipedia dump | S |
| Planck 3.0 heads (action, pointer, abstain) + training loop | Reuse Planck 1.3 encoder; new heads | M |
| SGS render pointer (G4) | Reuse `rendering.py` / `kernel.py` compositing | S |
| Calibration + eval harness (success, ECE, abstain precision, latency) | New; reuse `aggregate_disambig_seeds.py` for multi-seed stats | M |
| Consumer UI (chat + answer cards + watch list) | Reuse Satz 0.1 plan + Raum FastAPI/`DecomposerManager` hotswap | M |

## 9. Run instructions (Windows 4090 box, `C:\Users\feama\sgs`)

**One command does everything** (idempotent: re-run after any interruption and it resumes):

```powershell
cd C:\Users\feama\sgs
git pull
powershell -ExecutionPolicy Bypass -File scripts\planck3.ps1 all
```

`all` = `setup → smoke → searxng → g0 → g1 → report → commit+push results` (`-NoPush` to keep results local). It needs nothing beyond what the box already has: `.venv` (created if missing), `checkpoints/planck13/best.pt` + `data/wikipedia/tokenizer.model` (for head:planck), `models/gemma-4-e4b-it` (teacher). Missing pieces degrade gracefully and `setup` prints what to fetch. Docker Desktop is optional: without it, search falls back to the Wikipedia API.

**Individual stages:**

| Command | What it does | Output |
|---|---|---|
| `.\scripts\planck3.ps1 setup` | pip-installs trafilatura/requests/scipy/pytest into `.venv`; checks torch CUDA, checkpoints, Gemma, Docker | console table |
| `.\scripts\planck3.ps1 smoke` | 32 offline tests (fake web, synthetic Wikipedia dump, tiny Planck checkpoint, chat server), ~5s | pytest |
| `.\scripts\planck3.ps1 serve` | **local web chat** at http://127.0.0.1:8010: follow-ups, cited answers, decision trace per reply (`-Policy heuristic` for instant start) | browser |
| `.\scripts\planck3.ps1 chat` | the same in the terminal | console |
| `.\scripts\planck3.ps1 ask "Who founded SpaceX?" -Type entity` | one question → answer card, with the decision trace (`-Policy heuristic` to skip loading Gemma) | console; facts land in `results/planck3/personal_store.sqlite` |
| `.\scripts\planck3.ps1 searxng` | starts the `planck3-searxng` container on `127.0.0.1:8888` (config: `config/searxng/settings.yml`) | Docker |
| `.\scripts\planck3.ps1 g0` | G0: 47-task seed benchmark (37 fact, 4 compare, 6 chat) with the Gemma teacher, then the heuristic baseline; reports success, search-only snippet rate, cost per correct answer | `results/planck3/g0_<policy>/{summary.json, cards.md, results.jsonl, trajectories.jsonl}` |
| `.\scripts\planck3.ps1 g1` | G1: Simple English dump (356 MB) → link graph → BFS tasks → hash + Planck embeddings → heads → eval vs random/lexical/Gemma | `data/planck3/wikirace/`, `results/planck3/g1_head_*_s0/`, `results/planck3/g1_eval_s0/summary.json` |
| `.\scripts\planck3.ps1 g1 -Seed 1` | an extra seed (reseed before believing any head:planck vs head:hash delta) | `..._s1/` |
| `.\scripts\planck3.ps1 report` | one line per summary | console |
| `.\scripts\planck3.ps1 py <args>` | raw pass-through to `scripts/planck3.py` (e.g. `py wikirace eval --limit 50`) | |

**Where things live:** code `src/planck3/`, CLI `scripts/planck3.py`, runner `scripts/planck3.ps1`, tasks `scripts/assets/planck3_tasks.json`, tests `tests/test_planck3.py`. Page/search cache `data/planck3_cache/` makes every G0 re-run replayable (`--net replay`). Trajectories, stores and head weights stay local (gitignored); summaries, cards and results are committed.

**Gate notes from the first build (2026-09-29, Mac, Wikipedia search):**
- **G0 heuristic floor = 51.2%** success (41 tasks), answered 100%, wrong-when-answered 48.8%, ECE 0.285. It never abstains, so it is a floor, not a candidate.
- **`gold_reachable` = 78%:** the right value was among the candidates shown on some opened page in 78% of tasks. That is the toolset's ceiling, above the 60% G0 pass bar, so G0 tests the teacher's *choosing*. When a teacher fails a task whose gold was unreachable, the fix is the tools.
- **Overfitting guard:** the candidate generator got three generic fixes from the first failures (in-page IDF weighting, unit matching, possessive stripping). No further tuning against these 41 tasks. Fresh consumer tasks (page-grounded, snapshot-dated gold) are the next benchmark.
- **G1 gate clarified (pre-registered before any G1 run):** the original "80% of BFS-oracle-guided rate" was ill-defined, because the oracle is 100% by construction. It now reads: **head:planck rollout success ≥ 0.8× the Gemma teacher's, with warm per-decision latency < 100ms on CPU.** Two more rules: head:planck must beat **head:hash** (same head on hashed bag-of-words features) or Planck is adding nothing. Cold CPU latency (encoding every candidate title from scratch) is reported but not gated.
- **Objective reframed (2026-09-29, before any teacher run):** chat instead of search, cheaply (section 0). Two new G0 metrics: `serp_answer_rate_at3` (search-only comparator) and `cost` (usd per task / per correct answer, teacher tokens priced at Haiku rates). A chat family was added (tasks v2, 47 tasks).
- **Mac re-run on v2 (heuristic, Wikipedia search):** success 48.9% vs **search-only snippet rate 55.3%**. On these stable facts, plain search already surfaces the answer more often than the no-model floor extracts it, which is exactly the framing: search works, and the product has to turn it into a direct chat answer. Chat mechanics: answer-type inference **6/6**, follow-up rewrite **6/6** conversations; every chat miss is the heuristic's value choice. The store answered repeat questions with **0 web calls** (ch03, ch04 reused facts from earlier tasks). Heuristic cost ≈ $0.
- **G2-G4:** after G1.
- PowerShell on the box: backtick continuations, not `^`. No `--wandb`.

## 10. Open decisions (recommendation first)

1. **Encoder base:** Planck 1.3 (100M) first. It serves "runs on anything" and the task is choosing, not generating. Hertz 1.2 (640M) only as the G1 fallback ablation.
2. **Search backend:** self-hosted **SearXNG** (free, no key, local). Keep the Brave Search API as a paid fallback if result quality blocks G0.
3. **Teacher:** **Gemma 4 E4B** for bulk trajectories ($0, local), with **Haiku** on a 10% sample to measure the teacher gap.
4. **Name the consumer surface:** fold it into **Satz** (next minor) rather than a new product swimlane. Your call when G1 passes.

## 11. Recommendations

1. **Build G0 before any Planck training.** If a strong teacher can't solve the tasks through our typed actions, a 100M model won't either, and the fix is in the tools.
2. **Make Wikiracing the first training target.** Free perfect labels, fully offline, and a head-to-head story against Jev's own demo.
3. **Keep answers extractive and templated.** Planck 1.4 already proved a 100M base cannot write QA answers. Abstention plus calibration is the product feature, not a limitation.
4. **Put "gets smarter with use" in the store** (domain prior + facts + TTL), measured by G3. Weights stay frozen after distillation.
5. **Run G4 as the one SGS-specific research claim**, with pre-registered thresholds, and link it to the alpha-compositing theorem paper.
6. **Try snippet-first extraction next (proposal, not built).** On the seed set the answer sits in the top-3 snippets 55% of the time, so the cheapest chat answer often needs **zero page fetches**. That means offering the snippets as a first EXTRACT candidate list before any OPEN. It is one more pointer list in the phase table, not a new action. Decide after the first teacher run shows how often the teacher would take it.

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| 100M policy can't read messy page chunks well enough to extract | Deterministic parsers first (JSON-LD, microdata, tables); model only picks among parsed candidates; Hertz ablation in G1 |
| Benchmark rots as the web changes | Snapshot every fetched page; gold answers tied to the snapshot date; cached replay is the eval of record |
| Bot walls / rate limits (seen during this research) | Local SearXNG, polite fetch rate, robots.txt respected, personal-use scope, page cache |
| Prompt injection from pages | Closed action vocab; page text is only ever a scored candidate; no page-sourced free-text arguments |
| Confidently wrong answers | Calibrated p + ABSTAIN below τ; VERIFY against a 2nd source for low-margin facts; wrong-answer rate is a G2 pass criterion |
| Store accumulates stale or wrong facts | Per-attribute TTL; facts carry source + p; contradicted facts demoted; "forget" command |
| Teacher ceiling caps the student | Filter to successful trajectories; Haiku sample measures the gap; RLCD-style calibration RL only after G2 |
| Seed luck masquerading as a result (the VSP +5.7 lesson) | ≥3 seeds, held-out τ/λ, BH correction, pre-registered gates |
