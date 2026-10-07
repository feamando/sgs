# Planck 3.0: a "System One" SGS model for web retrieval (PoC plan)

Nikita Gorshkov · 2026-09-29 · Status: in progress (G0 + G1 pipelines built, awaiting first 4090 run) · Swimlane 1 (Planck) · Roadmap id `1-planck-3-0` · Consumer surface: Satz

**TL;DR.** Brain-style retrieval for the open web: a **direct answer plus retrieval in depth**, delivered in chat, **far cheaper than base Claude/ChatGPT**. It comes from a local knowledge graph that becomes **more relevant to its user with every question**: what they searched and what is adjacent to it, from the sources they trust, instead of ads. Search works and is the tool underneath, not the rival. Planck 3.0 keeps knowledge out of the weights. The model is a small (~100M), frozen-size **policy**. It never writes free text. It emits **typed decisions** from a closed action vocabulary (search, open result k, follow link k, extract span k, answer, abstain), each with a **calibrated probability**. Deterministic tools do the heavy lifting: search, fetch, main-content extraction, answer rendering. Everything it extracts goes into a local, growing **knowledge store**: typed facts plus SGS blobs, with source, timestamp and a per-domain trust prior. That store is what "improves with use". Loop: `query → store lookup → (miss) search → pick result → extract → verify → store → answer card with citations`. This is the Jev idea (typed, probabilistic, fast decisions) plus the Muse idea (a personal agent with memory), built SGS-native, local and read-only. The PoC is gated. **G0** proves the tool harness with a teacher policy, before any Planck training. **G1** distills on offline Wikiracing, where labels are free. **G2** covers three consumer task families. **G3** proves the store compounds. **G4** is the SGS research claim: does an alpha-compositing pointer beat a softmax pointer at candidate selection?

## 0. Why (objective, set 2026-09-29)

**Product in one line:** Brain-style retrieval for the open web. A direct answer plus retrieval in depth, from a local knowledge graph that grows with use and is faster, cheaper and more generic than asking base Claude/ChatGPT.

- **Customer:** behaviour is shifting from search boxes to **chat**. A chat turn gives:
  - a **direct answer** instead of ten links;
  - **follow-ups** that carry context ("and H&M?", "when was it founded?");
  - **retrieval in depth** behind the answer: ranked evidence passages, the sources read, further reading, and related facts already in memory. That is Brain's `query_knowledge` shape (ranked hits, then drill-down), applied to the web.
- **Commercial:** the rival is **base Claude/ChatGPT**, not search. Those pay a frontier model per turn, answer from weights with a cutoff, and rarely show their evidence. Planck 3.0 moves the reading into deterministic tools and the choosing into a ~100M local policy. The marginal cost per answer is search calls plus CPU milliseconds. Target: **≥100× cheaper per correct answer at comparable quality, with evidence attached.**
- **It gets more relevant with use, instead of ads.** The local knowledge graph keeps every passage read, the entity co-mention graph, what the user asks about (recency-weighted) and which sources they trust (earned or thumbs-up). So it can keep retrieving what they searched and what is adjacent to it, **from the sources they trust**, as a private "For you" digest. An ad-funded feed ranks what someone paid to show. This one ranks by the user's own interests and trust. Nothing leaves the machine except search queries and page fetches.

**Comparators (measured on every G0/G2 run):**

| Comparator | What it is | Role | Metric |
|---|---|---|---|
| **Base chat** | Gemma (or Haiku) answering from its weights, no tools, raw conversation for chat turns (`g0 --closed-book`) | **the rival** | success, `usd_per_correct` |
| **LLM chat on our tools** | the same LLM as the policy over our search/extract tools, tokens priced at Haiku API rates | teacher and cost ceiling | success, `usd_per_correct` |
| **Search only** | the user reads the top-3 result snippets | quality floor, not the rival | `serp_answer_rate_at3` |
| **Planck 3.0** | the distilled local policy | the product | success, `depth_evidence_recall`, `usd_per_correct`, ms/decision |

Planck 3.0 is good enough when:
- it **matches base chat on quality**: direct-answer success ≥0.9× closed-book, with a higher expected margin on fresh facts where base chat hits its cutoff;
- it **beats it on evidence**: `depth_evidence_recall`, the share of answers whose depth pack contains the correct value;
- it costs **≤1% per correct answer**.

Search-only is the floor it must clear. Both LLM comparators are priced at Haiku 4.5 API list rates ($1/$5 per MTok, `config/planck3_prices.json`). Real chat products use bigger models and read whole pages, so the reported ratio is conservative.

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
5. **For you (digest).** From the local graph only: the user's recency-weighted interests, fresh evidence about them, **adjacent** entities (co-mention edges), and stale facts due for a refresh, all restricted to **trusted domains**. `digest --explore` is the continuous-retrieval loop. It searches adjacent entities, keeps only trusted-domain results, and reads them into the graph (read-only, runs on a schedule). Thumbs up/down on a source moves its trust (weight 2 vs 1 for implicit signals).

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
| **G2 Consumer end-to-end** | Task families on cached benchmark (fact, compare, watch, chat) | ≥0.9× **base chat (closed-book)** success and ≥ the search-only snippet rate; `depth_evidence_recall` ≥0.8; **cost per correct answer ≤1% of base chat's**; wrong-answer rate on answered items <5% (abstain instead); ECE <0.05 | Large gap on extraction only → move extraction to deterministic schema parsers (JSON-LD/microdata first), keep the model for choosing |
| **G3 Graph compounding + relevance** | Replay a second batch of related queries against the warm graph; a week of real use with thumbs up/down | Web calls/task −40% at equal accuracy; domain prior raises OPEN precision@1; repeat questions answered from memory **with their evidence** at 0 web calls; digest thumbs-up rate rises week over week | No gain → graph/lookup design problem, not the model |
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

**Step-by-step guide for the box: `SETUP_planck_20260901.md`** (doctor -Deep → all -Quick -NoPush → all → serve; troubleshooting table; how to read the verdicts).

**One command does everything** (idempotent: re-run after any interruption and it resumes):

```powershell
cd C:\Users\feama\sgs
git pull
powershell -ExecutionPolicy Bypass -File scripts\planck3.ps1 all
```

`all` = `setup → smoke → doctor -Deep → searxng → g0 → g1 → report → commit+push results` (`-NoPush` to keep results local). It logs to `results/planck3/logs/` and pushes summaries, `REPORT.md`, logs and compressed trajectories, so results can be read from any machine. It needs nothing beyond what the box already has: `.venv` (created if missing), `checkpoints/planck13/best.pt` + `data/wikipedia/tokenizer.model` (for head:planck), `models/gemma-4-e4b-it` (teacher). Missing pieces degrade gracefully and `setup` prints what to fetch. Docker Desktop is optional: without it, search falls back to the Wikipedia API.

**Individual stages:**

| Command | What it does | Output |
|---|---|---|
| `.\scripts\planck3.ps1 doctor -Deep` | preflight: packages, CUDA/VRAM, disk, git, network, Planck ckpt vs tokenizer vocab, **Gemma emits valid typed decisions**, measured Planck embed speed, **ETA per stage** | console; `results/planck3/doctor.json` |
| `.\scripts\planck3.ps1 all -Quick -NoPush` | shakedown of every stage at small size into `*_quick` outputs (does not block the full run) | `results/planck3/*_quick/`, `REPORT.md` |
| `.\scripts\planck3.ps1 setup` | pip-installs trafilatura/requests/scipy/pytest/sentencepiece into `.venv` | pip |
| `.\scripts\planck3.ps1 schedule -At 08:00` | daily `digest -Explore` via Task Scheduler (`unschedule` removes it) | `results/planck3/digest.md` |
| `.\scripts\planck3.ps1 smoke` | 47 offline tests (fake web, synthetic Wikipedia dump, tiny Planck checkpoint, chat server), ~5s | pytest |
| `.\scripts\planck3.ps1 serve` | **local web chat** at http://127.0.0.1:8010: follow-ups, cited answers, decision trace per reply (`-Policy heuristic` for instant start) | browser |
| `.\scripts\planck3.ps1 chat` | the same in the terminal (`more` shows the evidence behind the last answer) | console |
| `.\scripts\planck3.ps1 digest` | **"For you"** from your local knowledge graph: interests, adjacent entities, stale facts, trusted sources only (`-Explore` also reads adjacent entities from trusted sources) | console; also the "For you" button in `serve` |
| `.\scripts\planck3.ps1 py feedback source en.wikipedia.org up` | thumbs up/down a source (or `entity`); also "trust more/less" under each passage in `serve` | store |
| `.\scripts\planck3.ps1 ask "Who founded SpaceX?" -Type entity` | one question → answer card, with the decision trace (`-Policy heuristic` to skip loading Gemma) | console; facts land in `results/planck3/personal_store.sqlite` |
| `.\scripts\planck3.ps1 searxng` | starts the `planck3-searxng` container on `127.0.0.1:8888` (config: `config/searxng/settings.yml`) | Docker |
| `.\scripts\planck3.ps1 g0` | G0: 47-task seed benchmark (37 fact, 4 compare, 6 chat): Gemma teacher on our tools, then **Gemma closed-book (the base-chat rival)**, then the heuristic floor; reports success, depth evidence recall, search-only rate, cost per correct answer | `results/planck3/g0_gemma/`, `g0_gemma_closedbook/`, `g0_heuristic/` | `results/planck3/g0_<policy>/{summary.json, cards.md, results.jsonl, trajectories.jsonl}` |
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
- **Rival clarified (2026-09-29):** cost is vs **base Claude/ChatGPT**, and the product is **direct answer + retrieval in depth** ("Brain for the web"). Added the closed-book base-chat comparator (`g0 --closed-book`) and the depth pack on every answer: ranked passages across everything read + snippets, mirror dedup, sources, related memory, no LLM and no extra fetches. Added `depth_evidence_recall`.
- **Graph grows with use (2026-09-29):** the store now keeps passages, a co-mention entity graph, the interest log and user feedback. `digest` / "For you" serves interests + adjacent entities from trusted sources only; `--explore` is the continuous-retrieval loop.
- **Mac v3 (heuristic, Wikipedia search, 47 tasks):** direct answer 48.9% · search-only 55.3% · **depth evidence recall 78.7%**. The evidence pack holds the right value far more often than the one-line answer is right, so the depth layer already carries value before any model is trained. One benchmark pass grew the graph to **258 passages / 477 entities**. A repeat question in the web chat came back from memory with 6 evidence passages and 0 web calls.
- **FIRST FULL RUN on the 4090 (2026-10-05, commit aec21e6, 19 min, SearXNG): results of record.**
  - **G0 (47 tasks):**
    - Gemma on our tools 68.1% (verdict PASS, 5,486 tok/task, $8.74 per 1k correct at Haiku prices)
    - base chat (Gemma closed-book) 95.7% (82 tok/task, $0.11 per 1k correct)
    - heuristic floor 61.7%
    - search-only @3 **95.7%** (Wikipedia search on the Mac: 55.3%)
    - depth evidence recall 1.000 (snippets are in the pack, so this tracks search-only; generous metric)
    - `gold_reachable` 80.9%
  - **G0 errors:**
    - The teacher misses are reading errors: "Kanbarra", subject echoes ("One Hundred Years of Solitude" for its author), later product variants.
    - Two wrong answers came with p=0.00, so an ANSWER-below-τ gate would turn them into abstains.
    - Base chat's 2 misses are confident and unsourced: Zalando HQ "Dusseldorf" (it is Berlin) and Spotify 2008 (launch vs founding).
  - **G1 (1,192 held-out races):**
    - head:planck **21.1%**, lexical 15.3%, head:hash 14.6%, random 0.1%, Gemma teacher 30.0% (first 100 pairs)
    - Planck features add **+6.5 pts** over the identical head on hash features, and at full data the Planck head now beats the untrained lexical baseline (it did not on the quick set).
    - Speed: 0.43 ms warm vs Gemma 223 ms (~500x); cold CPU median 94 ms / p90 536 ms.
    - **Pre-registered verdict: FAIL** (head/teacher 0.70 < 0.8).
    - Caveats: one seed, and the teacher ran on a 100-pair subset, so the ratio is unpaired. The eval does not yet save per-pair outcomes.
- **What the run says to do next:**
  1. **Snippet-first extraction.** With open-web search the answer is in the top-3 snippets 95.7% of the time, while reading full pages is where the teacher's errors come from. It is cheaper (0 fetches) and likely more accurate.
  2. **A fresh + long-tail benchmark.** On stable facts base chat is 95.7%, so the rival must be tested where its memory ends.
  3. **G1 in order:** a paired eval on the teacher subset + 2 more seeds, then a training-objective fix (distance regression on the full BFS labels / DAgger), then the Hertz encoder ablation (the plan's FAIL branch).
  4. **An ANSWER-below-τ abstain gate** in the harness.
- **Follow-up build (2026-10-07), PRE-REGISTERED before the next box run:**
  - **Snippet-first extraction (built):**
    - After SEARCH, typed candidates are read off the result snippets (`results_snip` phase: EXTRACT from snippets, or OPEN a page).
    - **Aboutness** weighs a snippet by whether its result title is about the question's subject (overlap over union on the title core), so "Netflix Animation" does not answer "when was Netflix founded".
    - VERIFY is free when another domain's snippet carries the same value.
    - G0 now also reads one page **after** answering for the depth pack (product setting: answer fast, then depth), and reports answer-path fetches separately.
    - **Adoption rule:** snippet-first becomes the default if `g0_gemma_snip` success ≥ `g0_gemma` (pages-first, 68.1%) minus 2 pts **and** answer-path fetches drop ≥50%.
    - Mac read (heuristic, Wikipedia API snippets, the worst case): 40.4% vs 48.9% pages-first at 0.15 vs ~1.2 answer fetches/question; depth evidence 78.7% with one page after. Not tuned further on Mac snippets; the box (SearXNG, where the answer is in the top-3 snippets 95.7% of the time) decides.
  - **Confidence gate (built):**
    - ANSWER with p < τ = 0.3 becomes ABSTAIN (`low_confidence`); the value is kept for analysis (`gated_value`, `gated_would_be_correct`).
    - Replayed over the first box run: blocks exactly Gemma's 2 confidence-0.00 answers (both wrong) and no right one, so wrong-when-answered goes 25.6% → 22.0% at equal success.
    - Gemma's p is near-binary, so τ in 0.3-0.7 is equivalent. Its 9 remaining wrong answers sit at p 0.9-0.95, which is a calibration problem for distillation, not one for the gate.
  - **G1 paired + seeds (built):**
    - The eval saves per-race outcomes. The teacher runs **once** on the first **300** test races (was 100) and is cached.
    - Every head is compared to it on the **same races** (exact McNemar), plus head-vs-hash and head-vs-lexical.
    - The new task format (per-candidate BFS distances) reproduces the first run's pairs and steps exactly (verified on the full graph).
    - **Primary rule (unchanged bar, now paired):** G1 PASS if head:planck's paired ratio to the teacher, **averaged over seeds 0, 1, 2**, is ≥0.8 **and** head:planck beats head:hash in **every** seed.
  - **Secondary arms (exploratory, declared now):** `head:planck-rank` (listwise soft targets over every candidate's BFS distance), and with `-Hertz` `head:hertz` / `head:hertz-rank` (Hertz 1.2, 640M). Same rule, reported separately; with 1-3 secondary arms, a lone pass near p≈0.05 needs a confirmation run before it counts.
  - Mac signal on the control features (1 seed): `head:hash-rank` 17.3% vs `head:hash` 14.8% on the same 1,192 races (106 vs 77 discordant, McNemar p=0.038), the first hash head above lexical (15.3%). A reason to expect the Planck rank arm to help, not a result.
- **FOLLOW-UP RUN (2026-10-07, commit 2f175cb, SearXNG): results of record.**
  - **G0, snippet-first: ADOPTED** (pre-registered rule passed: success ≥ pages-first −2 pts and answer fetches −50%).
    - **Gemma:** 85.1% vs 68.1% pages-first. Answer-path fetches 0.00 vs 2.81. Wrong-when-answered 9.1% vs 25.6%. ECE 0.039 vs 0.137. 4,206 vs 5,486 tok/task. 89% of answers taken straight from snippets.
    - **No-model heuristic:** **93.6%** vs 61.7% pages-first, at 0.02 answer fetches.
    - **That is 0.98× base chat** (95.7%) at ~$0 LLM cost, with sources, and it was right on **both** of base chat's confident hallucinations (Zalando HQ, Spotify founding year).
    - The gate did not fire: Gemma's low-confidence cases were explicit ABSTAINs.
  - **G0 reading:**
    - On stable facts, snippet-first + deterministic tools (aboutness, cross-domain agreement, echo penalty) already reach the G2 quality bar (≥0.9× closed-book) without any model.
    - The Gemma teacher is now **worse than the heuristic** (85.1 vs 93.6). Its misses are subject echoes ("Brazil", "Spotify") and abstains. Distilling Gemma would teach Planck those errors.
    - **Caveat:** the candidate generator was developed on these 47 tasks, so the 93.6% must be confirmed on fresh tasks before it counts.
  - **G1, 3 seeds × 1,192 races, teacher 33.7% on 300 shared races: FAIL under the pre-registered primary rule.**
    - **head:planck (nll):** mean paired ratio **0.58** (0.52 / 0.59 / 0.63); beats head:hash in every seed (p ≤ 1e-6).
    - **Secondary arm head:planck-rank:** **27.3% ± 0.4** rollout success (CI 0.264-0.282); paired ratio **0.78** (0.79 / 0.84 / 0.71), just under the bar.
    - Rank beats nll in every seed (McNemar p = 1.7e-5 / 0.038 / 0.0026) and is 4.5x more stable across seeds (sd 0.004 vs 0.018).
    - The teacher is still significantly better in 2 of 3 seeds (p = 0.044 / 0.117 / 0.004).
    - The objective, not capacity, was the main lever so far (0.58 → 0.78 with the same encoder).
    - Speed: 0.45 ms warm vs teacher 226-238 ms (~500x); cold CPU median 95 ms.
- **Round 3 (2026-10-07): built, pre-registered, awaiting the box.** Guide + rules: `SETUP_planck_20260903.md`.
  - A **fresh + long-tail benchmark**: 143 questions from Wikidata (60 fresh 2026 facts, 83 long-tail with ≤3 Wikipedia editions), dated, sourced, not tuned against any policy.
  - **G2 redesigned:** learn the decisions from known answers. 1,455 disjoint Wikidata training questions → labelled decision points → a candidate scorer with a none-of-these option → the `planck` policy.
  - The **G1 confirmation:** task seed 1, model seeds 3-5, primary `head:planck-rank`, plus the Hertz arm.
  - One command: `.\scripts\planck3.ps1 round3`.
- **Round 3, attempt 1 (2026-10-07, 343e0e2): partly invalid** (SearXNG blocked mid-run; details in `SETUP_planck_20260903.md`).
  - **Results of record (no web search involved):**
    - **Base chat collapses on the fresh + long-tail benchmark:** 4.9% (fresh 0.0%, long-tail 8.4%), answering 98.6% and wrong 95% of the time (rule A2 holds).
    - **G1 confirmation FAIL:** `head:planck-rank` paired ratio **0.71** on new races and seeds (0.67 / 0.80 / 0.66), 23.5% ± 0.5 vs teacher 33.7%; beats hash every seed (p ≤ 1e-18); the rank-over-nll gain shrinks to +1.7 pts (n.s.).
  - G0-fresh and G2 must be re-run after the search-robustness fixes.
- **ROUND 3 RESULTS OF RECORD (attempt 2, 2026-10-07, a77e502; all tools runs on Wikipedia search because SearXNG did not start in time):**
  - **A1/A2 hold:**
    - On 143 fresh + long-tail questions, base chat scores 4.9% (2026 facts **0/60**, wrong 95% of the time it answers).
    - The snippet-first heuristic scores 15.4% (fresh 5/60, long-tail 17/83), better than base chat overall (p = 0.006).
    - Absolute accuracy is low because Wikipedia search reaches the answer rarely (18.9% in the top-3 snippets).
  - **G2 FAIL:**
    - The learned scorer beats the deterministic ranker offline (choice accuracy 59.4% vs 39.8%; Planck > hash).
    - The policy abstains almost always: the single softmax with "none of these" makes it under-confident under train/deploy shift.
    - The next design must decouple choice from answerability.
  - **G1 FAIL confirmed:** planck-rank 0.71, hertz-rank 0.70 (capacity is not the lever), ~0.7× the teacher at ~500x speed.
- **G2-G4:** next design round pending (see `SETUP_planck_20260903.md` attempt 2).
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
