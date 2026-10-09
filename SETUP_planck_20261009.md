# Planck 3.0 round 5: read → weigh → write, small models, hand-scored trust

Nikita Gorshkov · 2026-10-09 · Status: ready to run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Previous round: `SETUP_planck_20261008.md` · Plan and results of record: `SETUP_092026_planck3.md`

**TL;DR.** One command, `.\scripts\planck3.ps1 round5` (about 4-5 h, idempotent, pushes results). It tests the product as you defined it: a simple AI search that writes an answer from retrieved sources, with confidence and source accuracy, every step auditable, and personal trust that can tilt the result by at most 30%. The answer is built in three steps:
1. **Read.** A small model picks which values in the sources answer the question.
2. **Weigh.** Plain arithmetic turns trust, agreement, conflicts and question type into a 0-10 confidence, itemized point by point.
3. **Write.** A small model writes the answer, and a check rejects any name or number that isn't in the sources.

Training happens once, for *skills* (reading, writing), never for *the world*: facts arrive through search, so nothing needs retraining as the news changes. **Planck 1.3 (100M) reads, Hertz 1.2 (640M) or Planck writes; Gemma is only the reference and the one-time teacher.** There are three benchmarks: the 143 fresh + long-tail questions, plus 60 news and 40 general-information questions drafted this round. The rules below were written **before** the run (2026-10-09).

## 0. Why this design

- **Customer:** an answer the user can audit. They see what was read, which sources say what, how each confidence point was earned or lost, and when their own preferences changed the answer.
- **Commercial:** it runs on a phone. A 100M reader and a 640M writer are 6-40x smaller than Gemma 4 E4B, and they never need retraining for new facts.

**Training the world vs training a skill.** Round 4's G2 v2 *gate* learned what answerable questions looked like in its training data (past events). It never trusted answers about 2026, so it was world-dependent. Round 4 also showed that its *choice head*, the reading skill, was right **62.9%** of the time against Gemma's **67.8%** (not significant). That gap is close enough to bet on. Round 5 keeps the reader, drops the gate, and replaces it with the transparent arithmetic. The writer is a skill too: it sees only the question, the chosen value and the numbered sources.

## 1. The design

**Read** (`src/planck3/readers.py`). A reader returns, for each candidate value, the probability that it answers the question. It never sees trust, so personal preferences cannot leak into reading.
- `heuristic`: the lexical ranker, no model.
- `planck`: the round 4 choice head on frozen Planck 1.3 features.
- `gemma`: the reference.

**Weigh** (`src/planck3/consolidate.py`). Equivalent values are merged ("Álex Palou" = "Alex Palou"). Copies of one source count once (Wikipedia + wikiwand + grokipedia = one family). The confidence is then built from fixed terms:

| Term | Points |
|---|---|
| The most trusted source stating the value | its effective trust, 0-10 |
| Each other independent source at 7+ that agrees | +1 (max +2) |
| A source at 7+ states a different value the reader also rated | −2 (both are shown) |
| The reader itself is unsure (p < 0.3 / p < 0.6) | −2 / −1 |
| Question type: news / encyclopedic / general information | −2 / −1 / 0 |

The total is clamped to 0-10 and banded like trust:

| Score | Band |
|---|---|
| 9-10 | high |
| 7-8 | good, might have bias |
| 5-6 | low ("Low confidence in my results") |
| below 5 | treat with scepticism |

Every term appears on the card.

**Write** (`src/planck3/writer.py`).
- **Template:** no model.
- **Hertz / Planck writers:** fine-tuned **once** on Gemma-written answers for the 1,200 G2 *training* questions. Only faithful examples are kept, and the loss covers the answer tokens only.
- **Gemma:** the reference writer.

Every model answer is checked. It must state the chosen value and cite only listed sources, and every number and name in it must appear in the question or the sources. A failing answer is replaced by the template, and the fallback is counted.

**Trust** (`config/planck3/source_registry.json`, `src/planck3/registry.py`).
- **Your rubric:** each domain gets its category's score unless you scored it explicitly. Unknown domains score 5 and are flagged "unscored".
- **Rubric-derived rules** cover sites nobody listed: government suffixes (`.gov.uk`, `.gouv.fr`, `.go.jp`...), `.edu`, `.int`, and operator keywords (airport, airways, museum...).
- **Personal score:** effective trust = 0.7 × system + 0.3 × yours, so a source moves at most 3 points.
  - **Per source:** a 3+ point gap between you and the system carries a "you disagree" badge.
  - **Per answer:** the answer is also computed without your scores. If yours change the value or the band, the card says so and shows the system-only answer.
- **"More results from here"** fetches more from that site in this chat. It never changes trust.
- **Usage stats no longer move trust.** Round 4's implicit "corroborated" signals are kept as an audit record only, so every score stays explainable.

**Your answers, applied:**
- **US agencies:** technical agencies (CDC, NIH, NOAA, NASA, Census, BLS, FEC...) score **9**, which is 10 historically minus 1 for the current administration. Other `.gov` sources score 8. Directly controlled sources (White House communications, VOA, RFE/RL, Stars and Stripes) score 5.
- **DW, ARD, ZDF:** **10** (licence fee, independence in the Basic Law). The BBC stays at 7 despite a similar licence-fee model, because of reported weaknesses in the robustness of its editorial process (owner, 2026-10-09).
- **Pressure jurisdictions:** **4** for media located in Russia, China, the Gulf states or Israel, state-controlled or not. **2** for state media (RT, TASS, Xinhua, CGTN, Global Times). Exiled outlets like Meduza and Novaya Gazeta Europe count as independent.
- **Grokipedia:** **4** (AI-generated), and counted as a Wikipedia copy, not independent confirmation.

**Rows marked PROPOSED in the registry extend your rubric and need your review.** They cover:
- intergovernmental agencies (9)
- UK / EEA governments (9)
- the organiser on its own results (9)
- other democracies' governments (8)
- universities (8)
- company self-descriptions (8)
- curated libraries (8)
- other democracies' news (7)
- specialist databases (7)
- aggregators (6)
- fan sites and blogs (5)
- scraped databases (5)
- content farms (4)
- betting (4)
- explicit domains: arXiv at 8 (preprints), medium.com at 4, andrewteale.me.uk at 6

## 2. Benchmarks

| Set | n | Built from | Question type |
|---|---|---|---|
| `planck3_tasks_fresh.json` | 143 | Wikidata (round 3); 60 fresh 2026 facts + 83 long-tail | news / encyclopedic |
| `planck3_tasks_news.json` | 60 | news 2026-09-08 to 10-08; 2 independent sources each, quotes checked word for word | news |
| `planck3_tasks_geninfo.json` | 40 | official sites (airports, operators, governments, museums) + a second source | general information |

**Your spot-check (10 questions, the ones the drafting flagged as least certain):**
- `news056`: Greenland bases "2", per Trump's statement
- `news044`: Greater Noida bus fire "9" (early toll)
- `news043`: Urubici crash date
- `news008`: Latvia result (preliminary)
- `news034`: Japan centenarians "107,677"
- `info009`: Kenya Airways at Nairobi T1A
- `info004`: Schiphol 6 runways (one source)
- `info040`: Robben Island R600 (peak-season price)
- `info030`: Singapore Deepavali observed on 9 Nov
- `info021`: UK visitor stay "6 months"

If a gold answer is wrong, fix it in the JSON and run `planck3.py r5 --rescore --out results/planck3/<run> --tasks ...`. That re-judges a finished run without searching or calling a model.

## 3. What runs (in order)

| Stage | What | Output | Time |
|---|---|---|---|
| setup + smoke + doctor + ddgs check | 71 offline tests | `doctor.json` | ~5 min |
| **floor** | heuristic reader + template writer, 243 questions | `r5_heuristic_template` | ~30 min (100 new searches) |
| **reader only** | Planck reader + template writer | `r5_planck_template` | ~15 min |
| **writer data** | Gemma writes answers for 1,200 G2 *training* questions, faithfulness-filtered (resumable) | `data/planck3/writer/train.jsonl` | ~1-1.5 h |
| **writers, once** | fine-tune Hertz 1.2 and Planck 1.3 (answer tokens only, best by validation loss) | `checkpoints/writer_{hertz,planck}/`, `writer_*/train_log.json` | ~15 min |
| **product** | Planck reader + Hertz writer; Planck reader + Planck writer | `r5_planck_hertz`, `r5_planck_planck` | ~30 min |
| **reference** | Gemma reader + Gemma writer | `r5_gemma_gemma` | ~40 min |
| **rule E** | product with the worst-case personal profile (every source scored 10 − system) | `r5_planck_hertz_contrarian` | ~15 min |
| **Brave** | product on the Brave Search API (your key in `.env`) | `r5_planck_hertz_brave` | ~25 min |
| **phone proxy** | product on CPU only, 4 threads, 10 questions (no KV cache yet, so this is slow) | `r5_planck_hertz_cpu` | ~10-30 min |
| report + push | REPORT.md gains "Round 5" with bands, writer checks and paired comparisons | git | 1 min |

## 4. Step by step

1. `git pull`
2. Check `.env` has `BRAVE_API_KEY` (you set it already; `.\scripts\planck3.ps1 setkey` if not). No Docker needed.
3. Shakedown, about 25 min (12 questions per run, 60 writer examples, `_quick` outputs):
   ```powershell
   .\scripts\planck3.ps1 round5 -Quick -NoPush
   ```
   Check that REPORT.md has a "Round 5" table with `r5_*_quick` rows.
4. The real run:
   ```powershell
   .\scripts\planck3.ps1 round5
   ```
   Re-run the same command if it stops; finished stages are skipped.
5. To try the product afterwards: `.\scripts\planck3.ps1 serve`. It picks up the Planck reader and the Hertz writer automatically.

## 5. Pre-registered rules (2026-10-09, before the run)

The product is the Planck reader + Hertz writer (the Planck writer if Hertz is missing), on ddgs, over all 243 questions unless stated. Paired comparisons use exact McNemar on the same questions.

- **C1, the confidence means something (primary):**
  - the **high** band (9-10) is right ≥ **90%** of the time;
  - the **good** band (7-8) is right ≥ **70%**;
  - precision falls band by band (high ≥ good ≥ low ≥ sceptical).

  A band with fewer than 20 answers is reported as "too few to judge"; bands with fewer than 5 are left out of the ordering check.
- **C2, no loss from the new weighing:** the product's shown answer is right ≥ **62.9%** on the 143 fresh + long-tail questions. That is round 4's figure for the same Planck reader.
- **C3, small models instead of Gemma:** the product is not significantly worse than Gemma reader + Gemma writer (FAIL if Gemma wins with p < 0.05).
- **W, the small writer is faithful:** its own answers pass the faithfulness check ≥ **95%** of the time (template fallback ≤ 5%). The shown answer is always faithful, because a failed one is replaced.
- **E, no echo chamber:** under the worst-case personal profile, accuracy drops by at most **10 points** against the product. Every changed answer is flagged on the card (checked per answer).
- **B, Brave vs ddgs:** reported, paired. Brave becomes the product search if it is not worse (no p < 0.05 the wrong way).
- **Reported, no bar:**
  - **L:** CPU-only latency (reader ms + writer ms per question; no KV cache yet).
  - **Q:** question-type accuracy.
  - **Registry coverage:** how often an answer rests on a hand-scored source.
  - **LLM tokens per question.**

## 6. Disclosures (before any round 5 number exists)

1. **The question-type word lists were extended after reading the drafted benchmarks:** service words like runways, terminals, emergency numbers and museum closing days, news events outranking service words, and seasons like "2025-26". Question-type accuracy on these sets is therefore reported, not judged.
2. **Registry domains** come from the round 4 search logs (the 160 most cited) and your named sources, never from the benchmark source lists. The government suffixes and operator keywords are general rules, and the keywords were chosen with the general-information questions in view. Registry coverage is reported so the effect is visible.
3. **The confidence formula's constants are set by hand above and not fitted.** If C1 fails, they change only with a stated reason and a new pre-registration, never by fitting to these questions.
4. **Known limits:**
   - the SGS models have a 512-token context (snippets are shortened to fit) and no KV cache, so generation on CPU is slower than it needs to be;
   - unscored sites score 5, which drags confidence down wherever the registry has gaps.

## 7. Effort shape (reuse, don't rebuild)

| Piece | Reuse vs. new | Rough effort |
|---|---|---|
| Reader | reuse the round 4 G2 v2 choice head unchanged; gate dropped | XS |
| Weighing | new `consolidate.py` (clusters, families, itemized confidence, divergence) | M |
| Writers | reuse `SGSLanguageModel` + the `train_decomposer.py` fine-tune pattern; new `writer.py` with faithfulness check | M |
| Trust | new `registry.py` + hand-scored registry; store keeps a personal score, 30% blend | M |
| Question type | new rule file `qtype.py` | XS |
| Benchmarks | 100 new questions drafted with 2 sources each; spot-check by you | M |
| Runner + report | `round5`, `Test-R5Run`, report rows, paired comparisons, `--rescore` | S |

## 8. Recommendations

1. Spot-check the 10 questions above, then run `round5 -Quick -NoPush`, then `round5`.
2. **Review the PROPOSED registry rows.** They set the scores most answers will rest on.
3. **If C1 and C3 pass,** the product is the Planck reader + Hertz writer, with Gemma out of the runtime. Next: a KV cache for the SGS models and a real phone measurement.
4. **If C3 fails because of the reader,** the next step is a better reader trained once on broader reading data (news-style snippets included). It is still a skill, so still no retraining per world change.
5. **If C1 fails,** the bands are not yet honest. Change the confidence terms with a stated reason before showing scores to anyone.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| The small writer invents names or numbers | faithfulness check on every answer, template fallback, fallback rate reported (rule W) |
| A personal profile turns the product into an echo chamber | 30% cap by construction, system-only answer computed every time, divergence shown, worst-case profile measured (rule E) |
| Registry gaps (unscored sites at 5) drag confidence down | coverage reported; rubric-derived suffix and keyword rules; review queue of most-cited unscored domains |
| Benchmark gold wrong (drafted this round) | 2 sources + verbatim quotes per answer, your spot-check, `--rescore` without re-running |
| Hertz is slow on CPU without a KV cache | latency measured (rule L); a KV cache is the next engineering step if the product passes |
| Gemma's writing style leaks Gemma's mistakes into the small writers | only faithful examples are kept; the writer only sees the sources, so it learns phrasing, not facts |
