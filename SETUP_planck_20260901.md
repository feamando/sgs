# Planck 3.0 run guide (Windows 4090 box)

Nikita Gorshkov · 2026-09-29 · Status: ready for first run · Swimlane 1 (Planck) · Box: `C:\Users\feama\sgs` · Plan and rationale: `SETUP_092026_planck3.md`

**TL;DR.** Three commands, in order: `doctor -Deep` (2 min, catches anything that would kill a run and measures an ETA) → `all -Quick -NoPush` (15-30 min shakedown of every stage) → `all` (the real run, pushes results to git so they can be read from any machine). Then open the product with `serve`. Every stage is idempotent: if anything stops, fix it and re-run the same command; finished work is skipped. Nothing here needs editing or flags beyond what is written below.

## 0. Before you start

| Need | Where it lives | If missing |
|---|---|---|
| Repo on the box | `C:\Users\feama\sgs` | `git clone https://github.com/feamando/sgs` |
| Python venv with CUDA torch | `.venv` (the existing main one) | the runner creates `.venv`; then `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Planck 1.3 checkpoint + tokenizer | `checkpoints\planck13\best.pt`, `data\wikipedia\tokenizer.model` | G1 still runs, without head:planck (verdict INCOMPLETE) |
| Gemma 4 E4B teacher | `models\gemma-4-e4b-it` | `huggingface-cli download google/gemma-4-E4B-it --local-dir models/gemma-4-e4b-it`; without it G0/G1 run the no-model floor only |
| Docker Desktop (optional) | running in the tray | search falls back to the Wikipedia API (valid, narrower) |
| Disk | ~5 GB free | doctor warns below 5 GB |
| Free GPU | nothing else on the 4090 | stop the Raum app (`infer_decomposer.py --serve`) if it is running; Gemma needs ~16 GB |

Open **PowerShell** (5.1 or 7 both work) in the repo folder. If Windows says "running scripts is disabled", prefix every command with `powershell -ExecutionPolicy Bypass -File` instead of `.\`, for example `powershell -ExecutionPolicy Bypass -File scripts\planck3.ps1 doctor -Deep`.

## 1. Step by step

### Step 1: get the latest code (10 s)

```powershell
cd C:\Users\feama\sgs
git pull
```

### Step 2: preflight (about 2 min)

```powershell
.\scripts\planck3.ps1 doctor -Deep
```

It prints one row per check (`OK` / `WARN` / `FAIL`, with the fix under each non-OK row) and then an **estimated time per stage**, measured from this machine. What matters:

- **No `FAIL` rows.** A FAIL stops everything; the fix is printed next to it. The one most worth catching early: `planck ckpt vs tokenizer` (a tokenizer that does not belong to the checkpoint would make head:planck garbage).
- **`gemma typed decisions` shows 3/3 valid.** This is the teacher producing correct typed actions on three fixed decision points. If it is below 3/3, stop and send me `results\planck3\doctor.json`: the G0 run would be worthless.
- **`gemma closed-book`** answers "What year was IKEA founded?" with 1943.
- **`planck embed speed`** gives the minutes G1 will spend embedding the Wikipedia graph.

WARN rows are fine to proceed with (they only switch off a comparison), as long as you know which one.

### Step 3: shakedown (about 15-30 min)

```powershell
.\scripts\planck3.ps1 all -Quick -NoPush
```

Runs every stage small (12 benchmark tasks, 300 Wikiracing targets, 20 teacher races) into `*_quick` folders, then prints the results table. It does **not** push and does **not** block the full run later. The first time it also downloads the Wikipedia dump (356 MB) and embeds the graph; those are shared with the full run, so they are not repeated.

Check in the printed table (also in `results\planck3\REPORT.md`):

- G0 rows exist for `g0_gemma_quick`, `g0_gemma_closedbook_quick`, `g0_heuristic_quick`.
- The Gemma run's log shows `n_invalid_decisions 0` (or close to it).
- The G1 table lists `head:planck` and `gemma`, not just random/lexical/head:hash.

If all three hold, the pipeline works end to end.

### Step 4: the real run (time from the doctor's ETA, typically 1-2 h)

```powershell
.\scripts\planck3.ps1 all
```

Same stages at full size. It ends by committing and pushing: summaries, answer cards, `REPORT.md`, the run log and compressed teacher trajectories (`results\planck3\...`). If it is interrupted, run the same command again; finished stages print `SKIP`.

### Step 5: try the product (as long as you like)

```powershell
.\scripts\planck3.ps1 serve
```

Open http://127.0.0.1:8010. A short script that exercises everything:

1. `When was IKEA founded?`, then `And H&M?`, then `What about Zara?` (follow-ups become standalone questions)
2. `Who founded SpaceX?`, then `When was it founded?` (pronoun carried over)
3. Open **In depth** under an answer: ranked evidence passages, sources, further reading. Click **trust more / trust less** on a source.
4. Ask `When was IKEA founded?` again: it answers **from memory**, with its evidence, and 0 web calls.
5. Click **For you**: your interests, what is adjacent to them, from trusted sources only.

Terminal alternative: `.\scripts\planck3.ps1 chat` (type `more` for the evidence, `new` to reset, `quit` to exit). One-off question: `.\scripts\planck3.ps1 ask "Who painted the Mona Lisa?"`.

### Step 6 (optional): keep the knowledge graph growing daily

```powershell
.\scripts\planck3.ps1 schedule -At 08:00     # daily "digest -Explore" via Task Scheduler
.\scripts\planck3.ps1 unschedule             # remove it
```

Each run reads entities adjacent to your interests from **trusted sources only** into the local graph and writes `results\planck3\digest.md`. It runs only while you are logged on.

### Step 7: results back to me

Nothing to do if Step 4 pushed: I `git pull` and read `REPORT.md`, the logs and the trajectories directly. If the push failed, paste `results\planck3\REPORT.md`.

## 2. Reading the results

`results\planck3\REPORT.md` has one row per run. The comparisons that matter:

| Question | Where to look | Good looks like |
|---|---|---|
| Can the teacher drive our tools? (G0) | `g0_gemma` verdict | **PASS** (success ≥60%) |
| Does it match the rival on quality? | `g0_gemma` vs `g0_gemma_closedbook` success, the "vs base chat" line | ≥0.9× closed-book. On stable facts base chat may win; that is expected and fine |
| Is the depth layer useful? | `depth evidence` column | ≥0.8 (Mac floor was 0.787) |
| Better than plain search? | success vs `search-only @3` | success ≥ search-only |
| Does Planck learn to choose? (G1) | G1 table, verdict line | **PASS**: head:planck ≥0.8× gemma rollout success, and head:planck > head:hash |
| Fast enough for any device? | G1 `ms / decision` for head:planck | <100 ms |

What each verdict leads to:

| Verdict | Meaning | Next |
|---|---|---|
| G0 **PASS** | teacher solves ≥60% through typed actions | distill (G2) from `trajectories.jsonl.gz` |
| G0 **GREY ZONE** (40-60%) | works, but failure cases matter | read `results\planck3\g0_gemma\cards.md`: tool miss (`gold_reachable` false) or teacher choice? |
| G0 **KILL** (<40%) | the action space or tools are wrong | fix tools before any training |
| G1 **PASS** | a ~100M SGS model routes almost like an 8B teacher | reseed (`g1 -Seed 1`, `-Seed 2`) before believing it |
| G1 **FAIL (try the Hertz encoder)** | Planck features too weak | capacity ablation with Hertz 1.2 |
| G1 **INCOMPLETE** | head:planck or gemma missing | fix the WARN from the doctor, re-run `g1` |

## 3. Re-running pieces

| I want to | Command |
|---|---|
| Re-run only G0 | `.\scripts\planck3.ps1 g0` (always re-runs; `all` skips finished runs) |
| Re-run only G1 | `.\scripts\planck3.ps1 g1` |
| A second seed | `.\scripts\planck3.ps1 g1 -Seed 1` |
| Force one stage from scratch | delete its folder under `results\planck3\` (or `data\planck3\wikirace\tasks_info.json` for tasks), re-run |
| Try another teacher | `.\scripts\planck3.ps1 g0 -Policy bedrock` (needs AWS credentials for Bedrock Haiku) |
| Re-score G0 without touching the web | `.\scripts\planck3.ps1 py g0 --policy gemma --net replay --out results/planck3/g0_gemma_replay` |
| See all results | `.\scripts\planck3.ps1 report` |
| Everything the CLI can do | `.\scripts\planck3.ps1 py --help` |

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "running scripts is disabled on this system" | PowerShell execution policy | use `powershell -ExecutionPolicy Bypass -File scripts\planck3.ps1 <command>` |
| doctor `FAIL planck ckpt vs tokenizer` | tokenizer from a different model | use `data\wikipedia\tokenizer.model` that shipped with Planck 1.3 |
| doctor `WARN cuda not available` | CPU-only torch in `.venv` | `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `CUDA out of memory` while loading Gemma | another process holds VRAM (Raum app, a notebook) | close it; check with `nvidia-smi` |
| `gemma typed decisions` below 3/3 | teacher replies are not parseable JSON | send `results\planck3\doctor.json`; do not run G0 yet |
| SearXNG never comes up | Docker Desktop not running, or first image pull slow | start Docker Desktop, then `.\scripts\planck3.ps1 searxng`; `docker logs planck3-searxng`. Runs fall back to Wikipedia search either way |
| Download stops during G1 build | network blip | re-run; the dump download resumes |
| `UnicodeEncodeError` | running python directly in a cp1252 console | always go through `.\scripts\planck3.ps1` (it sets UTF-8) |
| `git push` rejected at the end | the Mac pushed in between | `git pull --rebase`, then `.\scripts\planck3.ps1 all` (everything finished is skipped, it just pushes) |
| Anything else | | the full log is in `results\planck3\logs\`; the last lines name the failing command |

## 5. What is where

| Path | What |
|---|---|
| `scripts\planck3.ps1` | this runner |
| `scripts\planck3.py`, `src\planck3\` | the code (CLI, harness, tools, store/graph, chat, digest, Wikiracing, doctor) |
| `scripts\assets\planck3_tasks.json` | the 47-task seed benchmark (fact, compare, chat) |
| `config\planck3_prices.json` | cost model (Haiku 4.5 $1/$5 per MTok; editable) |
| `config\searxng\settings.yml` | local search engine config |
| `results\planck3\REPORT.md` | all results in one table |
| `results\planck3\g0_*\` | per G0 run: `summary.json`, `cards.md` (every answer, readable), `results.jsonl`, `trajectories.jsonl.gz` |
| `results\planck3\g1_*\` | G1 heads (`head.pt`, local only) and eval `summary.json` |
| `results\planck3\logs\` | full run logs |
| `results\planck3\personal_store.sqlite` | your knowledge graph from `serve` / `chat` / `ask` (local only, never pushed) |
| `data\planck3\wikirace\` | Wikipedia dump, graph, embeddings (local only) |
| `data\planck3_cache\` | every page and search result fetched, for exact replay (local only) |

## 6. After the first run

1. Read `REPORT.md` together; decide on G0 (PASS / grey / kill) and G1.
2. If G0 passes: G2 distillation from the teacher trajectories, plus fresh consumer tasks with dated gold (where base chat's cutoff should show).
3. Decide on snippet-first extraction (zero-fetch answers) with the teacher's real behaviour in hand.
