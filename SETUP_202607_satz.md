# SGS Satz Setup, July 2026 (model selector: Planck ↔ Hertz)

*Windows + RTX 4090. Other platforms not validated.*

Satz is the **text** product track: a local web app that demos "SGS LM +
blob retrieval". v0.1 (`satz/app.py` + `satz/static/`) is built and Planck-only.
This doc covers (a) firing up Satz as-is, and (b) adding a **model selector**
between Planck 1.3 and Hertz 1.2. Origin: 2026-07 work list. Companions:
`docs/plans/satz_01_plan.md`, [[project_hertz_resume_epoch_restart]] (Hertz is
trained, `best.pt` val 3.1832 / ppl 24.1).

## 0. Environment

```powershell
cd sgs
.venv\Scripts\Activate.ps1
python -c "import fastapi, uvicorn; print('satz deps ok')"
# if missing:  pip install -r satz\requirements.txt
```

## 1. Artifacts you need (all live on the box; data/ + *.pt are gitignored)

| Model | Checkpoint | Tokenizer | Blob store | Arch (d_s / d_f / passes / heads / ctx) |
|---|---|---|---|---|
| **Planck 1.3** | `checkpoints/planck/best.pt` | `data/wikipedia/tokenizer.model` | `data/blobs/wikipedia/{blobs.pt,meta.json}` | 128 / 1000 / 3 / 4 / 512 |
| **Hertz 1.2** | `checkpoints/hertz12/best.pt` | `data/hertz12_data/tokenizer.model` (32K SP) | *none built* | 256 / 3700 / 3 / 4 / 512 |

Confirm what's actually present before assuming:

```powershell
Get-ChildItem checkpoints\planck\best.pt, checkpoints\hertz12\best.pt -ErrorAction SilentlyContinue
Get-ChildItem data\blobs\wikipedia\blobs.pt -ErrorAction SilentlyContinue
Get-ChildItem data\wikipedia\tokenizer.model, data\hertz12_data\tokenizer.model -ErrorAction SilentlyContinue
```

The Planck arch defaults are already baked into `satz/app.py`. Hertz uses a
DIFFERENT arch (d_s=256, d_f=3700), which is why a naive `--checkpoint hertz`
would load with the wrong shapes, hence the selector below.

## 2. Fire up Satz TODAY (Planck, blob demo) — no code changes

```powershell
python -m satz.app `
  --checkpoint checkpoints\planck\best.pt `
  --tokenizer data\wikipedia\tokenizer.model `
  --blobs-dir data\blobs\wikipedia `
  --d-s 128 --d-f 1000 --n-passes 3 --n-heads 4 --context-len 512 `
  --port 8001
# then open http://localhost:8001
```

**Gate:** page loads, a prompt returns a continuation, ≥1 blob shows in the
right panel with non-zero weight, the `k` slider re-ranks. If the Planck ckpt or
blob index is missing, the app exits with a clear error — build them first
(`scripts/build_blobs.py`) or fall back to the Planck 1.1 + TinyStories bundle
flagged as placeholder (see `satz_01_plan.md` §2).

## 3. Add the model selector (Planck ↔ Hertz) — the v0.2 change

**Design decision, decide first:** Hertz has NO blob store. Two options:
- **(A) blob-free Hertz (recommended, cheap):** selector picks the model; for
  Hertz the blob panel is hidden/greyed and generation runs plain. Ships the
  "bigger model, better text" story immediately. ~1 session.
- **(B) build a Hertz blob index first:** run `scripts/build_blobs.py` against a
  corpus with the Hertz tokenizer, then Hertz gets the same blob UI. Correct but
  is its own task (GPU + index build); do only if the blob story needs the 1B.

Recommend A now, B later.

### 3.1 Backend (`satz/app.py`)
- Add a `MODELS` registry mapping name → {checkpoint, tokenizer, blobs_dir|None,
  arch preset}. Two entries: `planck` (arch 128/1000, blobs), `hertz`
  (arch 256/3700, blobs=None).
- Replace the single required `--checkpoint` with `--model {planck,hertz}` (plus
  optional path overrides). Build `SatzRuntime` from the selected preset. Make
  the blob store **optional**: if `blobs_dir` is None, skip loading and set a
  `has_blobs=False` flag.
- `/generate`: when `has_blobs` is False, skip retrieval and return
  `blobs: []` + `has_blobs: false`.
- `/models` (new): return the registry keys + which have blobs, so the frontend
  can populate the selector and grey the blob panel for Hertz.
- Keep both runtimes lazy-loaded (a 1B Hertz + a 100M Planck both resident is
  ~2.6GB fp/1.3GB bf16; fine on a 4090, but load on first use to keep startup
  fast). Simplest: load the default model at boot, swap on selector change.

### 3.2 Frontend (`satz/static/`)
- Add a model `<select>` (Planck / Hertz) at the top; on change, POST the model
  name and re-run. When the chosen model has `has_blobs=false`, grey out the
  right panel + `k` slider with a caption "Hertz runs blob-free in v0.2".

### 3.3 Gate
Switch to Hertz, get a coherent continuation with the blob panel greyed; switch
back to Planck, blob panel returns and re-ranks. Both on localhost.

## 4. Notes / traps
- **Arch must match the checkpoint** or `load_state_dict` fails on shape
  mismatch. The presets in §1 are the source of truth; don't pass Planck arch to
  a Hertz ckpt.
- **Tokenizers differ** (Planck Wikipedia SP vs Hertz 32K SP). The selector must
  swap the tokenizer with the model, not just the weights.
- **`best.pt` is a full ckpt**; `app.py` already handles `ckpt["model"]` and runs
  `migrate_state_dict`, so both Planck and Hertz `best.pt` load the same way.
- **Hertz `milestone_*_bf16.pt` are inference-only** and would also work for
  Satz (model-only), if you'd rather serve the lighter bf16 snapshot.

## 4b. Conversation logging (for analysis)

On by default. Every generation appends one JSON line to
`runs/satz_conversations.jsonl`: timestamp, session id (one per browser page
load), model, prompt, output, token counts, gen latency + tok/s, and retrieved
blob indices/scores. Writes are lock-guarded and best-effort (a log failure
never breaks a request).

```powershell
python -m satz.app --model hertz                              # logs to runs/satz_conversations.jsonl
python -m satz.app --model hertz --log-file runs\demo_day.jsonl   # custom path
python -m satz.app --model hertz --no-log                     # disable
```

Analyze:

```powershell
python -m satz.analyze_log --log runs\satz_conversations.jsonl
python -m satz.analyze_log --log runs\satz_conversations.jsonl --model hertz --show 5
# or: import pandas as pd; pd.read_json("runs/satz_conversations.jsonl", lines=True)
```

**PRIVACY:** the log holds real user prompts and the repo is PUBLIC. The log
files are gitignored (`runs/satz_conversations*.jsonl`). Do NOT commit them or
paste raw prompts into public posts.

## 5. Scope
v0.2 = selector + blob-free Hertz (option A). Do NOT build a Hertz blob index or
streaming in this pass; log them as v0.3. Stop if Hertz generation quality is
too weak to demo honestly (same stop condition as `satz_01_plan.md` §4).
