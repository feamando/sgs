# SGS — Hertz 1.2 Setup & Kickoff (June 2026, edition 1)

*Windows + RTX 4090, 1 TB drive. Other platforms not validated.*

Purpose: get **Hertz 1.2** (a ~1B SGS language model) training in a single
unattended run you can start before vacation and leave alone. Hertz is also the
encoder Raum will use (frozen base + decomposer head, later).

Requirements rationale: `docs/plans/hertz_12_requirements.md`.
Architecture/model: `src/sgs_lm.py`. Trainer: `scripts/train_hertz.py`.

**Important reality check (verified 2026-06-04):** much of this is ALREADY BUILT.
`scripts/train_hertz.py` exists with 1B defaults, FineWeb-Edu download/tokenize,
resume + warm-start, AND checkpoint rotation (`--keep-last`, `_rotate_step_checkpoints`).
So this is mostly **verify + fill small gaps + run**, not build-from-scratch. The
"What needs to be built" section at the end is the honest gap list.

## 0. Decisions locked for this run

| Decision | Choice | Why |
|----------|--------|-----|
| Multimodal base? | **No, text-only** | Raum = frozen-Hertz + decomposer head later. Don't couple two unsolved problems on an unattended run. |
| Corpus | **FineWeb-Edu + Wikipedia mix** (default, `--dataset fineweb-wiki-mix`, 30% wiki) | one shared tokenizer over both; aligns base + blob retrieval distribution |
| Model size | **`d_f=3700` → 640M** (not the 1.04B default) | measured ~20k tok/s vs ~2k at d_f=5000; fits a 7-day run (§3.2) |
| Token budget | **`--max-tokens 10B`** | ~5.8 days at 20k tok/s, ~1 day cushion for a 7-day trip |
| Tokenizer | **Reuse the existing 32K SP** for this run | `train_hertz.py` defaults to it; training a new one is a separate task, not blocker for kickoff |
| Blobs | **Deferred** — built post-pretrain (Faiss, no GPU) when back | Not part of the GPU run |
| Optimizer | **Plain AdamW** | Muon is a confirmed regression ([[project_sgs_accel_shelved]]) |
| Logging | **stdout → file, no --wandb** | wandb is paid ([[feedback_sgs_wandb_default]]) |

## 1. Environment setup

### 1.1 Repo + venv (Python 3.12)

```powershell
cd $HOME\Documents\GitHub\sgs
git pull
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 1.2 Core dependencies

```powershell
# CUDA 12.1 torch (matches the 4090 / existing setup)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project deps (torch, datasets, sentencepiece, etc. from requirements.txt)
pip install -r requirements.txt
```

`requirements.txt` already pins what Hertz needs: `torch>=2.0`, `datasets>=2.14`,
`sentencepiece`, `pyarrow`, `tqdm`, `triton-windows` (compile backend). No extra
installs required for the text-only run.

### 1.3 GPU smoke check

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# Expect: NVIDIA GeForce RTX 4090 True
```

bf16 must be True — the run uses bf16 mixed precision.

### 1.4 Disk pre-flight (1 TB constraint)

```powershell
Get-PSDrive C | Select-Object Used,Free
```

Need comfortably > 200 GB free: raw FineWeb cache is transient but large during
download, and rolling checkpoints + tokenized `.bin` add ~80-115 GB working set
(see requirements §3). If tight, lower `--max-tokens`.

## 2. Dependencies summary (what each piece needs)

| Phase | Needs | Status |
|-------|-------|--------|
| Data download/tokenize | `datasets`, `sentencepiece`, network | wired (`prepare_fineweb`) |
| Training | `torch` cu121, bf16 GPU, `triton-windows` | wired (`train_hertz.py`) |
| Checkpoint rotation | nothing extra | wired (`--keep-last`) |
| Blob index (later) | `faiss-cpu` | not needed for this run |
| Raum head (later) | the trained Hertz checkpoint | out of scope here |

## 3. The plan (what actually runs)

### 3.1 Pre-flight smoke test — DO NOT SKIP

An unattended run that OOMs on hour 2 wastes the trip. Run a short job at the
REAL config first and watch four things: no OOM, measured tok/s, disk delta, and
that a resume works.

**Two gotchas that bit a real smoke run (2026-06-12) — both baked into the
command below:**

1. **`--d-f 3700` is NOT a default.** The trainer defaults to `--d-f 5000`
   (1.04B, ~1-2k tok/s — the slow regression). If you omit `--d-f 3700` you
   smoke-test the WRONG architecture and measure useless throughput. The locked
   run is `d_f=3700` (§3.2); the smoke test must match it.
2. **`--max-tokens` does NOT size the smoke run, AND a stale data dir silently
   overrides it.** Two distinct facts: (a) `--max-tokens` only bounds *data
   prep*, never the training loop (§3.2b); (b) if `--data-dir` already contains
   `train.bin` + `val.bin` + `tokenizer.model`, prep is **skipped entirely** and
   the existing corpus is reused, ignoring both `--max-tokens` and
   `--wiki-fraction`. The shared `data\hertz_mix` already holds ~808M tokens from
   a prior prep, so a smoke test pointed there trains on 808M, not your tiny
   budget. **Always give the smoke test its own fresh `--data-dir`.**

```powershell
# ~hundreds of optimizer steps, REAL architecture (d_f=3700), TINY fresh corpus.
python -u scripts\train_hertz.py `
  --dataset fineweb-wiki-mix --wiki-fraction 0.1 `
  --data-dir data\hertz_smoke10m `
  --max-tokens 10M `
  --d-f 3700 --n-passes 3 --n-heads 4 --context-len 512 `
  --epochs 1 `
  --save-interval 50 `
  --keep-last 5 `
  --bf16-milestone-interval 100 `
  --eval-interval 50 `
  --save-dir checkpoints\hertz12_smoke 2>&1 | Tee-Object runs\hertz12_smoke.log
```

Check after it runs a few hundred steps (Ctrl-C is fine):
- **Phase 1 actually prepared ~10M tokens** — the log should say
  `Mix budget: 0.0B total -> ...` then `Train: ~10,000,000 tokens`, NOT
  `Data already prepared ... Train: 808,696,232 tokens`. If you see 808M, your
  `--data-dir` collided with the old cache — stop and use a fresh dir.
- **No OOM** at `--batch-size 2 --grad-accum 32 --d-f 3700`.
- **tok/s ≈ 20k** (NOT ~2k). The `ETA …h (…d)` on each log line is computed for
  the `--max-tokens` budget, so on this 10M smoke run ETA is meaningless; what
  matters is the raw tok/s. ~2k means you're accidentally on `d_f=5000`.
- **bf16 milestone** (`milestone_100_bf16.pt`, ~1.3 GB at 640M) written at step
  100 — confirm it appears and is ~1-2 GB, not 12 GB.
- **Disk delta** of `checkpoints\hertz12_smoke` → with `--keep-last 5` it
  plateaus at ~5 step checkpoints, not unbounded.
- **Resume works** — but `--keep-last` rotates early checkpoints away, so do NOT
  hardcode `step_100.pt` (it may already be deleted; this is exactly what failed
  on 2026-06-12). List the dir first and resume from whatever step exists:
  ```powershell
  dir checkpoints\hertz12_smoke\step_*.pt
  # pick the highest-numbered file that's actually present, e.g. step_250.pt:
  python -u scripts\train_hertz.py --resume checkpoints\hertz12_smoke\step_250.pt `
    --data-dir data\hertz_smoke10m --save-dir checkpoints\hertz12_smoke
  ```
  If resume throughput collapses (known Adam-state-reload issue on Windows), use
  `--warm-start` instead of `--resume`.

If OOM: enable `--grad-checkpoint` (costs ~30-40% throughput but fits VRAM), or
drop `--batch-size` to 1 and raise `--grad-accum` to 64.

### 3.2 Measured throughput → locked config (2026-06-04)

Smoke test on the actual 4090 settled this empirically:

| Config | Params | Measured tok/s | Verdict |
|--------|--------|----------------|---------|
| `d_f=5000` (1.04B) | 1.04B | ~1,100 (with save+eval every 50 steps; clean ~2k) | **too slow** — 10B ≈ weeks |
| **`d_f=3700` (640M)** | **640M** | **~20,000** | **locked — this is the run** |

`d_f=3700` reverts the regression the script's own comments flag (`d_f`
3700→5000 caused ~10k→~2k tok/s). It trains a legitimate ~640M base, ~18× faster.

**7-day trip budget at ~20k tok/s:**
- ~1.73B tokens/day → ~12.1B raw ceiling in 7 days.
- **`--max-tokens 10B`** = ~5.8 days pure compute, ~1 day cushion for save/eval
  overhead. Also Chinchilla-comfortable for 640M (20 tok/param = 12.8B).
- Conservative alternative if you want a guaranteed finish: `8B` (~4.6 days).

**Disk for 10B (1 TB, fine):** `train.bin` ~20 GB + checkpoints
(3×12.5 keep-last + best 12.5 + ~6 bf16 milestones ×2) ≈ ~62 GB → **~85 GB total.**

### 3.2b What actually controls run length (verified in code 2026-06-12)

`--max-tokens` is easy to misread as "train on this many tokens." It is not.
There are TWO independent levers and one silent trap:

- **`--max-tokens` sizes DATA PREP only.** `prepare_fineweb_wiki_mix` caps the
  corpus at `max_tokens` via a 4-chars-per-token budget (FineWeb gets
  `1 - wiki_fraction`, Wikipedia gets `wiki_fraction`). So a *fresh* prep at
  `10B` writes a ~10B-token `train.bin`. Good.
- **The training loop has NO token stop.** It runs `for epoch in range(--epochs)`
  over the ENTIRE prepared corpus. `--max-tokens` appears in the loop only to
  compute the displayed `ETA`. **Run length = corpus size × epochs**, full stop.
- **THE TRAP — stale data dir.** If `--data-dir` already has `train.bin` +
  `val.bin` + `tokenizer.model`, prep is **skipped** and the existing corpus is
  reused, silently ignoring `--max-tokens` AND `--wiki-fraction`. The shared
  `data\hertz_mix` currently holds **~808M tokens** from an earlier prep. Launch
  the 10B run against that dir and it trains 1 epoch over 808M (~11h at 20k
  tok/s), then idles for the rest of the trip — a badly undertrained model
  (~1.3 tok/param vs the intended ~16), with NO error. This is the single most
  dangerous failure mode for an unattended run.

**Consequence for the real run:** you MUST either prep a fresh 10B corpus into a
clean dir, or verify the existing `data\hertz_mix` genuinely contains ~10B
tokens. The §3.3 command and §3.4b checklist below enforce this.

### 3.3 Kick off the real run — LOCKED COMMAND

The smoke test is done; this is the exact command to run before leaving.
`python -u` = unbuffered stdout so the log flushes live (Tee-Object buffers
otherwise and the log lags badly — confirmed in the smoke test).

**FIRST, force a clean 10B prep (avoids the §3.2b stale-cache trap).** The old
`data\hertz_mix` holds ~808M tokens; do not let the run reuse it. Either point at
a brand-new dir (preferred — keeps the smoke corpus around), or delete the old
one:

```powershell
# Preferred: a dedicated dir for the real 10B corpus.
#   (If a previous *interrupted* 10B prep left a partial data\hertz12_data,
#    delete it first so prep restarts clean: Remove-Item -Recurse data\hertz12_data)
python -u scripts\train_hertz.py `
  --dataset fineweb-wiki-mix --wiki-fraction 0.3 `
  --data-dir data\hertz12_data `
  --d-f 3700 --n-passes 3 --n-heads 4 --context-len 512 `
  --max-tokens 10B `
  --epochs 1 `
  --batch-size 2 --grad-accum 32 `
  --mixed-precision bf16 `
  --lr 3e-4 --warmup-steps 2000 `
  --save-interval 2000 --keep-last 3 `
  --bf16-milestone-interval 5000 `
  --eval-interval 2000 `
  --save-dir checkpoints\hertz12 2>&1 | Tee-Object runs\hertz12_train.log
```

After it prints `Train: ...` confirm the count is **~10,000,000,000**, not 808M.
If it says `Data already prepared` and shows 808M, you pointed at a stale dir —
stop (Ctrl-C) and fix `--data-dir` before walking away.

At ~20k tok/s: `--save-interval 2000` ≈ every ~1.8h (crash loses little);
bf16 milestone every 5000 steps ≈ every ~4.5h.

To survive a closed terminal / logout (note `-u` and the `-df 3700`):

```powershell
Start-Process -NoNewWindow -FilePath python `
  -ArgumentList "-u scripts\train_hertz.py --dataset fineweb-wiki-mix --wiki-fraction 0.3 --data-dir data\hertz12_data --d-f 3700 --max-tokens 10B --epochs 1 --save-interval 2000 --keep-last 3 --bf16-milestone-interval 5000 --eval-interval 2000 --save-dir checkpoints\hertz12" `
  -RedirectStandardOutput runs\hertz12_train.log -RedirectStandardError runs\hertz12_err.log
```

Note `--data-dir data\hertz12_data` here too — without it the detached run reuses
the stale ~808M `data\hertz_mix` (§3.2b) and you'd never see the error because the
terminal is closed.

### 3.4 Unattended-safety checklist (tick before leaving)

- [ ] Smoke test passed at **`--d-f 3700`**: tok/s ≈ 20k (not ~2k), no OOM, disk plateaus with `--keep-last`, resume works (from an *existing* step, not hardcoded step_100)
- [ ] **Fresh `--data-dir` for the real run** (e.g. `data\hertz12_data`), OR the existing dir verified to hold ~10B tokens — NOT the stale 808M `data\hertz_mix` (§3.2b)
- [ ] `--d-f 3700` present (it is NOT a default — default is the slow 5000)
- [ ] `--keep-last 3` (or 2) set — caps step checkpoints; `*.pt` already gitignored
- [ ] No `--wandb`; logging to `runs\hertz12_train.log`
- [ ] Disk free > 200 GB after tokenization; raw cache deleted/transient
- [ ] Launched so it survives terminal close (Start-Process / Scheduled Task)
- [ ] Run length understood: it trains **1 epoch over the prepared corpus** (§3.2b); `--max-tokens` does NOT stop the loop, so the corpus size IS the cap

### 3.4b Before you walk out the door (final gate — do NOT skip)

Don't start this and immediately leave. With a fresh `--data-dir` (§3.3),
`--max-tokens 10B` triggers a fresh, large download + re-tokenize, so there's a
long Phase-1 stretch BEFORE training. Watch it reach a healthy steady state
first, then leave.

- [ ] **Confirm the corpus size FIRST.** The Phase-1 summary must print
      `Train: ~10,000,000,000 tokens`. If it says `Data already prepared` and a
      number like 808,696,232, you hit the stale-cache trap (§3.2b) — STOP, the
      run would train on 8% of the budget and idle for days. Fix `--data-dir`.
- [ ] **Launch detached** (the `Start-Process` variant in §3.3, with
      `--data-dir`) so a closed terminal / logout doesn't kill the multi-day run.
- [ ] **Watch it clear Phase 1** — fresh FineWeb+Wiki download (~20 GB train.bin)
      + new shared tokenizer over the corpus. Quiet, can take a while; not a hang
      (check `data\hertz12_data\train.bin` growing if unsure).
- [ ] **See Phase 2 actually start:** first 2-3 `loss … | NNNN tok/s | ETA …h
      (…d)` lines printed. Confirm tok/s ≈ 20k and **ETA ≈ 5-6 days** (fits the
      7-day window). If ETA is way off, stop and re-check `--d-f 3700`.
- [ ] **First bf16 milestone lands** (`checkpoints\hertz12\milestone_5000_bf16.pt`,
      ~2 GB not 12 GB) — proves milestone saves work over the long run.
- [ ] **`step_*.pt` rotation** keeps only the last 3 (disk won't fill).
- [ ] **Disk free** still > 200 GB after data prep + first checkpoints.
- [ ] Note the start time + the run command in `runs\` so resume is obvious if
      it dies (`--resume checkpoints\hertz12\step_<N>.pt`, or `--warm-start` if
      resume throughput collapses).

Only once all the above are green is it safe to leave for 7 days.

### 3.5 When you're back

1. Check `runs\hertz12_train.log` final val loss/ppl; confirm `final.pt` or latest `step_*.pt`.
2. **G-base gate:** held-out ppl beats Planck 1.3; MMLU-lite (no blobs) > Planck 1.3.
3. Build the blob index (Faiss, CPU) from the corpus — reuse Planck 1.3 pipeline.
4. Validate Wikipedia-blob QA end-to-end; watch **Gate 4a** (utilisation).
5. Then Raum: frozen-Hertz + decomposer-head fine-tune on composition trees.

## 4. Architecture (for reference)

**LOCKED for this run: `d_f=3700` → 640M params** (chosen for ~20k tok/s; see §3.2).
The `d_f=5000` / 1.04B default is too slow on the 4090 for a 7-day window.

- `d_s=256, d_f=3700, n_passes=3, n_heads=4, context_len=512` → ~640M params
- bf16 mixed precision, AdamW (betas 0.9/0.95, wd 0.1, fused), cosine + warmup
- torch.compile mode=default (no CUDA graphs — incompatible with grad accum here)
- Checkpoint = model + optimizer + scheduler (~12.5 GB); rotation keeps last N;
  bf16 milestone = model-only (~2 GB)
- Verified at kickoff: GPU 100%, ~23.7 GB used (fits 25.8 GB), loss falling
  cleanly from ~8.9, no OOM.

---

## What needs to be built (honest gap list)

### DONE (this session, 2026-06-04)
- ✅ **FineWeb-Edu + Wikipedia mix** — `prepare_fineweb_wiki_mix` in
  `src/tinystories.py`, wired into `train_hertz.py` as the default
  `--dataset fineweb-wiki-mix` (`--wiki-fraction`, default 0.3). One shared
  tokenizer over both; writes a reproducibility `manifest.json`.
- ✅ **bf16 weights-only milestone save** — `--bf16-milestone-interval`
  (default 10000); `_save_bf16_milestone` writes ~2 GB model-only snapshots,
  never rotated.
- ✅ **Throughput sanity** — each log line now prints `ETA …h (…d)` to finish the
  token budget at the current tok/s, so a too-slow run is visible in the smoke
  test. (`--profile-steps` still available for deep diagnosis.)

### Blocks nothing — run can start today
Nothing. The mix run is runnable now.

### Optional decision before kickoff
- **`d_f` throughput tradeoff.** If the smoke-test ETA is too long for the trip,
  lower `--d-f` toward 3700-4000 (smaller, faster) or cut `--max-tokens`. No code
  needed — both are flags.

### Nice-to-have (later quality upgrades)
- **Code slice in the mix** (The Stack v2). The mix builder currently does
  FineWeb+Wiki; adding code is a third source in the same pattern.
- **New code-aware tokenizer** (~48-64K). Reuse the 32K SP for now.

### Deferred by design (post-pretrain / later milestones)
- **Blob index build + QA** (Faiss, CPU) — when back, reuse Planck 1.3 pipeline.
- **Raum decomposer head** on frozen Hertz — the multimodal step, a later track.
- **Full blob-count sweep / progressive blob schedule** — supervised, later.

### Do NOT build
- Muon / compound accel recipes (confirmed regressions, [[project_sgs_accel_shelved]]).
- Joint multimodal pretrain (no paired data at scale; couples two hard problems).
