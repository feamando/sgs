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
| Corpus | **FineWeb-Edu + Wikipedia mix** (now the default, `--dataset fineweb-wiki-mix`, 30% wiki) | one shared tokenizer over both; aligns base + blob retrieval distribution |
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

```powershell
# ~100 optimizer steps on a tiny token budget, real architecture + mix corpus.
python scripts\train_hertz.py `
  --dataset fineweb-wiki-mix --wiki-fraction 0.3 `
  --max-tokens 200M `
  --epochs 1 `
  --save-interval 50 `
  --keep-last 2 `
  --bf16-milestone-interval 100 `
  --eval-interval 50 `
  --save-dir checkpoints\hertz12_smoke 2>&1 | Tee-Object runs\hertz12_smoke.log
```

Check after it runs a few hundred steps (Ctrl-C is fine):
- **No OOM** at `--batch-size 2 --grad-accum 32 --d-f 5000` (the 1B default).
- **tok/s AND the `ETA …h (…d)`** now printed on each log line → directly tells
  you whether `--max-tokens` finishes inside the trip window (§3.2). This is the
  throughput-sanity check; if ETA is too long, lower `--max-tokens` or `--d-f`.
- **bf16 milestone** (`milestone_*_bf16.pt`, ~2 GB) written at step 100 — confirm
  it appears and is ~2 GB, not 12 GB.
- **Disk delta** of `checkpoints\hertz12_smoke` → with `--keep-last 2` it should
  plateau at ~2 full checkpoints, not grow unbounded.
- **Resume works:**
  ```powershell
  python scripts\train_hertz.py --resume checkpoints\hertz12_smoke\step_100.pt --max-tokens 200M --save-dir checkpoints\hertz12_smoke
  ```
  If resume throughput collapses (known Adam-state-reload issue on Windows), use
  `--warm-start` instead of `--resume`.

If OOM: enable `--grad-checkpoint` (costs ~30-40% throughput but fits VRAM), or
drop `--batch-size` to 1 and raise `--grad-accum` to 64.

### 3.2 Set the token budget from measured tok/s

| Measured tok/s | ~tokens/day | 14-day trip budget | Recommended `--max-tokens` |
|----------------|-------------|--------------------|-----------------------------|
| ~2,000 (pessimistic) | ~170M | ~2.4B | `2B` |
| ~6,000 | ~520M | ~7.3B | `6B` |
| ~11,800 (optimistic, prior best) | ~1.0B | ~14B | `10B`–`14B` |

Pick a budget the trip window can actually finish (or cleanly checkpoint). The
run is resumable either way, so erring slightly high is fine — it just stops at
the token cap or you resume after.

### 3.3 Kick off the real run

```powershell
# Adjust --max-tokens from the smoke test. Logs to file; survives terminal close
# if launched via Start-Process or a Scheduled Task.
python scripts\train_hertz.py `
  --dataset fineweb-wiki-mix --wiki-fraction 0.3 `
  --max-tokens 10B `
  --epochs 1 `
  --batch-size 2 --grad-accum 32 `
  --d-f 5000 --n-passes 3 --n-heads 4 --context-len 512 `
  --mixed-precision bf16 `
  --lr 3e-4 --warmup-steps 2000 `
  --save-interval 5000 --keep-last 3 `
  --bf16-milestone-interval 10000 `
  --eval-interval 1000 `
  --save-dir checkpoints\hertz12 2>&1 | Tee-Object runs\hertz12_train.log
```

To make it survive a closed terminal / logout:

```powershell
Start-Process -NoNewWindow -FilePath python `
  -ArgumentList "scripts\train_hertz.py --max-tokens 10B --epochs 1 --save-dir checkpoints\hertz12 --keep-last 3" `
  -RedirectStandardOutput runs\hertz12_train.log -RedirectStandardError runs\hertz12_err.log
```

### 3.4 Unattended-safety checklist (tick before leaving)

- [ ] Smoke test passed: no OOM, tok/s measured, disk plateaus with `--keep-last`, resume works
- [ ] `--max-tokens` set from measured tok/s and trip length (§3.2)
- [ ] `--keep-last 3` (or 2) set — caps step checkpoints; `*.pt` already gitignored
- [ ] No `--wandb`; logging to `runs\hertz12_train.log`
- [ ] Disk free > 200 GB after tokenization; raw cache deleted/transient
- [ ] Launched so it survives terminal close (Start-Process / Scheduled Task)
- [ ] A hard `--max-tokens` cap so it STOPS cleanly, never runs to disk-full

### 3.5 When you're back

1. Check `runs\hertz12_train.log` final val loss/ppl; confirm `final.pt` or latest `step_*.pt`.
2. **G-base gate:** held-out ppl beats Planck 1.3; MMLU-lite (no blobs) > Planck 1.3.
3. Build the blob index (Faiss, CPU) from the corpus — reuse Planck 1.3 pipeline.
4. Validate Wikipedia-blob QA end-to-end; watch **Gate 4a** (utilisation).
5. Then Raum: frozen-Hertz + decomposer-head fine-tune on composition trees.

## 4. Architecture (for reference)

Hertz 1B defaults (`train_hertz.py`, validated against `src/sgs_lm.py`):
- `d_s=256, d_f=5000, n_passes=3, n_heads=4, context_len=512` → ~1.04B params
- bf16 mixed precision, AdamW (betas 0.9/0.95, wd 0.1, fused), cosine + warmup
- torch.compile mode=default (no CUDA graphs — incompatible with grad accum here)
- Checkpoint = model + optimizer + scheduler (~12 GB); rotation keeps last N

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
