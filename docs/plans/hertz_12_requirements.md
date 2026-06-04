# Hertz 1.2 — Requirements & Unattended-Run Kickoff

**Radiance Labs, 2026-06-04**
**Goal:** define what Hertz 1.2 (SGS LM at ~1B params) needs, and specify a run
that can be **kicked off before vacation and left unattended** on the single
RTX 4090 (Windows, 1 TB drive).

Hertz is also the model Raum will use (frozen encoder + learned decomposer head,
per the [[project_sgs_raum_1model_pivot]] 1-model design). That dual role shapes
the answers below.

Confidence tags: **CL1** measured in-repo, **CL2** strong external, **CL3**
reasoned estimate, **CL4** speculative.

## TL;DR recommendations (the five concerns)

1. **Training data:** text-only open mix — FineWeb-Edu (quality web) + Wikipedia
   (reuse the Planck 1.3 pipeline) + a code slice (The Stack v2 / StarCoder).
   Reproducible by pinned revision. **Not** 3D/multimodal data.
2. **Multimodal now? No.** Train a strong **text-only** base. Raum consumes it as
   a frozen encoder + a small decomposer head fine-tuned on composition trees.
   Multimodality is a later adapter, not the base pretrain. (De-risks exactly
   like 0.5→1.5 and the 1.7 staging did.)
3. **Space (1 TB):** the current save code is unbounded and **will fill the disk**.
   Add checkpoint rotation (keep last 2 full + best + bf16 milestones), bf16
   milestones, capped token budget, streamed corpus. Budget table below.
4. **Model density:** capability per parameter. Raise it via data quality,
   tokens-per-param (train the 1B long), optional distillation, and the
   SGS-native lever — **blobs add knowledge without adding parameters**, which
   is density by construction.
5. **Another blobs run? Yes, but decoupled.** Blobs are built *post-pretrain*
   from the same corpus (a Faiss index, no GPU retrain). Kick off the base
   pretrain now; build/eval blobs when you're back. Carry forward the Planck 1.3
   Gate-4a fix.

---

## 1. Training data

**Recommendation: a pinned, reproducible text-only mix.** Hertz is the
"open corpus" tier (vs Helmholtz = multilingual+code, Einstein = frontier per
the deck). For a 1B model that must (a) be a competent base and (b) serve Raum's
decomposer, the mix should be:

| Source | Share | Why | Conf |
|--------|------|-----|------|
| **FineWeb-Edu** (HF) | ~60% | High-quality filtered web; best quality-per-token for small models | CL2 |
| **Wikipedia** (`wikimedia/wikipedia`, pinned rev) | ~25% | Encyclopedic priors; **reuses the Planck 1.3 ingest pipeline** and doubles as the blob index | CL1 (pipeline exists) |
| **Code** (The Stack v2 / StarCoder data, permissive licenses only) | ~15% | Raum emits structured JSON trees; code/structured data improves structured-output reliability | CL3 |

Notes:
- **Tokenizer:** Planck's 32K SentencePiece was trained on Wikipedia. A code +
  open-web mix wants a code-aware tokenizer. **Decision needed** (see questions):
  reuse the 32K SP (fast, lower structured-output quality) vs train a new ~48-64K
  tokenizer on the mix (half-day, better, but invalidates Planck-1.x checkpoint
  reuse). Recommend a **new tokenizer** since Hertz is a fresh scale.
- **Reproducibility:** pin each source by revision into
  `data/hertz/manifest.json` (HF dataset ids + revisions + mix weights + seed),
  per the existing snapshot-id convention.
- **No 3D/tree data in the base.** There is no large (text, composition-tree)
  corpus (the concept article states this). Raum's trees (3,450 today) are a
  **fine-tune** set, not pretrain. Mixing them into base pretrain would waste the
  budget and underfit both objectives.

## 2. Multimodal now, since Raum uses Hertz?

**Recommendation: No. Text-only base; Raum as a downstream head.**

Reasoning:
- The 1-model Raum pivot already settled this: Raum = **frozen Planck/Hertz
  encoder + a small learned bridge/decomposer**, not a jointly-trained
  multimodal base. (CL1, [[project_sgs_raum_1model_pivot]])
- Joint multimodal pretrain couples two unsolved problems (language + structured
  3D) and needs paired data that **does not exist at scale** (CL1, concept
  article §6). Risk explodes on an unattended run.
- The proven de-risking pattern in this project is *sequence, don't couple*
  (0.5 de-risked 1.5; 1.7 staged renderer→proportions→head). Same here: strong
  text base first, then a 3D adapter.
- The architecture still earns the "one substrate" pitch: the same
  alpha-compositing math (proven ⊋ softmax) renders Raum scenes and runs Hertz
  attention. Multimodality is added as a **fine-tuned decomposer head /
  adapter** on the frozen base, cheaply, after pretrain.

So Hertz 1.2 = text LM. Raum 2.x = Hertz-frozen + decomposer head fine-tune.
Keep the encoder's blob/retrieval interface clean so the head drops in.

## 3. Space — fitting an unattended 1B run in 1 TB

**This is the run-killer if unaddressed.** The current `train_planck11.py:_save`
writes `step_{global_step}.pt` every `save_interval` with **no rotation**, and
each checkpoint is model + AdamW optimizer + scheduler.

**Checkpoint size math (1B params, CL3):**
- Model weights fp32: ~4 GB (bf16: ~2 GB)
- AdamW optimizer state (m + v, fp32): ~8 GB
- **Full resumable checkpoint ≈ 12 GB.** bf16 weights-only milestone ≈ 2 GB.

At, say, a save every 2k steps over a multi-day run that's dozens of 12 GB files
→ **hundreds of GB to >1 TB. The disk fills and the run crashes.**

**Required changes before kicking off (code, ~1-2 hrs):**

| Lever | Policy | Saves |
|-------|--------|-------|
| **Rotate full checkpoints** | keep only last **2** `step_*.pt` (for resume) | caps optimizer-state files at ~24 GB |
| **Milestone = weights-only, bf16** | every N steps save model-only bf16 (~2 GB), no optimizer | milestones cost 2 GB not 12 GB |
| **Keep `best.pt` + `final.pt`** | weights-only bf16 | ~4 GB |
| **Don't commit checkpoints** | already covered: `.gitignore` has `*.pt` (so the auto-commit feedback won't push 12 GB blobs) | already safe |

**Budget table (target: stay < 1 TB with headroom, CL3):**

| Item | Size |
|------|------|
| Tokenized corpus (streamed/sharded, ~10-14B tokens as `.bin`) | ~25-60 GB |
| Raw corpus cache (delete after tokenizing) | transient, ~100-200 GB peak |
| Rolling full checkpoints (2 × 12 GB) | ~24 GB |
| bf16 milestones (say 10 × 2 GB) | ~20 GB |
| best + final | ~4 GB |
| Logs / eval artifacts | < 5 GB |
| **Working total (post-tokenize)** | **~80-115 GB** |

The transient raw-corpus cache is the real squeeze: **tokenize-then-delete in
shards**, never hold the full raw + tokenized corpus simultaneously. Cap the
token budget (below) so the `.bin` is bounded.

## 4. Model density — what it is, how to raise it

**Definition (CL2):** *capability density* = capability per parameter (Densing
Law, Tsinghua 2024: the params needed to hit a fixed capability roughly halves
every few months via better data/recipe/architecture). For us: get more
capability out of 1B params rather than chasing more params we can't afford to
train on one 4090.

**Levers, in order of value for our constraints:**

1. **Data quality** — FineWeb-Edu over raw web is the single biggest density
   lever for small models. (CL2)
2. **Tokens-per-parameter** — train the 1B *long*. Chinchilla-optimal ≈ 20
   tokens/param (~20B); a "small model trained long" past that still gains.
   Our budget is compute-bound (below), so push tokens as far as the vacation
   window allows. (CL2)
3. **SGS-native: blobs add knowledge without parameters.** Retrieval-by-
   construction means effective capability rises without growing the weights —
   density by construction, the SGS differentiator. (CL1 concept, CL3 at 1B)
4. **Distillation (optional, later)** — logits from a larger open teacher
   (e.g. a 7B) raise density per param, but add pipeline complexity; not for the
   unattended run. (CL3)
5. **Recipe** — plain AdamW (Muon is a confirmed regression on our mixed-param
   landscape; do **not** retry — [[project_sgs_accel_shelved]]). (CL1)

**Note on the SGS blob-count reading:** if by "density" you meant Gaussian/blob
count in the model (the 50k→500k blob sweep), that's a separate axis handled by
**progressive blob scheduling** (start k=10k, grow) — both a memory-relief and a
convergence lever (accel v2 §Phase 2). I've addressed it there, not as the
primary "density" meaning.

## 5. Another blobs run?

**Recommendation: yes, but decoupled from the kicked-off pretrain.**

- Blobs are the built-in RAG and Hertz's differentiator (deck: "index by
  construction, grows without retraining, frequency-weighted"). Worth doing.
- **But the blob index is built *after* pretrain**, from the same corpus, as a
  Faiss index — **no GPU retrain**. So it is *not* part of the long unattended
  run; it's a cheap CPU/IO step you do when back.
- Carry forward the **Planck 1.3 lesson**: validate the static (Wikipedia) blob
  index end-to-end before layering live/RSS, and watch **Gate 4a** (intra-sample
  utilisation, which FAILED on Planck 1.1 per the deck). Hertz should reuse the
  1.3 fix (top-k + transmittance sweep, frozen-base retrain for clean Gate 1).
- **For the unattended run:** the safest choice is **progressive blob schedule
  starting low (k=10k)** so the render passes fit memory; do not start at
  k=200k (OOMs per accel v2). Full blob-count sweep happens supervised, later.

---

## Kickoff runbook (the part you start before leaving)

**What gets kicked off:** the **base text pretrain only** — the long,
GPU-bound, unattended-friendly phase. Tokenizer, blob index, Raum head, and
multimodal work are all *not* in this run.

### Pre-flight (do NOT skip — an unattended run that OOMs on hour 2 wastes the trip)

```powershell
# 0. Land the checkpoint-rotation + bf16-milestone change in train script first
#    (Section 3). Without it the disk fills mid-run.

# 1. Build the pinned corpus manifest, tokenize to sharded .bin, DELETE raw cache
python scripts\build_hertz_corpus.py `
  --manifest data\hertz\manifest.json `
  --out data\hertz\tokens `
  --token-budget 12e9 `
  --shard-and-delete-raw

# 2. SMOKE TEST: 1000 steps at the real config. Confirm:
#    - no OOM at chosen (batch, k_blobs, grad-checkpoint, fp8-encoder)
#    - throughput (tok/s) -> sanity-check the token budget vs days available
#    - disk delta over 1000 steps -> extrapolate to full run, confirm < 1 TB
#    - a resume from the rolling checkpoint works
python scripts\train_hertz.py --config configs\hertz_12.yaml --max-steps 1000 --smoke
```

### Memory config (REQUIRED on the 4090 or it OOMs — accel v2 Phase 1)

- gradient checkpointing on the **render passes** (not the encoder)
- **FP8 on the transformer encoder**, BF16 on Gaussian output params
- **progressive blob schedule:** k start 10k → grow to target over first ~30%
- gradient accumulation to hit effective batch within 22 GB usable VRAM

### Token budget vs vacation window (CL3, calibrate with the smoke-test tok/s)

- 4090 SGS throughput observed ~1.7k–11.8k tok/s, compile-off ([[project_sgs_pivot_2026_04_20]]).
- At ~11.8k tok/s ≈ ~1B tokens/day; at the low end far less. **The smoke test's
  measured tok/s decides the budget.**
- Recommended target: **10–14B tokens** (≈ Chinchilla-ish for 1B, fits a ~2-week
  trip at the optimistic rate). Set `--token-budget` so the run *finishes or
  checkpoints cleanly* within the window; resumable either way.

### Unattended-safety checklist

- [ ] Checkpoint rotation live (keep-last-2 full + bf16 milestones) — Section 3
      (note: `*.pt` is already gitignored, so the disk-fill risk is rotation, not git)
- [ ] No `--wandb` (paid; default stdout per [[feedback_sgs_wandb_default]]) — log to file
- [ ] Smoke test passed: no OOM, tok/s measured, disk delta extrapolated < 1 TB, resume works
- [ ] Run under a process that survives terminal close (e.g. nohup-equiv / scheduled task), logs to `runs\hertz_12\train.log`
- [ ] A hard `--max-steps` / token cap so it stops cleanly, not at disk-full

## Gates

| Gate | Pass condition |
|------|----------------|
| **G-smoke** | 1k steps, no OOM, resume works, disk extrapolation < 1 TB |
| **G-base** | held-out perplexity beats Planck 1.3 base; MMLU-lite (no blobs) > Planck 1.3 |
| **G-blobs** (post-run) | Wikipedia-blob QA validated end-to-end; Gate-4a utilisation fixed |
| **G-raum** (later) | frozen-Hertz + decomposer head emits valid trees beyond trained categories (Raum G1) |

## What is NOT in this run (deliberately deferred)

- Multimodal / 3D-paired pretrain (→ Raum decomposer head, later)
- Blob index build + QA (→ post-pretrain, CPU/IO)
- Full blob-count sweep 50k→500k (→ supervised, later)
- Distillation from a teacher (→ optional density lever, later)
- Muon / compound accel recipes (→ confirmed regressions, do not retry)

## Open decisions (lock before kicking off)

1. **Tokenizer:** reuse Planck 32K SP, or train new ~48-64K on the mix? (recommend new)
2. **Multimodal base:** confirm text-only (recommend yes, text-only).
3. **Token budget / stop condition** for the vacation window.
4. **Code in the mix:** include the ~15% code slice, or pure text? (recommend include)
