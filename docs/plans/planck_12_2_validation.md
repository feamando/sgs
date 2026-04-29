# Planck 1.2.2 — Validation plan

*Status: active. Written 2026-04-29. Depends on the 1.2.2
implementation commit. Source plan: `docs/plans/planck_12_2_plan.md`.
Supersedes 1.2.1's matrix; does NOT re-run the runs that completed
cleanly in 1.2.1 (`shk`).*

## What this validates

Three targeted bug fixes from 1.2.1:

1. `MuonWithAuxAdam` subclasses `Optimizer` (scheduler-compat).
2. `_causal_render_sparse` gather uses flat `index_select` (OOM).
3. tl reweight uses additive floor + batch-mean renormalisation
   (T-pinned-at-1.0 collapse).

Plus the compound gate that was unreachable in 1.2.1 because four of
eight runs crashed before producing any signal.

---

## Harness

Same `scripts/validate_planck12.py`; 1.2.2 adds five `*_fix` run IDs
with the same flags as their 1.2.1 counterparts. `--results-dir
results/planck_12_2` keeps the JSON separable.

The 1.2 `baseline` row is adopted wholesale (unchanged across 1.2.1
and 1.2.2). `ap` is NOT re-adopted — it's diagnostic-only and the
1.2.1 row in `results/planck_12_1/ablation.json` is the reference.

Run order (step dependencies):

1. Smoke tests (§Smoke). Abort on any failure.
2. `sk_fix`, `muon_fix`, `tl_fix` in parallel (per-proposal gates).
3. `all_fix`, `all_plus_fix` (compounds — only after all three
   per-proposal runs land cleanly).

---

## Shared configuration

Identical to 1.2.1. Pinned so the adopted `baseline` row remains a
valid reference:

| Field | Value |
|---|---|
| Dataset | FineWeb-Edu subset |
| Seed | 1337 |
| Arch | d_s=128, d_f=1000, n_heads=4, n_passes=3, ctx 512 |
| Vocab | 32000 |
| Batch size | 32 |
| LR (AdamW) | 3e-4, betas (0.9, 0.95), wd 0.1 |
| LR (Muon)  | 0.02, momentum 0.95 |
| Warmup steps | 1000 |
| Precision | bf16 |
| Grad clip | 1.0 |
| Max steps | 66,750 (token-matched to adopted baseline) |
| Save interval | 10,000 |

NaN fallback (one retry): Muon runs → `--muon-lr 0.01`; non-Muon runs
→ `--batch-size 16`.

---

## Runs

Six rows total; one adopted, five fresh.

| run_id | Label | Extra flags | Origin |
|---|---|---|---|
| `baseline` | Plain CE, 3 passes | *(none)* | **adopted** from 1.2 |
| `sk_fix` | §2.3 w/ `index_select` | `--sparse-k 64` | fresh (Bug 2) |
| `muon_fix` | Muon w/ `Optimizer` subclass | `--optimizer muon` | fresh (Bug 1) |
| `tl_fix` | §2.1 additive floor + renorm | `--transmittance-loss --tl-warmup-steps 5000 --tl-floor-eps 0.05` | fresh (Bug 3) |
| `all_fix` | SGS-native compound | tl_fix + sk_fix flags + `--shared-kernel --shk-schedule mix` | fresh |
| `all_plus_fix` | SGS-native + Muon | all_fix flags + `--optimizer muon` | fresh |

---

## Smoke tests (mandatory precursor)

Run in sequence. A failure at any step aborts the matrix.

### 1. Muon scheduler-compat — 500 steps

```
python scripts/train_lm.py --data-dir data/fineweb \
    --optimizer muon --max-steps 500 \
    --save-dir checkpoints/smoke/muon_1_2_2
```

Must: (a) finish without the `TypeError: MuonWithAuxAdam is not an
Optimizer` crash, (b) log monotonically decreasing 50-step moving
average loss. Wall-clock ~2 min.

### 2. sk forward + backward parity — 500 steps

```
python scripts/train_lm.py --data-dir data/fineweb \
    --max-steps 500 \
    --save-dir checkpoints/smoke/baseline_1_2_2

python scripts/train_lm.py --data-dir data/fineweb \
    --sparse-k 64 --sparse-warmup-steps 0 \
    --max-steps 500 \
    --save-dir checkpoints/smoke/sk_1_2_2
```

Must: (a) `sk_1_2_2` finishes without OOM, (b) loss curves agree
within 0.02 nats at matched log steps (0.01 is the strict target; 0.02
accepts the tiny numerical difference between advanced-indexing and
index_select reductions). Log peak VRAM from
`torch.cuda.max_memory_allocated()` at the end of the sk run; must be
≤ 18 GB at the default config.

### 3. tl T-drift diagnostic — 2,000 steps

```
python scripts/train_lm.py --data-dir data/fineweb \
    --transmittance-loss --tl-warmup-steps 500 --tl-floor-eps 0.05 \
    --max-steps 2000 --log-interval 50 \
    --save-dir checkpoints/smoke/tl_1_2_2
```

Must: (a) `T_mean` at step 2000 is strictly less than 0.95 (i.e. T
has started to drift off the 1.0 attractor after the warmup ends at
step 500). If T stays at 1.0, Bug 3's fix is insufficient and we
escalate per the plan's risk register (instrument raw T_all
distribution before spending the full 66k-step budget).

---

## Per-run sanity checks

Not gates; used to triage regressions.

**`sk_fix` — OOM must not recur.**
- Completes 66,750 steps without OOM.
- Val loss within 0.05 nats of baseline.
- Throughput ≥ 1.25× baseline past step 5,000.
- Peak VRAM ≤ 18 GB.

**`muon_fix` — convergence faster than AdamW.**
- Val loss at step 20,000 ≤ baseline's val loss at step 30,000
  (1.5× sample-eff claim).
- No NaN. On NaN, retry once with `--muon-lr 0.01`.

**`tl_fix` — T drifts off 1.0 and val loss doesn't regress.**
- `T_mean` < 0.5 by step 20,000.
- Val loss at step 20,000 within 0.05 nats of baseline's step-20k
  val loss.
- Final val loss ≤ baseline + 0.05 nats.

**`all_fix` and `all_plus_fix` — compounds don't collapse.**
- Train loss stays within 1.0 nat of val loss throughout (1.2.1's
  `all` run had train 0.07 while val 5.49 — that's the collapse
  signature).
- `T_mean` < 0.5 by step 20,000.
- Sample efficiency ≥ product of per-proposal sample-effs minus
  0.2× slack (if `sk_fix`=1.3×, `muon_fix`=1.5×, `all_plus_fix` ≥
  `1.3·1.5·0.8 = 1.56`).

---

## Gate criteria (either-path, unchanged from 1.2.1)

| compound | sample eff. | wall-clock |
|---|---:|---:|
| `all_fix` | ≥ 1.43× | ≥ 1.8× |
| `muon_fix` | ≥ 1.35× | ≥ 1.0× (no regression) |
| `all_plus_fix` | ≥ 1.55× | ≥ 1.8× |

If ≥1 passes, flip Planck 1.2.1 and 1.2.2 to `done`, unblock Hertz
1.2 with the winning recipe in SETUP.md §6.5.

If none pass, the SGS-native track is declared not-an-accelerator at
the 100M scale. Actions:
- If `muon_fix` clears its own bar (≥ 1.35× sample-eff, no wall-clock
  regression), consider Hertz 1.2 on plain Muon alone.
- Otherwise, keep Hertz 1.2 blocked and open a different accelerator
  investigation (Planck 1.2.3 scope TBD — not predetermined here).

---

## Budget

- `baseline`: adopted, 0h.
- 3 per-proposal runs × ~2.5-3h ≈ 8h.
- 2 compound runs × ~1.7h ≈ 3.5h.
- Smoke tests: ~15 min total.
- 1 NaN retry slack: ~3h.

**Total budget: ~12-15h wall-clock** on the RTX 4090.

---

## Out of scope (unchanged from 1.2.1)

- No seed sweep.
- No new proposals. `ap` stays dropped. FP8, FA-2, Liger all
  deferred.
- No hyperparam tuning of `tl_gamma`, `tl_floor_eps`, or
  `shk-schedule`. 1.2.2 proves the bug fixes; tuning is 1.2.3 work
  only if the gate passes and we want to push it further.

---

## Reporting

On completion, `results/planck_12_2/` contains:

- `ablation.json` (6 rows: baseline + 5 `*_fix`).
- `README.md` with gate verdict, val_loss + tok/s deltas vs baseline
  for each run, and the Hertz 1.2 recommendation.

If any compound gate passes:
- Flip both `Planck 1.2.1` and `Planck 1.2.2` to `done` in
  `roadmap.md`.
- Flip `Hertz 1.2` from blocked to open.
- Record winning recipe in SETUP.md §6.5.
