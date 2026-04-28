# Planck 1.2.1 — Validation plan

*Status: active. Written 2026-04-28. Depends on implementation commit
`48f89d4`. Source plan: `docs/plans/planck_12_1_plan.md`. Supersedes
the compound gate in `docs/plans/planck_12_validation.md`.*

This doc pins the ablation matrix, per-run sanity checks, and the
either-path gate that unblocks Hertz 1.2. The runbook references this
matrix directly.

---

## Harness

`scripts/validate_planck12.py` drives all runs via subprocess. Each
run:

- writes to `checkpoints/planck_12_1/<run_id>/`
- logs to `results/planck_12_1/<run_id>/train_log.txt`
- emits a single-row summary to `results/planck_12_1/ablation.json`

Results live in a new `results/planck_12_1/` sibling so the 1.2 and
1.2.1 tables stay separable. The harness accepts `--results-dir` to
point at the new location without editing the script.

The 1.2 `baseline` row is adopted wholesale (no re-training) via
`--adopt baseline=results/planck_12/baseline/train_log.txt` at the
start of the matrix. Wall-clock and tok/s carry over from the 1.2 JSON.

---

## Shared configuration

Pinned across all runs, identical to 1.2 so the `baseline` row remains
the reference:

| Field | Value |
|---|---|
| Dataset | FineWeb-Edu subset (`data/fineweb/train.bin`) |
| Val split | `data/fineweb/val.bin` |
| Seed | 1337 |
| Arch | d_s=128, d_f=1000, n_heads=4, n_passes=3, context-len=512 |
| Vocab | 32000 |
| Batch size | 32 |
| LR (AdamW) | 3e-4, betas (0.9, 0.95), wd 0.1 |
| LR (Muon)  | 0.02, momentum 0.95 |
| Warmup steps | 1000 |
| Mixed precision | bf16 |
| Grad clip | 1.0 |
| Max steps | 66,750 (token-matched to adopted baseline) |
| Save interval | 10,000 (up from 2,000; disk budget) |
| Log interval | 50 |
| Eval interval | 500 |
| Eval steps | 50 |

Any run that NaNs is re-queued once with a per-run fallback:

- Muon runs: `--muon-lr 0.01` (half the default).
- Non-Muon runs: `--batch-size 16`.

A second failure is logged in the JSON and skipped.

---

## Runs

Ten runs total. Six are re-runs / carry-overs of the 1.2 matrix at the
same labels so we can read regression vs. progress side-by-side. Four
are new.

| run_id | Label | Extra flags | Origin |
|---|---|---|---|
| `baseline` | Plain CE, 3 passes | *(none)* | **adopted** from 1.2 |
| `tl` | §2.1 remediated | `--transmittance-loss --tl-warmup-steps 5000 --tl-floor-eps 0.05` | re-run (1.2 failed) |
| `ap` | §2.2 only | `--adaptive-passes` | **adopted** from 1.2 (inert, diagnostic) |
| `sk` | §2.3 remediated | `--sparse-k 64` | re-run (1.2 OOM; gather rewrite in commit `48f89d4`) |
| `shk` | §2.4 remediated | `--shared-kernel --shk-schedule mix` | re-run (1.2 underperformed) |
| `all` | SGS-native compound (no ap) | `--transmittance-loss --tl-warmup-steps 5000 --tl-floor-eps 0.05 --sparse-k 64 --shared-kernel --shk-schedule mix` | new compound |
| `muon` | Muon only | `--optimizer muon` | new |
| `liger` | Liger only | `--liger` | new |
| `muon_liger` | Muon + Liger | `--optimizer muon --liger` | new |
| `all_plus` | SGS-native `all` + Muon + Liger | `--sparse-k 64 --shared-kernel --shk-schedule mix --optimizer muon --liger` | new (tl dropped, see note) |

**`all_plus` tl exclusion.** Liger + tl are mutually exclusive (tl
needs per-token CE on materialised logits; Liger fuses the logits
away). `all_plus` therefore drops `--transmittance-loss`. If tl carries
the SGS-native track, that signal shows up in the `all` run, not here.

**`all` changes vs. 1.2.** Drops `--adaptive-passes` (inert in 1.2);
adds `--shk-schedule mix` to avoid the always-shared quality regression;
adds `--tl-warmup-steps 5000 --tl-floor-eps 0.05`.

---

## Metrics per run

Same columns as 1.2, pulled from stdout + final checkpoint:

1. **Final val loss** on `val.bin` (50 steps, fixed order).
2. **Final val perplexity** = `exp(val_loss)`.
3. **Wall-clock** from first training step to final eval.
4. **Tokens seen** = `global_step * batch_size * context-len`.
5. **tok/s throughput** = mean over last 50% of log-interval samples.
6. **Passes EMA** (meaningful for `ap` only).
7. **Mean T_diag** at final 500 steps (for runs exposing T).
8. **Peak VRAM** via `torch.cuda.max_memory_allocated`.

Post-hoc:

- **Tokens-to-target**: first step where val loss ≤ baseline's val
  loss at step 66,750 (i.e. 4.10). For runs that clear it mid-training
  we use the first crossing; for runs that never clear it the value
  is `inf`.
- **Wall-clock speedup** = `baseline.wall_clock / run.wall_clock`.
- **Sample efficiency** = `baseline.tokens_to_target / run.tokens_to_target`.

---

## Per-run sanity checks

*Not gates; used to triage regressions before re-running the whole
matrix. Columns "sanity OK?" populate the results README.*

**`tl` — must not collapse this time.**
- `T_mean` drifts below `T_max = 0.3` by step 10,000.
- Val loss at step 20,000 within 0.05 nats of baseline's step-20,000
  val loss. (Parity check: remediation should not regress quality.)
- Val loss at step 66,750 ≤ baseline + 0.05 nats. If worse, `tl` is
  still broken → drop from `all`, re-evaluate with Planck 1.2.2.

**`sk` — OOM must not recur.**
- No OOM through 66,750 steps at `B=32, sparse-k=64, d_f=1000`.
- Val loss within 0.05 nats of baseline (sparse-k is a compute trick).
- Throughput ≥ 1.25× baseline post step-5000 warmup. If < 1.15×,
  demote to `--sparse-k 32` or drop from `all`.

**`shk` — schedule must outperform `always`.**
- Val loss within 0.10 nats of baseline (up from the 1.2 spec's 0.05;
  `mix` still shares some kernels).
- Throughput ≥ 1.08× baseline (soft; the win here is quality recovery,
  not throughput).

**`ap` — carry the 1.2 row forward unchanged.** Diagnostic only.

**`muon` — convergence faster than AdamW.**
- Val loss at step 20,000 ≤ baseline's val loss at step 30,000
  (1.5× sample-efficiency claim validated mid-run, not at the end).
- No NaN divergence. If it NaNs, fall back to `--muon-lr 0.01` once
  then flag.

**`liger` — forward-parity on smoke test, then throughput.**
- Startup smoke test: Liger loss matches `F.cross_entropy` within 1e-3
  on a random `[32, 512, 1000] @ [32000, 1000]` tensor pair. (Smoke
  test lives in `scripts/validate_planck12.py`, runs before the
  matrix; failure aborts the run.)
- Throughput ≥ 1.15× baseline.
- Val loss within 0.05 nats of baseline (numerical parity).

**`muon_liger` — composition is additive, not interacting.**
- Sample efficiency within 0.1× of `muon` (Liger should not disturb
  convergence).
- Wall-clock speedup ≥ `liger`'s (Muon adds a small Newton-Schulz
  cost per step, but the AdamW→Muon swap on embeddings isn't a
  throughput regression).

**`all` and `all_plus` — no hidden interaction.**
- If the compound `all` speedup is substantially below the product of
  individual speedups, flag it and inspect the sk+shk pair (both
  touch the kernel).
- `all_plus` should land between `all` and `muon_liger` on both axes
  (sample eff from the SGS-native track, wall-clock from the hedge).

---

## Gate criteria (either-path)

**Compound gate (Planck 1.2.1 → Hertz 1.2 unblock):** at least one of
the three compounds below must clear its thresholds on both axes.

| compound | sample eff. | wall-clock | notes |
|---|---:|---:|---|
| `all` | ≥ 1.43× | ≥ 1.8× | SGS-native thesis, original 1.2 target |
| `muon_liger` | ≥ 1.30× | ≥ 1.7× | industry-standard hedge (Muon's claimed 1.5× is our only prior) |
| `all_plus` | ≥ 1.50× | ≥ 1.9× | stacked (eligible only if both above pass) |

If `all_plus` passes, it becomes the Hertz 1.2 recipe. Otherwise the
winning compound carries.

**If none pass:**
- The worst per-run regression in each compound is demoted (e.g. drop
  `shk` from `all` if `shk`'s quality cost exceeds 0.10 nats).
- Planck 1.2.2 is opened to fix the specific proposal; Hertz 1.2 stays
  blocked.

---

## Smoke test (precursor to the matrix)

Before queuing the full matrix on the Windows box:

1. **sk gather parity.** Run `baseline` and `sk` for 500 steps with
   the same seed. Loss curves must agree within 0.01 nats at every
   log point up to step 500. The sk path now goes through advanced
   indexing; the forward pass should be bitwise identical on the
   non-sparse branch and numerically identical on the sparse branch
   (since the gather-vs-indexing rewrite is a pure algebraic
   simplification).
2. **Liger forward parity.** See `liger` sanity above.
3. **Muon stability.** Run `muon` for 500 steps; confirm no NaN and
   loss is monotonically decreasing on the moving average.

A failure at any of these aborts the full-matrix run and files a bug.

---

## Budget

- Baseline: **adopted**, 0h.
- 9 new 66,750-step runs × ~3h each on RTX 4090 bf16 = ~21h GPU time.
- `liger` and `muon_liger` may run slightly faster (~2.5h) if the
  +15% wall-clock claim holds; `all_plus` slightly faster still.
- Allow 2h slack for smoke tests + one NaN-retry budget.

**Total budget: ~25h wall-clock** on the Windows box. User has
confirmed ±21h is acceptable; the slack absorbs one retry.

---

## Out of scope

- No seed sweep. Single seed 1337 across all runs; wins < 5% are noise.
- No per-run hyperparameter tuning beyond the 1.2.1 plan's specified
  defaults. Tuning is Planck 1.2.2 work.
- FP8, FA-2, speculative rollout: deferred. See plan §5.
- Fixing `ap`: deferred to Planck 1.2.2 as a standalone probe.

---

## Reporting

On completion, `results/planck_12_1/` contains:

- `ablation.json` (10 rows).
- `README.md` summarising gate pass/fail, one table comparing all
  runs, any unexpected per-run behaviour, and the Hertz 1.2
  recommendation.

If any compound gate passes, `roadmap.md` flips Planck 1.2.1 → done
and Hertz 1.2 unblocks with the winning recipe recorded in SETUP.md
§6.5.
