# Planck 1.2 — Ablation results

*Completed 2026-04-28. Source: `ablation.json` (this directory).*

See:
- `docs/plans/planck_12_plan.md` — design
- `docs/plans/planck_12_validation.md` — ablation matrix + gate
- `docs/plans/planck_12_runbook.md` — how to run

Each run's stdout is teed to `<run_id>/train_log.txt`. The aggregate
JSON at `ablation.json` is the source of truth for the summary table.

## Summary table

Token-matched to the adopted baseline at 66,750 steps (~1.09 B FineWeb-Edu tokens, batch 32, L=512, seed 1337, RTX 4090, bf16).

| run | label | status | val loss | ppl | tok/s | wall (s) | speedup |
|-----|-------|--------|---------:|----:|------:|---------:|--------:|
| baseline | Plain CE, 3 passes | adopted | 4.1031 | 60.5 | 97,394 | 11,229 | 1.00× |
| tl | §2.1 only | ok | **8.5481** | 5156.9 | 97,944 | 11,537 | 0.97× |
| ap | §2.2 only | ok | 4.0674 | 58.4 | 97,388 | 11,608 | 0.97× |
| sk | §2.3 only | exit_1 (OOM) | 5.0877 @ step 10,100 | 162.0 | 97,122 | 1,867 | — |
| shk | §2.4 only | ok | 4.3359 | 76.4 | 107,019 | 10,562 | 1.06× |
| all | all four composed | ok | **8.546** | 5146.2 | 107,346 | 10,527 | 1.07× |

## Gate

Compound `all` vs. `baseline`:
- Sample efficiency ≥ 1.43× (same val-loss target on ≤70% tokens): **FAIL** (compound val loss 2× worse than baseline)
- Wall-clock speedup ≥ 1.8×: **FAIL** (1.07×)

**Verdict: FAIL.** Planck 1.2 → Hertz 1.2 gate is not met. Hertz 1.2 stays blocked; remediation work tracked as Planck 1.2.1.

## Per-proposal notes

**§2.1 tl — catastrophic.** `T_mean_final = 1.0` all 66k steps. The
`(1-T)^γ` weighting multiplies CE by ~0, so the effective training
signal is the `max(0, T - T_max)²` penalty alone. Transmittance never
moves because CE is no longer pushing it. Collapse to val loss 8.55 is
the expected endpoint of that dynamic. Fix direction: either warmup on
plain CE until T stabilises below T_max before engaging the reweighting,
or use `(1 - T * 0.99).pow(γ)` to preserve a gradient floor.

**§2.2 ap — inert.** `passes_ema_final = 3.0`; adaptive-passes never
fired once across 66k steps. Val loss landed 0.04 nats under baseline
(likely seed noise). No throughput gain because every batch still ran
all three passes. Either the per-batch gate (`T < 1 - eps`) is never
true with `T_mean ≈ 0.994` and `eps = 0.02`, or the early-exit condition
is masked by the `--ap-min-step 2000` warmup in a way we didn't
anticipate. Diagnostic only; no lever here until the gate fires.

**§2.3 sk — implementation bug (OOM).** Died at step 10,100 with a
15.62 GiB allocation inside `torch.gather(feat_exp, 2, idx_exp)` at
`src/sgs_lm.py:325`. At `B=32, L=512, k=64, d_f=1000`, a correct sparse
path should materialise `[B, L, k, d_f] ≈ 4.0 GiB bf16`, not 15.62 GiB.
Likely `feat_exp` is being broadcast to `[B, L, L, d_f]` before gather
rather than using an indexed lookup into `[B, L, d_f]`. The up-to-10k
loss curve is strictly worse than baseline (5.09 vs baseline's ~5.2 at
matched step, so within noise), so there's no signal on the quality side
yet. Fix direction: rewrite the gather so `feat` is indexed per-position
rather than expanded, then re-run.

**§2.4 shk — works but underperforms both gates.**
- Throughput: +9.9% (target was ≥ 20%).
- Quality: val loss +0.23 nats vs baseline (target was within 0.05 nats).
- Saves compute but at a real quality cost. Plausibly the shared-kernel
  assumption is too aggressive at d_s = 128 (fine-grained passes need
  distinct kernels). Fix direction: tune the sharing weight or mix
  shared + per-pass kernels on a schedule.

**`all` — inherits the tl collapse.** 1.07× throughput is the sum of
`shk` + a sliver of `sk` headroom; the val loss just tracks `tl`'s
collapse. No additivity audit is meaningful until `tl` produces a
non-degenerate training run.

## Additivity audit

Not meaningful this matrix. `all` is dominated by `tl`'s collapse, so
the product-of-individual-speedups cross-check (the matrix we wanted to
use to flag interactions) is overwhelmed by a single broken proposal.
Revisit after Planck 1.2.1.

## Recommendation

1. Flip `Planck 1.2` to `done` in `roadmap.md` as **gate failed, proposals diagnostic only** (not a pass, but the work is done).
2. Open `Planck 1.2.1` tracking targeted fixes: `tl` reweight floor, `sk` gather rewrite, `shk` sharing-weight tune. Drop `ap` from the compound until it fires on its own.
3. `Hertz 1.2` stays `open` and blocked on `Planck 1.2.1`. The §2.1 + §2.3 gate predicate for Hertz 1.2 from SETUP.md §6.5 is not yet satisfied.
