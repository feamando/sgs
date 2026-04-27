# Planck 1.2 — Validation plan

*Status: active. Written 2026-04-27. Depends on implementation commit
`ce195c8`. Source plan: `docs/plans/planck_12_plan.md`.*

This doc pins the exact configurations, metrics, gates, and failure
modes for the Planck 1.2 acceleration-recipe ablation. The runbook
(separate doc) will reference this matrix directly.

---

## Harness

`scripts/validate_planck12.py` drives the six runs via subprocess calls
to `scripts/train_lm.py`. Each run:

- writes to `checkpoints/planck_12/<run_id>/`
- logs to `results/planck_12/<run_id>/train_log.txt` (stdout tee)
- emits a single-row summary to `results/planck_12/ablation.json`
  on completion

The JSON accumulates across runs so partial results are preserved if a
run crashes. Each row contains: run_id, flags, seed, tokens_seen,
wall_clock_s, final_val_loss, final_val_ppl, tok_per_sec, passes_ema,
T_mean, peak_vram_mb.

---

## Shared configuration

Pinned across all six runs:

| Field | Value |
|---|---|
| Dataset | FineWeb-Edu subset (`data/fineweb/train.bin` head 500M tokens) |
| Val split | `data/fineweb/val.bin` (~44M tokens) |
| Seed | 1337 |
| Arch | d_s=128, d_f=1000, n_heads=4, n_passes=3, context-len=512 |
| Vocab | 32000 (same SentencePiece as Planck 1.1) |
| Batch size | 32 |
| LR | 3e-4, AdamW betas (0.9, 0.95), weight-decay 0.1 |
| Warmup steps | 1000 |
| Mixed precision | bf16 |
| Grad clip | 1.0 |
| Epochs | computed so total tokens ≈ 500M |
| Log interval | 50 |
| Eval interval | 500 |
| Eval steps | 50 |

Any ablation that diverges NaN or OOMs is re-queued with `--batch-size 16`
once; a second failure is logged in the JSON and skipped.

---

## Six runs

| run_id | Label | Extra flags |
|---|---|---|
| `baseline` | Plain CE, 3 passes | *(none)* |
| `tl` | §2.1 only | `--transmittance-loss` |
| `ap` | §2.2 only | `--adaptive-passes` |
| `sk` | §2.3 only | `--sparse-k 64` |
| `shk` | §2.4 only | `--shared-kernel` |
| `all` | All four composed | all flags above |

Default hyperparameters (`tl-gamma=1.5`, `tl-lambda=0.01`, `tl-tmax=0.3`,
`ap-eps=0.02`, `ap-min-step=2000`, `sparse-warmup-steps=5000`,
`sparse-tau-gate=30.0`) from the implementation plan. Not tuned per-run
in this matrix; tuning is reserved for Planck 1.2.1 if an individual
proposal underperforms.

---

## Metrics logged per run

Pulled from the stdout log + final checkpoint:

1. **Final val loss** on `val.bin` (first 50 steps, fixed order).
2. **Final val perplexity** = `exp(val_loss)`.
3. **Wall-clock** from first training step to final eval.
4. **Tokens seen** = `global_step * batch_size * context-len`.
5. **tok/s throughput** = mean of last 50% of log-interval samples.
6. **Passes EMA** (only meaningful for `ap`, `all`).
7. **Mean T_diag** at final 500 steps (only for runs where T is exposed).
8. **Peak VRAM** via `torch.cuda.max_memory_allocated`.

Also computed post-hoc:

- **Tokens-to-target**: first step where val loss ≤ 3.80 (Planck 1.1
  crossed this at ~1.2B TinyStories tokens; for FineWeb-Edu the same
  target should hit later, so this number is interpreted *relative to
  baseline*, not as an absolute).
- **Wall-clock speedup** = `baseline.wall_clock / run.wall_clock`.
- **Sample efficiency** = `baseline.tokens_to_target / run.tokens_to_target`.

---

## Gate criteria

**Compound gate (Planck 1.2 → Hertz 1.2):** the `all` run must hit

1. Sample efficiency ≥ 1.43× baseline (same val-loss target on ≤70%
   of the tokens).
2. Wall-clock speedup ≥ 1.8× baseline.

Either miss → revisit individual proposals using the per-flag numbers.

**Per-proposal sanity checks** (not gates, diagnostic only):

- `tl` should match or slightly improve val loss at equal tokens;
  throughput unchanged (loss reshape only).
- `ap` should show `passes_ema` < 3.0 past step 2000 and improve
  tok/s by ≥15%.
- `sk` should match val loss within 0.05 nats of baseline and improve
  tok/s by ≥25% (once past warmup at step 5000).
- `shk` should match val loss within 0.05 nats of baseline and improve
  tok/s by ≥20%.

If any single proposal catastrophically breaks (val loss > 2× baseline),
flag it in the results JSON and exclude it from the compound rerun.

---

## Isolation checks

We rely on the compound number as the primary gate, but the per-flag
runs serve two diagnostic roles:

- **Numerical baseline.** The `baseline` run on FineWeb-Edu establishes
  the new val-loss floor. All Planck 1.1 numbers were TinyStories and
  don't transfer.
- **Additivity audit.** If the compound `all` speedup ≠ approx. product
  of individual speedups, something is interacting. The validation
  report notes the discrepancy and the most likely suspect
  (§2.3 + §2.4 stacking is the highest-risk pair: both touch the
  kernel).

---

## Backward-compat smoke test (precursor to the ablation)

Before running the six-config matrix:

1. Run `baseline` for 500 steps.
2. Checkpoint the loss curve.
3. Check-in validation: loss at step 500 should match the
   Planck 1.1 loss curve on the *same* FineWeb-Edu subset to ±0.02
   nats (accounting for dataset noise). If it doesn't, the accel
   implementation has broken the default path — fix before proceeding.

The implementation commit has a CPU smoke test covering return shapes,
but numerical drift only shows up at scale.

---

## Out of scope

- No seed sweep. Single seed per run; accept wins <5% are noise.
- No hyperparameter tuning of `tl-gamma`, `sparse-k`, etc. within the
  matrix. Tuning is Planck 1.2.1 work, gated on the matrix result.
- No curriculum (§2.5) — deferred to Hertz 2.x.
- No new evals beyond val loss/ppl. Downstream task eval (HellaSwag,
  ARC-e) is Hertz 1.2 work.

---

## Reporting

On completion, `results/planck_12/ablation.json` is committed alongside
a short `results/planck_12/README.md` summarising:

- Gate pass/fail.
- One table comparing all six runs on the metrics above.
- Any unexpected per-proposal behaviour.
- Recommendation to proceed with Hertz 1.2, retune, or drop a specific
  proposal.

If the compound gate passes, `roadmap.md` flips Planck 1.2 → done and
Hertz 1.2 unblocks.
