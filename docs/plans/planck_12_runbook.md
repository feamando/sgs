# Planck 1.2 — Runbook

*Status: active. Written 2026-04-27. Pairs with
`planck_12_plan.md` (design) and `planck_12_validation.md` (ablation
matrix).*

Concrete commands for reproducing the Planck 1.2 ablation. All paths
relative to the repo root.

---

## 0. Prereqs

```
# Pre-flight checks
ls -lh data/fineweb/train.bin    # expect ~17 GB
ls -lh data/fineweb/val.bin      # expect ~174 MB

# Quick CPU smoke: default path still returns bare logits
python -c "
import torch
from src.sgs_lm import SGSLanguageModel
m = SGSLanguageModel(vocab_size=256, d_s=16, d_f=32, n_heads=2, n_passes=3, max_len=16)
y = m(torch.zeros(1, 8, dtype=torch.long))
print('OK' if isinstance(y, torch.Tensor) else 'FAIL', y.shape)
"
```

If the smoke fails, do not proceed — the implementation's backward-
compat guarantee is broken.

---

## 1. Full six-run matrix

```
python scripts/validate_planck12.py --data-dir data/fineweb --wandb
```

Each run is a subprocess call to `scripts/train_lm.py` with a distinct
flag combo (see `docs/plans/planck_12_validation.md` for the exact
matrix). Results accumulate in `results/planck_12/ablation.json`;
per-run stdout is teed to `results/planck_12/<run_id>/train_log.txt`.

Expected wall clock on an RTX 4090 at the defaults (1 epoch over
500M FineWeb-Edu tokens, batch 32, L=512):

| run | est. wall |
|---|---|
| baseline | 6.5 h |
| tl / ap / shk | 6.0–6.5 h |
| sk | 4.8 h |
| all | 3.6 h |

If the budget is too big, drop the FineWeb-Edu subset to 250M tokens
(edit the first N records of `train.bin` through a smaller `--epochs`,
or pre-slice the bin). All conclusions scale since we compare ratios,
not absolute numbers.

---

## 2. Individual runs (for iteration / re-runs)

```
# Just baseline
python scripts/validate_planck12.py --data-dir data/fineweb --only baseline

# Re-run a single config after a crash
python scripts/validate_planck12.py --data-dir data/fineweb --only sk --force
```

The harness skips any run already marked `status: ok` in
`ablation.json` unless `--force` is passed.

---

## 3. Profiling a hot step

If throughput numbers look off, grab a single-step torch.profiler dump
inside any of the configs:

```
python scripts/train_lm.py --data-dir data/fineweb \
    --sparse-k 64 --profile-step 3000 --epochs 1
```

The profiler dumps the top 20 CUDA ops at step 3000 to stdout. Use
this to confirm whether top-k selection is a meaningful slice of step
time (plan §Resolved has the threshold: flag if >25%).

---

## 4. Inspecting results

```
cat results/planck_12/ablation.json
```

Summary table is printed at the end of every harness invocation. To
rebuild it from an existing JSON without running anything:

```
python scripts/validate_planck12.py --data-dir data/fineweb --dry-run
# (dry-run still prints the summary of runs already in the JSON)
```

---

## 5. Publishing results

1. `git add results/planck_12/ablation.json results/planck_12/README.md`
2. Commit with message body covering: gate pass/fail, per-run val
   loss + tok/s deltas, any anomalies.
3. If compound gate passed: flip `Planck 1.2` to `done` in
   `roadmap.md` and unblock `Hertz 1.2`.
4. If it failed: keep `Planck 1.2` `in progress`, open a `1.2.1` row
   describing the remediation (retune the losing proposal, or drop it).

---

## 6. Gotchas

- **FineWeb-Edu val split**: if `data/fineweb/val.bin` is missing,
  `src/tinystories.py` will error before training starts. Carve a
  held-out split from the tail of `train.bin` before running.
- **Sparsity warmup interacts with short runs**: if total training
  steps < `--sparse-warmup-steps` (default 5000), the `sk` and `all`
  runs never enter the sparse path. Ensure `total_steps > 5000`, or
  lower `--sparse-warmup-steps`.
- **Adaptive exit on short runs**: `--ap-min-step` defaults to 2000.
  If the full run is 500 steps, adaptive-passes never triggers.
  Lower `--ap-min-step` for debug runs.
- **Numerical drift vs. pre-1.2 main**: the default code path should
  be numerically identical to main. If baseline val loss drifts >0.02
  nats vs. the last Planck 1.1 FineWeb-Edu baseline, audit the
  default path before trusting the accel deltas.
- **Checkpoint bloat**: the harness writes a checkpoint to
  `checkpoints/planck_12/<run_id>/` per save-interval. Six runs × the
  default 2000-step save-interval can fill disk on a 500M-token run.
  Either pass `--save-interval <big>` or clean up with the existing
  `scripts/cleanup_planck11.py` pattern between runs.
