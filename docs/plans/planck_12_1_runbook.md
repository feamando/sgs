# Planck 1.2.1 — Runbook

*Status: active. Written 2026-04-28. Pairs with
`planck_12_1_plan.md` (design) and `planck_12_1_validation.md`
(ablation matrix). Implementation commit: `48f89d4`.*

Concrete commands for running the 10-run 1.2.1 ablation. All paths
relative to the repo root. This supersedes the 1.2 runbook for the
remediation work; the 1.2 runbook is preserved for historical
reproducibility.

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

# Install Liger (1.2.1 new dep; only needed for --liger runs)
pip install liger-kernel

# Verify Muon + Liger imports
python -c "
from src.optim.muon import Muon, MuonWithAuxAdam
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
print('deps OK')
"
```

If any of these fails, do not proceed.

---

## 1. Adopt the 1.2 baseline into the 1.2.1 results dir

The 1.2 baseline row is identical in every way to what 1.2.1 needs —
same data, seed, arch, token budget. We carry it across instead of
re-training:

```
python scripts/validate_planck12.py \
    --results-dir results/planck_12_1 \
    --adopt baseline=results/planck_12/baseline/train_log.txt \
    --adopt-wall-s 11229
```

This copies the log into `results/planck_12_1/baseline/train_log.txt`,
parses final val loss / tok/s / last step, and writes a
`status: adopted` row to `results/planck_12_1/ablation.json`. Subsequent
runs will see baseline as already-done and skip it.

---

## 2. Full 10-run matrix

```
python scripts/validate_planck12.py \
    --data-dir data/fineweb \
    --results-dir results/planck_12_1
```

Each of the remaining 9 runs is a subprocess call to
`scripts/train_lm.py` with a distinct flag combo (see
`docs/plans/planck_12_1_validation.md`). Results accumulate in
`results/planck_12_1/ablation.json`; per-run stdout is teed to
`results/planck_12_1/<run_id>/train_log.txt`.

Expected wall clock on an RTX 4090 at the defaults
(`--max-steps 66750` ≈ 1.09 B tokens):

| run | est. wall | notes |
|---|---|---|
| `baseline` | adopted (0h) | carried over from 1.2 |
| `tl` | ~3.0 h | warmup + floor adds no perf cost |
| `ap` | ~3.0 h (but adopted if already in 1.2 JSON) | see §2a |
| `sk` | ~2.3 h | fixed gather, no longer OOMs |
| `shk` | ~2.8 h | `mix` schedule has one fewer reuse than `always` |
| `all` | ~1.7 h | SGS-native compound |
| `muon` | ~3.0 h | Newton-Schulz overhead is ~1%/step |
| `liger` | ~2.5 h | fused kernel wins on 32k-vocab lm_head |
| `muon_liger` | ~2.5 h | |
| `all_plus` | ~1.5 h | stacked |

Total: ~22h on RTX 4090 if baseline and ap are adopted;
~25h if everything runs fresh.

### 2a. Option: adopt `ap` too

The 1.2 `ap` run was inert (passes never fired, val loss within seed
noise of baseline). If you trust the 1.2 diagnosis, carry it across:

```
python scripts/validate_planck12.py \
    --results-dir results/planck_12_1 \
    --adopt ap=results/planck_12/ap/train_log.txt
```

Saves ~3h. The 1.2.1 gate doesn't depend on `ap`, it's diagnostic.

---

## 3. Smoke tests (before the full matrix)

Run these in sequence; each must pass before proceeding to the next.

### 3a. sk gather parity (500 steps)

```
python scripts/train_lm.py \
    --data-dir data/fineweb \
    --max-steps 500 \
    --save-dir checkpoints/smoke/baseline

python scripts/train_lm.py \
    --data-dir data/fineweb \
    --sparse-k 64 \
    --sparse-warmup-steps 0 \
    --max-steps 500 \
    --save-dir checkpoints/smoke/sk
```

Loss curves must agree within 0.01 nats at matched log steps. If they
diverge, the gather rewrite has broken the sparse path — inspect
`src/sgs_lm.py:_causal_render_sparse`.

### 3b. Liger forward-parity

```
python -c "
import torch, torch.nn.functional as F
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

torch.manual_seed(0)
B, L, D, V = 4, 16, 128, 1024
hidden = torch.randn(B*L, D, device='cuda', dtype=torch.bfloat16)
weight = torch.randn(V, D, device='cuda', dtype=torch.bfloat16)
targets = torch.randint(0, V, (B*L,), device='cuda')

ref = F.cross_entropy(hidden @ weight.T, targets)
liger = LigerFusedLinearCrossEntropyLoss()(weight, hidden, targets)
print(f'ref={ref.item():.6f} liger={liger.item():.6f} diff={abs(ref-liger).item():.2e}')
assert abs(ref - liger).item() < 1e-3, 'Liger parity check failed'
print('OK')
"
```

### 3c. Muon stability (500 steps)

```
python scripts/train_lm.py \
    --data-dir data/fineweb \
    --optimizer muon \
    --max-steps 500 \
    --save-dir checkpoints/smoke/muon
```

Loss must be finite and the 50-step moving average monotonically
decreasing. If it NaNs, retry once with `--muon-lr 0.01`; a second
failure means bail.

---

## 4. Individual runs (for iteration / re-runs)

```
# Just muon
python scripts/validate_planck12.py \
    --data-dir data/fineweb \
    --results-dir results/planck_12_1 \
    --only muon

# Re-run a single config after a crash with --force
python scripts/validate_planck12.py \
    --data-dir data/fineweb \
    --results-dir results/planck_12_1 \
    --only all_plus \
    --force
```

The harness skips any run already marked `status: ok` or `adopted` in
`ablation.json` unless `--force` is passed.

---

## 5. Muon NaN fallback

If `muon` or `muon_liger` or `all_plus` NaNs mid-run:

```
# One-shot retry at a safer LR
python scripts/train_lm.py \
    --data-dir data/fineweb \
    --optimizer muon \
    --muon-lr 0.01 \
    --max-steps 66750 \
    --save-dir checkpoints/planck_12_1/muon \
    > results/planck_12_1/muon/train_log.txt 2>&1

# Then re-adopt the new log into the JSON manually
python scripts/validate_planck12.py \
    --results-dir results/planck_12_1 \
    --adopt muon=results/planck_12_1/muon/train_log.txt
```

Second NaN: file an issue; don't mask with wd/clip tuning. Muon
divergence at 0.01 lr signals a compatibility problem with the Gaussian
param shapes.

---

## 6. Inspecting results

```
cat results/planck_12_1/ablation.json
```

Summary table is printed at the end of every harness invocation. To
rebuild it from an existing JSON without running anything:

```
python scripts/validate_planck12.py \
    --data-dir data/fineweb \
    --results-dir results/planck_12_1 \
    --dry-run
```

---

## 7. Publishing results

1. Write `results/planck_12_1/README.md` covering: gate pass/fail
   (`all`, `muon_liger`, `all_plus`), per-run val loss + tok/s deltas,
   any anomalies.
2. `git add results/planck_12_1/ablation.json results/planck_12_1/README.md`
3. Commit.
4. If any compound gate passed:
   - Flip `Planck 1.2.1` to `done` in `roadmap.md`.
   - Unblock `Hertz 1.2` and record the winning recipe in SETUP.md §6.5.
5. If none passed: keep `Planck 1.2.1` `in progress`, open a `1.2.2`
   row for the targeted fix on the weakest proposal.

---

## 8. Gotchas

- **Disk budget.** `--save-interval 10000` is the default for 1.2.1;
  do not lower it. 10 runs × 6 checkpoints × ~400 MB ≈ 24 GB.
- **Liger + tl.** `--liger` overrides `--transmittance-loss` with a
  one-line warning; don't panic. The tl signal comes from the `all`
  run, not `all_plus`.
- **Muon on 1D params.** Muon itself raises if given a non-2D tensor.
  The `MuonWithAuxAdam` wrapper routes automatically; if you
  hand-build an optimizer, respect the partition (see
  `scripts/train_lm.py` after commit `48f89d4`).
- **Adopted runs and --force.** `--force` re-queues prior `ok` runs
  but does not re-execute this-invocation `--adopt` specs (adoption
  and execution in the same call makes no sense).
- **wandb.** Optional and paid. Prefer the stdout log files; JSON +
  per-run logs capture every metric the gate uses.
