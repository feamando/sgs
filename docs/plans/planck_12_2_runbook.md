# Planck 1.2.2 — Runbook

*Status: active. Written 2026-04-29. Pairs with
`planck_12_2_plan.md` (three-bug design) and
`planck_12_2_validation.md` (five-fresh-run matrix). Bug fixes land
in `src/optim/muon.py`, `src/sgs_lm.py`, `scripts/train_lm.py`; the
harness at `scripts/validate_planck12.py` has five new `*_fix` rows.*

Concrete commands for running the 1.2.2 remediation. All paths
relative to the repo root. This supersedes the 1.2.1 runbook only
for the remediation work, the 1.2.1 runbook stays for historical
reproducibility.

---

## 0. Prereqs

```powershell
# Pre-flight: data is intact
Get-ChildItem data\fineweb\train.bin, data\fineweb\val.bin `
  | Select-Object Name, @{N='GB';E={[math]::Round($_.Length/1GB, 2)}}

# Default path still returns bare logits (backward compat).
python -c "import torch; from src.sgs_lm import SGSLanguageModel; m = SGSLanguageModel(vocab_size=256, d_s=16, d_f=32, n_heads=2, n_passes=3, max_len=16); y = m(torch.zeros(1, 8, dtype=torch.long)); print('OK' if isinstance(y, torch.Tensor) else 'FAIL', y.shape)"

# Bug 1 specifically: MuonWithAuxAdam must now pass isinstance(..., Optimizer).
python -c "from torch.optim import Optimizer; from src.optim.muon import MuonWithAuxAdam; import torch.nn as nn; m = nn.Linear(8, 8); opt = MuonWithAuxAdam([m.weight], [m.bias]); print('Optimizer subclass:', isinstance(opt, Optimizer))"
```

If any of these fails, stop. Don't burn smoke-test GPU time until
the imports line up.

---

## 1. Adopt the 1.2 baseline into the 1.2.2 results dir

The adopted row is unchanged across 1.2, 1.2.1, and 1.2.2. Carry it
across instead of re-training:

```powershell
python scripts\validate_planck12.py `
    --results-dir results\planck_12_2 `
    --adopt baseline=results\planck_12\baseline\train_log.txt `
    --adopt-wall-s 11229
```

This copies the log into `results/planck_12_2/baseline/train_log.txt`,
parses final val loss / tok/s / last step, and writes a
`status: adopted` row to `results/planck_12_2/ablation.json`.
Subsequent runs will skip baseline.

`ap` is NOT re-adopted, it's diagnostic-only. The 1.2.1 row at
`results/planck_12_1/ablation.json` remains the reference.

---

## 2. Smoke tests (mandatory before the full matrix)

Run in sequence. A failure at any step aborts the matrix.

### 2a. Muon scheduler-compat (500 steps)

```powershell
python scripts\train_lm.py --data-dir data\fineweb `
    --optimizer muon `
    --max-steps 500 `
    --save-dir checkpoints\smoke\muon_1_2_2
```

Must: (a) finish without `TypeError: MuonWithAuxAdam is not an
Optimizer`, (b) log monotonically decreasing 50-step moving-average
loss. Wall-clock ~2 min.

### 2b. sk forward+backward parity (500 steps)

```powershell
python scripts\train_lm.py --data-dir data\fineweb `
    --max-steps 500 `
    --save-dir checkpoints\smoke\baseline_1_2_2

python scripts\train_lm.py --data-dir data\fineweb `
    --sparse-k 64 --sparse-warmup-steps 0 `
    --max-steps 500 `
    --save-dir checkpoints\smoke\sk_1_2_2
```

Must: (a) `sk_1_2_2` finishes without OOM, (b) loss curves agree
within 0.02 nats at matched log steps (0.01 is the strict target;
0.02 accepts the tiny numerical difference between advanced-indexing
and `index_select` reductions). Peak VRAM from
`torch.cuda.max_memory_allocated()` at the end of the sk run must be
≤ 18 GB at the default config.

### 2c. tl T-drift diagnostic (2,000 steps)

```powershell
python scripts\train_lm.py --data-dir data\fineweb `
    --transmittance-loss --tl-warmup-steps 500 --tl-floor-eps 0.05 `
    --max-steps 2000 --log-interval 50 `
    --save-dir checkpoints\smoke\tl_1_2_2
```

Must: `T_mean` at step 2000 < 0.95 (T has started to drift off the
1.0 attractor after the warmup ends at step 500). If T stays pinned,
Bug 3's additive-floor fix is insufficient and we stop before
burning the full 66k-step `tl_fix` budget — instrument raw `T_all`
distribution per the plan's risk register.

---

## 3. Matrix

Three per-proposal runs in a first pass (parallel-safe on separate
GPUs, sequential on the single 4090), then the two compounds once
all three per-proposal runs land clean.

### 3a. Per-proposal (sk_fix, muon_fix, tl_fix)

```powershell
python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2 `
    --only sk_fix

python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2 `
    --only muon_fix

python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2 `
    --only tl_fix
```

Abort the matrix on any sanity-check failure (see
`planck_12_2_validation.md` §Per-run sanity checks). Re-run the
failing config only after patching the root cause.

### 3b. Compounds (all_fix, all_plus_fix)

Only after all three per-proposal runs show `status: ok` in
`ablation.json`:

```powershell
python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2 `
    --only all_fix

python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2 `
    --only all_plus_fix
```

### 3c. Full-matrix one-shot (alternative)

If you trust the smoke tests and want to queue everything
back-to-back:

```powershell
python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2
```

Skips adopted rows (baseline) and any previously-ok rows. Drives
each remaining run as a subprocess; stdout tees to
`results\planck_12_2\<run_id>\train_log.txt`.

Expected wall-clock at `--max-steps 66750`:

| run | est. wall |
|---|---|
| `baseline` | adopted (0h) |
| `sk_fix` | ~2.5h |
| `muon_fix` | ~3.0h |
| `tl_fix` | ~3.0h |
| `all_fix` | ~1.7h |
| `all_plus_fix` | ~1.7h |

Total: ~12h GPU on RTX 4090 with baseline adopted. +~3h NaN slack.

---

## 4. Muon NaN fallback

If `muon_fix` or `all_plus_fix` NaNs mid-run, one retry at a safer
LR:

```powershell
python scripts\train_lm.py --data-dir data\fineweb `
    --optimizer muon `
    --muon-lr 0.01 `
    --max-steps 66750 `
    --save-dir checkpoints\planck_12_2\muon_fix `
    > results\planck_12_2\muon_fix\train_log.txt 2>&1

# Re-adopt the retry log into the JSON
python scripts\validate_planck12.py `
    --results-dir results\planck_12_2 `
    --adopt muon_fix=results\planck_12_2\muon_fix\train_log.txt
```

For non-Muon NaNs, one retry at `--batch-size 16` instead. A second
NaN on either path: file an issue, don't mask with wd/clip tuning.

---

## 5. Inspecting results

```powershell
cat results\planck_12_2\ablation.json
```

Summary table is printed at the end of every harness invocation. To
rebuild it from an existing JSON without running anything:

```powershell
python scripts\validate_planck12.py `
    --data-dir data\fineweb `
    --results-dir results\planck_12_2 `
    --dry-run
```

The `speedup` column is masked to `—` for any row that either didn't
complete or finished <90% of the baseline's token budget, so a
crashed-at-step-0 run no longer advertises a bogus multi-× wall-clock
win.

---

## 6. Publishing

1. Write `results/planck_12_2/README.md` covering: gate verdict
   (`all_fix`, `muon_fix`, `all_plus_fix`), per-run val loss + tok/s
   deltas vs baseline, any anomalies.
2. Stage and commit:

```powershell
git add results\planck_12_2\ablation.json results\planck_12_2\README.md
git commit -m "Planck 1.2.2: ablation results + gate verdict"
git push
```

3. If any compound gate passed:
   - Flip both `Planck 1.2.1` and `Planck 1.2.2` to `done` in
     `roadmap.md`.
   - Flip `Hertz 1.2` from blocked to open.
   - Record the winning recipe in `SETUP.md` §6.5 (pick template
     A, B, or C).

4. If none passed:
   - If `muon_fix` cleared its own bar (≥ 1.35× sample-eff, no
     wall-clock regression), consider pivoting Hertz 1.2 to plain
     Muon alone (SETUP §6.5 template B).
   - Otherwise keep Hertz 1.2 blocked and open a 1.2.3 row only if
     there's a concrete, well-scoped next lever. Don't open an empty
     bucket.

---

## 7. Gotchas

- **`MuonWithAuxAdam` state-dict round-trip.** `load_state_dict`
  now re-links `self.param_groups` to the loaded inner optimizers.
  If you write custom checkpoint code that stashes `param_groups`
  separately, drop that — it's redundant and desynchronises on load.
- **`index_select` vs advanced indexing parity.** Forward tensors
  are bitwise-identical; backward differs only in the sparse scatter
  vs. dense path. 0.02 nats is the tolerance at 500 steps; tighter
  divergence at longer horizons is expected (different update
  application order).
- **Additive floor weight renormalisation.** The `weight.mean()` is
  detached — do NOT let gradient flow through the normaliser, or the
  reweight becomes a no-op.
- **`--force` and adoption.** `--force` re-queues prior `ok` runs
  but does not re-execute `--adopt` specs. If you're re-adopting the
  baseline from a different log, delete the prior baseline row from
  `ablation.json` first.
- **wandb.** Optional and paid. Prefer the stdout log files; JSON +
  per-run logs capture every metric the gate uses.
