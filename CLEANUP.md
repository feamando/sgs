# Disk Cleanup Guide

When the SGS working tree (or the Windows user profile) runs low on
space, work top-to-bottom through this doc. Every section is in the
form: **(a) how to measure, (b) what it is, (c) whether it's safe to
delete, (d) the exact command**.

PowerShell-first. All paths assume the repo root is `C:\Users\feama\sgs`
and the Windows user is `feama`. Adjust if different.

Last updated: 2026-05-06.

---

## 1. Diagnose — find the culprits first

### 1.1 Top-level repo dirs by size

From the repo root:

```powershell
Get-ChildItem -Directory | ForEach-Object {
  $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
           Measure-Object -Property Length -Sum).Sum / 1GB
  [PSCustomObject]@{Path=$_.Name; SizeGB=[math]::Round($size,2)}
} | Sort-Object SizeGB -Descending
```

Usual heavy hitters (biggest first): `data\`, `checkpoints\`, `results\`,
`klang\`, `.venv\`.

### 1.2 Wikipedia staging dir breakdown

```powershell
Get-ChildItem data\wikipedia -Directory -ErrorAction SilentlyContinue |
  ForEach-Object {
    $s=(Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum/1GB
    "{0,-50} {1,8:N2} GB" -f $_.Name,$s
  }
```

### 1.3 Checkpoints breakdown

```powershell
Get-ChildItem checkpoints -Directory -ErrorAction SilentlyContinue |
  ForEach-Object {
    $s=(Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum/1GB
    "{0,-50} {1,8:N2} GB" -f $_.Name,$s
  }
```

### 1.4 User-profile caches (outside the repo)

```powershell
"{0,8:N2} GB  HF hub cache"      -f ((Get-ChildItem $env:USERPROFILE\.cache\huggingface          -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
"{0,8:N2} GB  HF datasets cache" -f ((Get-ChildItem $env:USERPROFILE\.cache\huggingface\datasets -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
"{0,8:N2} GB  pip cache"         -f ((Get-ChildItem $env:LOCALAPPDATA\pip\cache                  -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
"{0,8:N2} GB  torch hub cache"   -f ((Get-ChildItem $env:USERPROFILE\.cache\torch                -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
```

---

## 2. Safe-to-delete catalogue

Ordered by typical size won back per action, biggest first.

### 2.1 Old raw Wikipedia dump (~24 GB) — SAFE

The bz2 XML dump you downloaded before the `wikiextractor` pivot. Not
used by Planck 1.3 anymore; HF Parquet superseded it.

```powershell
# Check what's there first
Get-ChildItem data\wikipedia\*.bz2, data\wikipedia\*.xml, data\wikipedia\*.xml.bz2 -ErrorAction SilentlyContinue

# Delete
Remove-Item data\wikipedia\*.bz2, data\wikipedia\*.xml.bz2 -ErrorAction SilentlyContinue
```

### 2.2 Old ablation checkpoints (~50-200 GB) — usually SAFE

Planck 1.2 / 1.2.1 / 1.2.2 were large ablation sweeps; each track kept
periodic checkpoints. They're all marked `done` as FAIL on the roadmap
and are not reopened.

```powershell
# List candidates
Get-ChildItem checkpoints\planck_12*, checkpoints\planck_1_2* -Recurse -File -ErrorAction SilentlyContinue |
  Measure-Object Length -Sum | ForEach-Object { "{0:N2} GB" -f ($_.Sum/1GB) }

# Keep best.pt, drop the rest
Get-ChildItem checkpoints\planck_12*, checkpoints\planck_1_2* -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -notmatch 'best\.pt$' } | Remove-Item -WhatIf
# Remove `-WhatIf` once you've reviewed the list
```

**Keep**: `checkpoints\planck\best.pt` (Planck 1.0), `checkpoints\planck11\best.pt`
(Planck 1.1 — frozen encoder for Raum), and `checkpoints\planck13\best.pt`
once 1.3 finishes.

### 2.3 Hertz 1.0 remnants (can be 50+ GB) — SAFE

Hertz 1.0 was paused 2026-04-20 as infeasible. Any checkpoints from the
aborted run are not needed.

```powershell
Get-ChildItem checkpoints\hertz*, results\hertz* -Recurse -File -ErrorAction SilentlyContinue |
  Measure-Object Length -Sum | ForEach-Object { "Hertz footprint: {0:N2} GB" -f ($_.Sum/1GB) }

Remove-Item checkpoints\hertz*, results\hertz* -Recurse -Force -WhatIf
```

### 2.4 HF Parquet download for the Wikipedia Arrow build (~20 GB) — DELETE AFTER §2.2 ships

Once `python -m src.tinystories --dataset wikipedia ...` has packed
`train.bin` + `val.bin`, the Parquet files in
`data\wikipedia\hf\wikimedia___wikipedia\<revision>\downloads\` are no
longer needed. The Arrow shards a few dirs up stay in use only during
dataset iteration — after packing, even those are optional.

```powershell
# Space the HF cache currently uses
Get-ChildItem data\wikipedia\hf -Recurse -File -ErrorAction SilentlyContinue |
  Measure-Object Length -Sum | ForEach-Object { "HF cache: {0:N2} GB" -f ($_.Sum/1GB) }

# Only AFTER train.bin / val.bin are packed and you've confirmed with:
#   python scripts/train_lm.py --data-dir data\wikipedia --epochs 1 --max-steps 10
# then:
Remove-Item data\wikipedia\hf -Recurse -Force -WhatIf
```

**Warning**: if you later re-run §2.2 or need the raw dataset, you'll
re-download ~20 GB. Only delete if disk pressure is real.

### 2.5 Klang intermediate artefacts (~10-30 GB) — usually SAFE

Klang 1.0 / 1.1 / 1.2 sweeps wrote many per-run `.pt` + `.wav` bundles.
Variant A and Variant B reference files are keepers; the rest can go
once the gate comparison is finished.

```powershell
Get-ChildItem klang\runs, results\klang* -Recurse -File -ErrorAction SilentlyContinue |
  Measure-Object Length -Sum | ForEach-Object { "Klang artefacts: {0:N2} GB" -f ($_.Sum/1GB) }

# Review before removing — Variant A's 3000g reference is the Klang 1.1
# benchmark we still measure 1.3 against
```

**Keep**: `klang\references\variant_a_3000g\*`, `klang\references\variant_b_*\*`,
anything cited in `docs/klang/*.md`.

### 2.6 pip wheel cache (~5-15 GB) — SAFE

Pure convenience cache. Deleting costs a re-download on the next
`pip install`.

```powershell
pip cache purge
```

### 2.7 HF hub cache (~5-50 GB) — CONTEXT-DEPENDENT

`$env:USERPROFILE\.cache\huggingface\hub` holds downloaded models and
datasets *outside* the repo. On a fresh box it's small; if you ever ran
frontier-model experiments it can be huge.

```powershell
Get-ChildItem $env:USERPROFILE\.cache\huggingface\hub -Directory -ErrorAction SilentlyContinue |
  ForEach-Object {
    $s=(Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum/1GB
    "{0,-60} {1,8:N2} GB" -f $_.Name,$s
  } | Sort-Object -Property {[double]($_ -replace '.*?(\d[\d\.]*)\s*GB','$1')} -Descending
```

Safe to delete entries for models you're not actively using.

### 2.8 torch hub cache — SAFE

```powershell
Remove-Item $env:USERPROFILE\.cache\torch -Recurse -Force -WhatIf
```

### 2.9 `wandb\` local run dirs — usually SAFE

Only relevant if wandb was ever enabled (it's off by default per the
`feedback_sgs_wandb_default` memory).

```powershell
Get-ChildItem wandb -Directory -ErrorAction SilentlyContinue |
  Measure-Object | Select-Object -ExpandProperty Count
# If non-zero, and you don't need local run history:
Remove-Item wandb -Recurse -Force -WhatIf
```

### 2.10 Results dir (`results\` / `paper\`) — PRUNE, DON'T WIPE

Training logs, JSON eval outputs, figures. Individual files are small
but they accumulate. Keep anything referenced by a `docs/plans/**.md`
or `docs/analysis/**.md`.

```powershell
# List largest files under results\
Get-ChildItem results -Recurse -File -ErrorAction SilentlyContinue |
  Sort-Object Length -Descending | Select-Object -First 30 Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}, FullName
```

---

## 3. Do NOT delete

These look big and deletable but are load-bearing right now:

| path | why keep |
|---|---|
| `data\wikipedia\train.bin`, `val.bin` | Planck 1.3 training data, ~8 GB |
| `data\wikipedia\tokenizer.model`, `tokenizer.vocab` | SentencePiece trained on Wikipedia; re-training is 1-3 hours |
| `data\wikipedia\snapshot_id.txt` | Revision pin for reproducibility |
| `checkpoints\planck11\best.pt` | Frozen encoder for Raum 1.1, Satz 0.1 fallback |
| `checkpoints\planck13\best.pt` | (once it exists) primary Planck base |
| `checkpoints\klang\*\best.pt` | Klang 1.2 Gate A/B reference checkpoints |
| `.venv\` | the Python environment; recreating costs 10-30 min + bandwidth |
| `.git\` | repo history (should be small; don't touch regardless) |
| `klang\references\variant_a_3000g\*` | the benchmark Klang 1.3 chases |
| `data\blobs\wikipedia\*` | (once built) Planck 1.3.1a blob index |

---

## 4. Fast recipe for "I just need 100 GB back, right now"

In priority order, stop after each step if you're under the bar:

1. **`Remove-Item data\wikipedia\*.bz2`** → ~24 GB
2. **Prune ablation checkpoints** (§2.2) → 50-200 GB (biggest single win)
3. **`Remove-Item checkpoints\hertz*`** (§2.3) → 10-50 GB
4. **`pip cache purge`** (§2.6) → 5-15 GB
5. **Prune old HF hub models** (§2.7) → 5-50 GB
6. **Delete `data\wikipedia\hf` after §2.2 packs the bins** (§2.4) → ~20 GB
7. **Clean `klang\runs`** (§2.5) → 10-30 GB

If all of that still doesn't clear it, the problem is outside the SGS
tree — run WinDirStat or TreeSize against `C:\Users\feama` and
`C:\Users` to find it.

---

## 5. Prevent future bloat

- Don't enable `--wandb` by default; it writes a local run dir per
  invocation (see `feedback_sgs_wandb_default` memory).
- Don't keep every intermediate checkpoint — configure `train_lm.py`
  and `train_hertz.py` with `--keep-last N` where available.
- Don't pin HF datasets in the repo tree (`data\wikipedia\hf`) if a
  shared location like `%USERPROFILE%\.cache\huggingface\datasets` is
  acceptable. The current setup puts Wikipedia inside the repo on
  purpose so the snapshot is co-located with the training binaries,
  but this costs ~20-40 GB extra during clustering.
- Run `CLEANUP.md §1` diagnostics monthly. Catching a 50 GB ablation
  artefact before it becomes five is easier than finding 500 GB of
  surprises.
