# Disk Cleanup Guide

Run this top-to-bottom when the SGS checkpoints dir has ballooned.
Every step is: **dry-run first**, review, **then delete**.

PowerShell-first. Repo root assumed to be `C:\Users\feama\sgs`.

Last updated: 2026-05-06.

---

## TL;DR — 2026-05-06 reclaim plan (~380 GB)

As of 2026-05-06, `checkpoints\` is 860 GB. 468 GB is active
Planck 1.3 training — **do not touch it**. The rest is safe to clean.

| # | target | reclaim | risk |
|---|---|---|---|
| 1 | Planck 1.2 / 1.2.1 / 1.2.2 ablations (FAIL, shelved) | ~320 GB | none — all closed FAIL |
| 2 | Planck 1.0 old checkpoints | ~50 GB | none — superseded by 1.1 and 1.3 |
| 3 | Planck 1.1 ablation variants (`*_5k`, `*_k16`, `*_noablob`, `*_t50`) | ~12 GB | none — canonical `planck11\` kept |
| 4 | Hertz 1.0 artefacts | negligible on this box | none — shelved 2026-04-20 |

**Total: ~380 GB.** Planck 1.3 (active, 468 GB) and canonical
`planck11\best.pt` (frozen-encoder bootstrap, 3-4 GB) stay intact.

Run the four steps below in any order; each is independent.

---

## Step 1 — Planck 1.2.x ablation runs (~320 GB)

All three 1.2.x tracks closed FAIL 2026-05-01. They are not reopened.

**Dry run** (lists what would be deleted):

```powershell
foreach ($dir in "planck_12","planck_12_1","planck_12_2") {
  Get-ChildItem "checkpoints\$dir" -File -Recurse |
    Where-Object { $_.Name -ne 'best.pt' } | Remove-Item -WhatIf
}
```

Review the output. Then **execute** by removing `-WhatIf`:

```powershell
foreach ($dir in "planck_12","planck_12_1","planck_12_2") {
  Get-ChildItem "checkpoints\$dir" -File -Recurse |
    Where-Object { $_.Name -ne 'best.pt' } | Remove-Item
}
```

Keeps `best.pt` in each dir as a historical reference (a few GB
total). If you want to purge those too:

```powershell
Remove-Item checkpoints\planck_12, checkpoints\planck_12_1, checkpoints\planck_12_2 -Recurse -Force
```

---

## Step 2 — Planck 1.0 checkpoints (~50 GB)

Planck 1.0 is the foundation 100M LM. It's superseded by Planck 1.1
(blob-concept validator) and, once it ships, Planck 1.3 (Wikipedia
base). The `best.pt` in `checkpoints\planck\` is the historical
artefact — keep that, drop the rest.

**Dry run**:

```powershell
Get-ChildItem checkpoints\planck -File -Recurse |
  Where-Object { $_.Name -ne 'best.pt' } | Remove-Item -WhatIf
```

**Execute**:

```powershell
Get-ChildItem checkpoints\planck -File -Recurse |
  Where-Object { $_.Name -ne 'best.pt' } | Remove-Item
```

---

## Step 3 — Planck 1.1 ablation variants (~12 GB)

`checkpoints\planck11\` is the canonical Planck 1.1 checkpoint (the
bootstrap encoder for Raum when Planck 1.3 isn't ready — see
`SETUP_202605.md` §3). The four `planck11_*` variants are ablations
from the blob-concept experiments and aren't referenced anywhere.

**Dry run**:

```powershell
Remove-Item checkpoints\planck11_5k, checkpoints\planck11_k16, `
            checkpoints\planck11_noablob, checkpoints\planck11_t50 `
            -Recurse -Force -WhatIf
```

**Execute**:

```powershell
Remove-Item checkpoints\planck11_5k, checkpoints\planck11_k16, `
            checkpoints\planck11_noablob, checkpoints\planck11_t50 `
            -Recurse -Force
```

**Do NOT touch `checkpoints\planck11\`** — that's the canonical
bootstrap encoder.

---

## Step 4 — Hertz 1.0 artefacts

Hertz 1.0 was paused 2026-04-20 as infeasible. Any checkpoints from
the aborted run are not used by Hertz 1.2 (which starts from scratch
on plain AdamW).

**Dry run**:

```powershell
Get-ChildItem checkpoints\hertz, checkpoints\hertz_nocompile, checkpoints\hertz_profile `
  -Recurse -File -ErrorAction SilentlyContinue |
  Measure-Object Length -Sum | ForEach-Object { "Hertz footprint: {0:N2} GB" -f ($_.Sum/1GB) }

Remove-Item checkpoints\hertz, checkpoints\hertz_nocompile, checkpoints\hertz_profile `
  -Recurse -Force -WhatIf
```

**Execute**:

```powershell
Remove-Item checkpoints\hertz, checkpoints\hertz_nocompile, checkpoints\hertz_profile `
  -Recurse -Force
```

On the current box these are ~0 GB each per the inventory, but clear
the directories so the next Hertz 1.2 launch starts from a clean
slate.

---

## Do NOT touch

| path | reason |
|---|---|
| `checkpoints\planck13\` | **Active Planck 1.3 training** (468 GB; do not interrupt) |
| `checkpoints\planck11\best.pt` | Bootstrap encoder for Raum when 1.3 isn't ready (see `SETUP_202605.md` §3) |
| `checkpoints\raum_10\`, `raum_11\`, `raum_c*\`, `raum_d\` | Live Raum track, small anyway |
| `data\wikipedia\train.bin`, `val.bin`, `tokenizer.model` | Planck 1.3 training data, ~8 GB |
| `klang\references\variant_a_3000g\*` | The benchmark Klang 1.3 chases |
| `.venv\` | Python environment; recreating costs time + bandwidth |
| `.git\` | Repo history |

---

## Verify after cleanup

Re-run the top-level inventory:

```powershell
Get-ChildItem checkpoints -Directory | ForEach-Object {
  $s=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
      Measure-Object Length -Sum).Sum/1GB
  "{0,-50} {1,8:N2} GB" -f $_.Name,$s
}
```

Expected after all four steps, with Planck 1.3 still running:

```
planck                                                 <5 GB
planck11                                               ~3 GB
planck13                                              <in flux, ~470 GB>
planck_12 / planck_12_1 / planck_12_2                  <1 GB each (or gone)
planck11_5k / k16 / noablob / t50                      (gone)
hertz / hertz_nocompile / hertz_profile                (empty)
raum_*                                                 <1 GB each
```

---

## Appendix — diagnostics (if numbers drift again)

### Top-level repo dirs by size

```powershell
Get-ChildItem -Directory | ForEach-Object {
  $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
           Measure-Object -Property Length -Sum).Sum / 1GB
  [PSCustomObject]@{Path=$_.Name; SizeGB=[math]::Round($size,2)}
} | Sort-Object SizeGB -Descending
```

### User-profile caches (outside the repo)

```powershell
"{0,8:N2} GB  HF hub cache"      -f ((Get-ChildItem $env:USERPROFILE\.cache\huggingface          -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
"{0,8:N2} GB  pip cache"         -f ((Get-ChildItem $env:LOCALAPPDATA\pip\cache                  -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
"{0,8:N2} GB  torch hub cache"   -f ((Get-ChildItem $env:USERPROFILE\.cache\torch                -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)
```

Free actions: `pip cache purge`; delete unused models in
`$env:USERPROFILE\.cache\huggingface\hub`.

---

## Prevent future bloat

- `--wandb` off by default (it writes a local run dir per
  invocation; see `feedback_sgs_wandb_default` memory).
- `train_hertz.py --keep-last 3` rotates `step_*.pt` checkpoints.
  `train_lm.py` should too — if Planck 1.3's checkpoint dir keeps
  growing past 500 GB, add the same flag to that script.
- Run the `TL;DR` block at least once per major version (Planck 1.4,
  Hertz 1.2, Raum 0.1) to keep the FAIL-track detritus from
  accumulating again.
