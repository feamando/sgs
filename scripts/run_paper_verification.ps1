<#
 .SYNOPSIS
   One-shot paper verification for the VSP negative-result paper (JMLR target).
   Native PowerShell port of scripts/run_paper_verification.sh (for the 4090 box,
   where WSL/bash is unavailable).

 .DESCRIPTION
   Runs the two blocking controls a main-conference reviewer will demand:
     C1  SCRAMBLED-BUNDLE control  (grounded init vs a permuted-bundle init)
     C2  40k REPRODUCTION           (>=3 seeds of grounded vs baseline at full compute)

   Everything is idempotent: a checkpoint or result that already exists is
   skipped, so you can re-run after an interruption and it resumes. When all
   results are present it runs scripts/verify_paper.py and, unless -NoPush,
   commits + pushes the new result JSONs.

 .EXAMPLE
   powershell -ExecutionPolicy Bypass -File scripts\run_paper_verification.ps1
   # full run + push

 .EXAMPLE
   .\scripts\run_paper_verification.ps1 -NoPush     # compute only

 .EXAMPLE
   .\scripts\run_paper_verification.ps1 -Quick      # 2k-only smoke (no 40k)

 .NOTES
   GPU budget: 3x 40k runs (~10h each) dominate. The 2k control runs are ~1.3h.
   Expect ~30h wall on a single 4090 for the full grid; -Quick is ~4h.
#>
[CmdletBinding()]
param(
    [switch]$NoPush,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'

# ── repo root (parent of this script's dir) ──────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

$TOKENS  = "data/wiki_vsps"
$VOCAB   = "data/vsps/vocab.json"
$GLOVE   = "data/glove.6B.300d.txt"
$SUBWORD = "data/hertz12_data/tokenizer.model"
$PAIRS   = "scripts/assets/disambig_pairs.json"

$PUSH      = -not $NoPush
$SEEDS_40K = @(0, 1, 2)   # C2: 3 seeds at full compute (s0 already exists)

function Log([string]$msg) {
    Write-Host ""
    Write-Host "[verify $(Get-Date -Format HH:mm:ss)] $msg"
}

# Invoke a native command and hard-fail on non-zero exit (mimics `set -e`).
# NOTE: parameter must NOT be named $Args — that is a PowerShell automatic
# variable, so a param of that name never binds and the splat expands empty.
function Invoke-Checked {
    param([string]$Exe, [string[]]$CmdArgs)
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "command failed (exit $LASTEXITCODE): $Exe $($CmdArgs -join ' ')"
    }
}

# train_one <arm-flag> <regime-steps> <freeze-flag> <seed> <ckpt-dir>
#   ArmFlag: "" (grounded) | "--random-init" | "--shuffle-bundles"
#   FreezeFlag: "" | "--freeze-vp-forever"
function Train-One {
    param([string]$ArmFlag, [int]$Steps, [string]$FreezeFlag, [int]$Seed, [string]$Dir)
    if (Test-Path "$Dir/final.pt") {
        Log "SKIP train (exists): $Dir/final.pt"
        return
    }
    Log "TRAIN $Dir  (steps=$Steps seed=$Seed flags='$ArmFlag $FreezeFlag')"
    $a = @("scripts/train_planck2.py", "--tokens", $TOKENS, "--vocab", $VOCAB,
           "--opt-steps", "$Steps", "--seed", "$Seed")
    if ($ArmFlag)    { $a += $ArmFlag }
    if ($FreezeFlag) { $a += $FreezeFlag }
    $a += @("--save-dir", $Dir)
    Invoke-Checked "python" $a
}

# eval_one <ckpt-dir> <out-json>
function Eval-One {
    param([string]$Dir, [string]$Out)
    if (Test-Path $Out) {
        Log "SKIP eval (exists): $Out"
        return
    }
    if (-not (Test-Path "$Dir/final.pt")) {
        Log "WARN: no checkpoint at $Dir, cannot eval -> $Out"
        return
    }
    Log "EVAL $Dir -> $Out"
    Invoke-Checked "python" @("scripts/eval_disambiguation.py", "--checkpoint", "$Dir/final.pt",
        "--vocab", $VOCAB, "--glove", $GLOVE, "--subword-model", $SUBWORD,
        "--pairs", $PAIRS, "--out", $Out)
}

# ── C1: scrambled-bundle control (grounded is s0, already done; add scrambled) ──
# freeze-forever so the init actually persists (matches the 2k grounded/baseline arm)
Log "=== C1: scrambled-bundle control ==="
Train-One "--shuffle-bundles" 2000 "--freeze-vp-forever" 0 "checkpoints/planck2_scrambled_2k_s0"
Eval-One  "checkpoints/planck2_scrambled_2k_s0" "results/disambig_scrambled_2k_s0.json"

if (-not $Quick) {
    Train-One "--shuffle-bundles" 40000 "" 0 "checkpoints/planck2_scrambled_40k_s0"
    Eval-One  "checkpoints/planck2_scrambled_40k_s0" "results/disambig_scrambled_40k_s0.json"

    # ── C2: 40k reproduction. s0 exists as legacy disambig_{vsp,baseline}.json ──
    Log "=== C2: 40k reproduction (seeds $($SEEDS_40K -join ' ')) ==="
    foreach ($s in $SEEDS_40K) {
        if ($s -eq 0) {
            # seed 0 already trained+evaled under legacy names; skip retrain, alias eval
            if ((-not (Test-Path "results/disambig_vsp_40k_s0.json")) -and (Test-Path "results/disambig_vsp.json")) {
                Copy-Item "results/disambig_vsp.json" "results/disambig_vsp_40k_s0.json"
            }
            if ((-not (Test-Path "results/disambig_baseline_40k_s0.json")) -and (Test-Path "results/disambig_baseline.json")) {
                Copy-Item "results/disambig_baseline.json" "results/disambig_baseline_40k_s0.json"
            }
            continue
        }
        Train-One ""             40000 "" $s "checkpoints/planck2_vsp_40k_s$s"
        Eval-One  "checkpoints/planck2_vsp_40k_s$s" "results/disambig_vsp_40k_s$s.json"
        Train-One "--random-init" 40000 "" $s "checkpoints/planck2_baseline_40k_s$s"
        Eval-One  "checkpoints/planck2_baseline_40k_s$s" "results/disambig_baseline_40k_s$s.json"
    }
}
else {
    Log "QUICK mode: skipping all 40k runs (C2 and scrambled-40k)"
}

# ── verdict ──
Log "=== verification summary ==="
# `python ... | tee` equivalent: capture, print, and write the summary file.
& python "scripts/verify_paper.py" | Tee-Object -FilePath "results/paper_verification_summary.txt"
if ($LASTEXITCODE -ne 0) { throw "verify_paper.py failed (exit $LASTEXITCODE)" }

# ── push ──
if ($PUSH) {
    Log "=== git push ==="
    git add results/disambig_scrambled_*.json results/disambig_vsp_40k_*.json `
            results/disambig_baseline_40k_*.json results/paper_verification_summary.txt 2>$null
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Log "nothing new to commit"
    }
    else {
        $commitMsg = @"
paper verification: scrambled-bundle control + 40k reproduction

Automated run of scripts/run_paper_verification.ps1. Adds the two blocking
controls for the JMLR submission: scrambled-bundle init (C1) and multi-seed 40k
reproduction of the -3.8 (C2). See results/paper_verification_summary.txt.
"@
        Invoke-Checked "git" @("commit", "-q", "-m", $commitMsg)
        Invoke-Checked "git" @("push")
        Log "pushed."
    }
}
else {
    Log "-NoPush: leaving results uncommitted."
}

Log "DONE."
