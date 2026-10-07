<#
 .SYNOPSIS
   Planck 3.0 one-stop runner for the Windows 4090 box.
   Step-by-step guides: SETUP_planck_20260901.md (rounds 1-2), SETUP_planck_20260903.md (round 3)
   Plan: SETUP_092026_planck3.md

 .DESCRIPTION
   Every stage is idempotent: finished outputs are skipped, so re-running after
   an interruption resumes. First time: doctor -Deep, then all -Quick, then all.

     doctor    preflight + ETA per stage (-Deep loads Gemma/Planck and measures them)
     all       setup -> smoke -> doctor -> searxng -> g0 -> g1 -> report -> commit+push results
               -Quick = ~15-30 min shakedown of every stage into *_quick outputs (full run not skipped)
     setup     install deps into .venv
     smoke     offline tests (no network, no GPU), ~5s
     searxng   start the local SearXNG search container (Docker Desktop, optional)
     serve     local web chat at http://127.0.0.1:8010 (answer + "In depth" + "For you")
     chat      the same in the terminal ('more' shows the evidence)
     ask       one question:  .\scripts\planck3.ps1 ask "Who founded SpaceX?"
     digest    "For you" from your local knowledge graph (-Explore also reads adjacent entities from trusted sources)
     schedule  run "digest -Explore" daily via Windows Task Scheduler (-At 08:00); unschedule removes it
     g0        G0: seed benchmark, snippet-first + confidence gate: Gemma on tools, Gemma closed-book
               (the base-chat rival), heuristic floor (pages-first runs from 2026-10-05 stay as baseline)
     g1        G1: Wikiracing: graph -> tasks -> embed -> heads (nll + rank) per seed (-Seeds 0,1,2)
               -> eval vs random/lexical/Gemma (teacher on 300 races, cached) -> paired stats -> aggregate
               -Hertz adds the Hertz 1.2 encoder arm (needs checkpoints/hertz/best.pt)
     round3    round 3 (SETUP_planck_20260903.md): fresh + long-tail benchmark vs base chat, G2 (learn the
               decisions from known answers), G1 confirmation (new task seed, seeds 3-5, rank primary, Hertz)
     report    every result in one table -> results/planck3/REPORT.md
     py        pass-through: .\scripts\planck3.ps1 py wikirace eval --limit 50

   Heavy commands (all, g0, g1, doctor) log to results/planck3/logs/. `all` commits the
   summaries, REPORT.md, logs and compressed trajectories, so they can be read from any machine.

 .EXAMPLE
   powershell -ExecutionPolicy Bypass -File scripts\planck3.ps1 doctor -Deep
 .EXAMPLE
   .\scripts\planck3.ps1 all -Quick -NoPush     # shakedown, keep results local
 .EXAMPLE
   .\scripts\planck3.ps1 all                    # the real run; pushes results
 .EXAMPLE
   .\scripts\planck3.ps1 g1 -Seed 1             # extra seed before believing a delta
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Command = "help",
    [Parameter(Position = 1)][string]$Question = "",
    [string]$Type = "auto",
    [string]$Policy = "gemma",
    [int]$Seed = 0,
    [string]$Seeds = "0,1,2",
    [int]$Limit = 0,
    [string]$At = "08:00",
    [switch]$Quick,
    [switch]$Deep,
    [switch]$NoPush,
    [switch]$Explore,
    [switch]$Hertz,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

# Python writes UTF-8; make PowerShell DECODE it as UTF-8 too, or non-ASCII page text and
# progress glyphs land in the console/log as mojibake (seen in the first doctor log).
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:TQDM_DISABLE = "1"                # transformers' weight-loading bars flood the logs
$env:PYTHONUTF8 = "1"                  # page text is not cp1252
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"            # live progress through the log tee
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$PLANCK_CKPT = "checkpoints/planck13/best.pt"
$PLANCK_TOK  = "data/wikipedia/tokenizer.model"
# Hertz 1.2 lives under different names on different boxes: take the first that exists
$HERTZ_CKPT  = @("checkpoints/hertz/best.pt", "checkpoints/hertz12/best.pt", "checkpoints/hertz12/final.pt") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $HERTZ_CKPT) {
    $m = Get-ChildItem -Path "checkpoints/hertz12" -Filter "milestone_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    $HERTZ_CKPT = if ($m) { $m.FullName } else { "checkpoints/hertz/best.pt" }
}
$HERTZ_TOK   = "data/hertz12_data/tokenizer.model"
$GEMMA       = "models/gemma-4-e4b-it"
$WR          = "data/planck3/wikirace"
$RES         = "results/planck3"
$SEARX_NAME  = "planck3-searxng"
$SEARX_PORT  = 8888
$TASK_NAME   = "Planck3Explore"
$FRESH       = "scripts/assets/planck3_tasks_fresh.json"
$G2_TRAIN    = "scripts/assets/planck3_g2_train_tasks.json"
$Tag         = if ($Quick) { "_quick" } else { "" }
$script:LogFile = $null

function Write-Log([string]$line) { if ($script:LogFile) { Add-Content -Path $script:LogFile -Value $line -Encoding UTF8 } }
function Log([string]$msg) { $l = "[planck3 $(Get-Date -Format HH:mm:ss)] $msg"; Write-Host ""; Write-Host $l -ForegroundColor Cyan; Write-Log $l }
function Warn([string]$msg) { Write-Host "  ! $msg" -ForegroundColor Yellow; Write-Log "  ! $msg" }

function Start-RunLog([string]$name) {
    New-Item -ItemType Directory -Force -Path "$RES/logs" | Out-Null
    $script:LogFile = Join-Path $Root "$RES/logs/${name}${Tag}_$(Get-Date -Format yyyyMMdd_HHmm).log"
    Log "logging to $($script:LogFile)"
}

# NOTE: must not be named $Args (PowerShell automatic variable).
function Invoke-Checked {
    param([string]$Exe, [string[]]$CmdArgs)
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "command failed (exit $LASTEXITCODE): $Exe $($CmdArgs -join ' ')" }
}

# Probe a native command without letting stderr trip $ErrorActionPreference='Stop'
# (Windows PowerShell 5.1 turns redirected native stderr into terminating errors).
function Invoke-Quiet {
    param([string]$Exe, [string[]]$CmdArgs)
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = & $Exe @CmdArgs 2>$null; $code = $LASTEXITCODE } catch { $out = $null; $code = 1 }
    $ErrorActionPreference = $old
    return @{ Out = $out; Code = $code }
}

function Get-Python {
    if (Test-Path ".venv\Scripts\python.exe") { return (Resolve-Path ".venv\Scripts\python.exe").Path }
    Log "no .venv found; creating one"
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv } else { & python -m venv .venv }
    return (Resolve-Path ".venv\Scripts\python.exe").Path
}
$PY = Get-Python

# Run scripts/planck3.py. $Rest may be $null, so empty elements are dropped. When a run
# log is active, stdout+stderr are teed into it (UTF-8; Tee-Object would write UTF-16).
function P3 {
    param([string[]]$A)
    $argv = @("scripts/planck3.py") + @($A | Where-Object { $_ })
    if ($script:LogFile) {
        Write-Log "> python $($argv -join ' ')"
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $PY @argv 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Write-Log $line }
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
    } else {
        & $PY @argv
        $code = $LASTEXITCODE
    }
    if ($code -ne 0) { throw "command failed (exit $code): python $($argv -join ' ')  (log: $($script:LogFile))" }
}

function Test-Docker { return [bool](Get-Command docker -ErrorAction SilentlyContinue) }
function Test-Searx { return (Invoke-Quiet $PY @("scripts/planck3.py", "search-check")).Code -eq 0 }
function Get-TeacherPolicy { if ($Policy -eq "gemma" -and -not (Test-Path $GEMMA)) { return "heuristic" } return $Policy }

# ── setup / doctor ───────────────────────────────────────────────────────
function Do-Setup {
    Log "installing Planck 3.0 deps into .venv"
    # install-if-missing (no --upgrade): fast on re-runs, no surprise version bumps mid-experiment
    # lxml_html_clean: lxml >= 5.2 split html.clean out; without it trafilatura/justext fail with
    # ImportError and extraction silently falls back to the crude tag stripper (first box run)
    Invoke-Checked $PY @("-m", "pip", "install", "--quiet", "trafilatura", "lxml_html_clean", "requests", "scipy", "pytest", "sentencepiece")
}

function Do-Doctor([bool]$DeepRun) {
    $a = @("doctor"); if ($DeepRun) { $a += "--deep" }
    try { P3 $a } catch { throw "doctor found blocking problems (FAIL rows above). Fix them, then re-run." }
}

# ── searxng ──────────────────────────────────────────────────────────────
function Do-Searxng {
    if (-not (Test-Docker)) { Warn "Docker not found; search falls back to the Wikipedia API (still valid, narrower)"; return }
    if (Test-Searx) { Log "SearXNG up and answering on :$SEARX_PORT"; return }
    $running = (Invoke-Quiet docker @("ps", "--filter", "name=^$SEARX_NAME`$", "--format", "{{.Names}}")).Out -eq $SEARX_NAME
    if ($running) {
        # up but returning nothing = upstream engines rate-limit it; blocks are usually temporary
        Warn "SearXNG is running but returns no results (upstream engines are rate-limiting it); waiting up to 10 min"
        for ($i = 0; $i -lt 10; $i++) {
            Start-Sleep -Seconds 60
            if (Test-Searx) { Log "SearXNG answers again"; return }
        }
        Warn "still blocked: continuing; every empty SearXNG search falls back to Wikipedia (recorded per run as search_health)"
        return
    }
    $exists = (Invoke-Quiet docker @("ps", "-a", "--filter", "name=^$SEARX_NAME`$", "--format", "{{.Names}}")).Out -eq $SEARX_NAME
    if ($exists) {
        Log "starting existing SearXNG container"
        Invoke-Quiet docker @("start", $SEARX_NAME) | Out-Null
    } else {
        Log "creating SearXNG container (first run pulls the image)"
        $cfg = (Resolve-Path "config/searxng").Path
        $r = Invoke-Quiet docker @("run", "-d", "--name", $SEARX_NAME, "--restart", "unless-stopped",
            "-p", "127.0.0.1:${SEARX_PORT}:8080", "-v", "${cfg}:/etc/searxng",
            "-e", "SEARXNG_BASE_URL=http://localhost:${SEARX_PORT}/", "searxng/searxng:latest")
        if ($r.Code -ne 0) { Warn "docker run failed (is Docker Desktop running?)"; return }
    }
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Searx) { Log "SearXNG is up on http://localhost:$SEARX_PORT"; return }
    }
    Warn "SearXNG did not answer in 60s; check: docker logs $SEARX_NAME  (search falls back to Wikipedia)"
}

# ── G0 ───────────────────────────────────────────────────────────────────
function Do-G0 {
    $pol = Get-TeacherPolicy
    if ($pol -ne $Policy) { Warn "no Gemma at $GEMMA; running the heuristic floor only (G0 verdict needs the teacher)" }
    $sample = @(); if ($Quick) { $sample = @("--sample", "12") } elseif ($Limit -gt 0) { $sample = @("--limit", "$Limit") }
    $runs = @(@{ Name = "g0_${pol}_snip$Tag"; Args = @("--policy", $pol); What = "G0 teacher on our tools, snippet-first + confidence gate: $pol" })
    if ($pol -ne "heuristic") {
        $runs += @{ Name = "g0_${pol}_closedbook$Tag"; Args = @("--policy", $pol, "--closed-book"); What = "G0 base-chat rival: $pol closed-book (no tools, no sources)" }
        $runs += @{ Name = "g0_heuristic_snip$Tag"; Args = @("--policy", "heuristic"); What = "G0 heuristic floor, snippet-first" }
    }
    foreach ($r in $runs) {
        $out = "$RES/$($r.Name)"
        if ((Test-Path "$out/summary.json") -and $Command -eq "all") { Log "SKIP $($r.Name) (exists)"; continue }
        Log $r.What
        P3 (@("g0") + $r.Args + @("--out", $out, "--gemma-path", $GEMMA) + $sample)
    }
}

# ── G1 ───────────────────────────────────────────────────────────────────
function Get-TasksFormat([string]$dir) {
    if (-not (Test-Path "$dir/tasks_info.json")) { return 0 }
    $f = (Get-Content "$dir/tasks_info.json" -Raw | ConvertFrom-Json).format
    if ($f) { return [int]$f } else { return 1 }
}

function Do-G1 {
    # graph + embeddings are shared by the full and quick task sets
    if (-not (Test-Path "$WR/graph.npz")) { Log "G1 build: Simple English Wikipedia link graph (356 MB download)"; P3 @("wikirace", "build") }
    else { Log "SKIP build (exists): $WR/graph.npz" }
    if (-not (Test-Path "$WR/emb_hash.npy")) { Log "G1 embed: hash control"; P3 @("wikirace", "embed", "--encoder", "hash") }
    $havePlanck = (Test-Path $PLANCK_CKPT) -and (Test-Path $PLANCK_TOK)
    if ($havePlanck -and -not (Test-Path "$WR/emb_planck.npy")) {
        Log "G1 embed: frozen Planck 1.3 over the whole graph"
        P3 @("wikirace", "embed", "--encoder", "planck", "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK)
    }
    $haveHertz = $Hertz -and (Test-Path $HERTZ_CKPT) -and (Test-Path $HERTZ_TOK)
    if ($Hertz -and -not $haveHertz) { Warn "-Hertz: $HERTZ_CKPT or $HERTZ_TOK missing; skipping the Hertz arm" }
    if ($haveHertz -and -not (Test-Path "$WR/emb_hertz.npy")) {
        Log "G1 embed: frozen Hertz 1.2 (640M) over the whole graph (capacity ablation)"
        P3 @("wikirace", "embed", "--encoder", "hertz", "--checkpoint", $HERTZ_CKPT, "--tokenizer", $HERTZ_TOK)
    }
    $td = if ($Quick) { "$WR/tasks_quick" } else { $WR }
    $tagArg = if ($Quick) { @("--tag", $Tag) } else { @() }   # never pass an empty --tag value
    $tasksArgs = if ($Quick) { @("--targets", "300") } else { @() }
    if ((Get-TasksFormat $td) -lt 2) {
        # format 2 adds per-candidate BFS distances (for the rank objective); same seed = same races
        Log "G1 tasks${Tag}: BFS labels + candidate distances, split by target"
        P3 (@("wikirace", "tasks") + $tagArg + $tasksArgs)
    } else { Log "SKIP tasks$Tag (format 2 exists)" }

    $seedList = if ($Quick) { @(0) } else { @($Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }) }
    $arms = @(@{ Enc = "hash"; Obj = "nll" })
    if ($havePlanck) { $arms += @{ Enc = "planck"; Obj = "nll" }; $arms += @{ Enc = "planck"; Obj = "rank" } }
    else { Warn "no Planck checkpoint/tokenizer; G1 verdict will be INCOMPLETE" }
    if ($haveHertz) { $arms += @{ Enc = "hertz"; Obj = "nll" }; $arms += @{ Enc = "hertz"; Obj = "rank" } }
    $epochs = if ($Quick) { @("--epochs", "2") } else { @() }
    $haveGemma = Test-Path $GEMMA
    if (-not $haveGemma) { Warn "no Gemma teacher; G1 verdict will be INCOMPLETE" }
    foreach ($sd in $seedList) {
        foreach ($a in $arms) {
            $key = if ($a.Obj -eq "nll") { $a.Enc } else { "$($a.Enc)-$($a.Obj)" }
            if (-not (Test-Path "$RES/g1_head_${key}_s$sd$Tag/head.pt")) {
                Log "G1 train head:$key seed=$sd$Tag"
                P3 (@("wikirace", "train", "--encoder", $a.Enc, "--objective", $a.Obj, "--seed", "$sd") + $tagArg + $epochs)
            } else { Log "SKIP train head:$key s$sd$Tag (exists)" }
        }
        $pols = "random,lexical,heads" + $(if ($haveGemma) { ",gemma" } else { "" })
        Log "G1 eval seed=$sd${Tag}: $pols (teacher races cached after the first seed)"
        $a = @("wikirace", "eval", "--policies", $pols, "--seed", "$sd",
               "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK, "--gemma-path", $GEMMA) + $tagArg
        if ($Quick) { $a += @("--limit", "100", "--teacher-limit", "20") } elseif ($Limit -gt 0) { $a += @("--limit", "$Limit") }
        if ($sd -ne $seedList[0]) { $a += "--no-latency" }
        P3 $a
    }
    if ($seedList.Count -gt 1) {
        Log "G1 aggregate over seeds $($seedList -join ',')"
        P3 (@("wikirace", "aggregate", "--seeds", ($seedList -join ",")) + $tagArg)
    }
}

# ── round 3 (SETUP_planck_20260903.md) ────────────────────────────────────
# A stage is done only if its summary is VALID: closed-book needs no search; every other run
# needs search_health.valid (round 3's first attempt ran on a blocked search engine).
function Test-ValidRun([string]$dir) {
    if (-not (Test-Path "$dir/summary.json")) { return $false }
    $j = Get-Content "$dir/summary.json" -Raw | ConvertFrom-Json
    if ($j.mode -eq "closed_book") { return $true }
    return ($null -ne $j.search_health) -and [bool]$j.search_health.valid
}

function Test-Newer([string]$a, [string]$b) {  # is $a newer than $b (or $b missing)?
    if (-not (Test-Path $b)) { return $true }
    return (Get-Item $a).LastWriteTime -gt (Get-Item $b).LastWriteTime
}
function Do-G0Fresh {
    $pol = Get-TeacherPolicy
    $sample = if ($Quick) { @("--sample", "12") } else { @() }
    $runs = @()
    if ($pol -ne "heuristic") {
        $runs += @{ Name = "g0f_${pol}_closedbook$Tag"; Args = @("--policy", $pol, "--closed-book"); What = "fresh/long-tail: base chat ($pol closed-book)" }
        $runs += @{ Name = "g0f_${pol}_snip$Tag"; Args = @("--policy", $pol); What = "fresh/long-tail: $pol on our tools (snippet-first)" }
    } else { Warn "no Gemma: the fresh benchmark runs without the base-chat rival" }
    $runs += @{ Name = "g0f_heuristic_snip$Tag"; Args = @("--policy", "heuristic"); What = "fresh/long-tail: heuristic (snippet-first)" }
    foreach ($r in $runs) {
        $out = "$RES/$($r.Name)"
        if (Test-ValidRun $out) { Log "SKIP $($r.Name) (valid)"; continue }
        if (Test-Path "$out/summary.json") { Warn "$($r.Name) exists but is INVALID (search failed): re-running" }
        Log $r.What
        P3 (@("g0") + $r.Args + @("--tasks", $FRESH, "--out", $out, "--gemma-path", $GEMMA) + $sample)
    }
}

function Do-G2 {
    if (-not (Test-Path $G2_TRAIN)) { throw "missing $G2_TRAIN (git pull: generated on the Mac from Wikidata and committed)" }
    $lim = if ($Quick) { @("--limit", "100") } else { @() }
    Log "G2 collect: labelled decision points from known answers (resumable; 1 search + 1 page per question)"
    P3 (@("g2", "collect") + $lim)
    $encs = @("hash"); if ((Test-Path $PLANCK_CKPT) -and (Test-Path $PLANCK_TOK)) { $encs += "planck" }
    foreach ($e in $encs) {
        $pts = "data/planck3/g2/points.jsonl"
        if ((Test-Newer $pts "data/planck3/g2/emb_$e.npy") -or $Quick) {
            Log "G2 embed: $e"; P3 @("g2", "embed", "--encoder", $e, "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK)
        }
        if ((Test-Newer "data/planck3/g2/emb_$e.npy" "$RES/g2_head_${e}_s0/head.pt") -or $Quick) {
            Log "G2 train head:$e"; P3 @("g2", "train", "--encoder", $e)
        }
        foreach ($b in @(@{ P = "g0f"; T = $FRESH }, @{ P = "g0"; T = "scripts/assets/planck3_tasks.json" })) {
            $out = "$RES/$($b.P)_planck-g2-${e}$Tag"
            if ((Test-ValidRun $out) -and -not (Test-Newer "$RES/g2_head_${e}_s0/head.pt" "$out/summary.json") -and -not $Quick) {
                Log "SKIP $out (valid, head unchanged)"; continue
            }
            Log "G2 eval head:$e on $($b.T)"
            $smp = if ($Quick) { @("--sample", "12") } else { @() }
            P3 (@("g0", "--policy", "planck", "--g2-head", "$RES/g2_head_${e}_s0/head.pt", "--tasks", $b.T, "--out", $out,
                  "--planck-checkpoint", $PLANCK_CKPT, "--planck-tokenizer", $PLANCK_TOK) + $smp)
        }
    }
}

function Do-G1Confirm {
    # PRE-REGISTERED (SETUP_planck_20260903.md): new task seed 1 (new targets + test races), model seeds
    # 3-5, primary = head:planck-rank; Hertz arms included whenever the checkpoint is present.
    $tt = "_t1"
    if ((Get-TasksFormat "$WR/tasks$tt") -lt 2) { Log "G1 confirm: new task set (seed 1)"; P3 @("wikirace", "tasks", "--seed", "1", "--tag", $tt) }
    $haveHertz = (Test-Path $HERTZ_CKPT) -and (Test-Path $HERTZ_TOK)
    if ($haveHertz -and -not (Test-Path "$WR/emb_hertz.npy")) {
        Log "G1 embed: Hertz 1.2 (640M) over the graph"; P3 @("wikirace", "embed", "--encoder", "hertz", "--checkpoint", $HERTZ_CKPT, "--tokenizer", $HERTZ_TOK)
    }
    $arms = @(@{ Enc = "hash"; Obj = "nll" }, @{ Enc = "planck"; Obj = "nll" }, @{ Enc = "planck"; Obj = "rank" })
    if ($haveHertz) { $arms += @{ Enc = "hertz"; Obj = "nll" }; $arms += @{ Enc = "hertz"; Obj = "rank" } } else { Warn "no Hertz checkpoint: confirmation runs without the Hertz arm" }
    $seeds = if ($Quick) { @(3) } else { @(3, 4, 5) }
    foreach ($sd in $seeds) {
        foreach ($a in $arms) {
            $key = if ($a.Obj -eq "nll") { $a.Enc } else { "$($a.Enc)-$($a.Obj)" }
            if (-not (Test-Path "$RES/g1_head_${key}_s$sd$tt/head.pt")) {
                Log "G1 confirm train head:$key seed=$sd"
                P3 @("wikirace", "train", "--encoder", $a.Enc, "--objective", $a.Obj, "--seed", "$sd", "--tag", $tt)
            }
        }
        Log "G1 confirm eval seed=$sd (primary head:planck-rank)"
        $a = @("wikirace", "eval", "--policies", "random,lexical,heads,gemma", "--seed", "$sd", "--tag", $tt,
               "--primary", "head:planck-rank", "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK, "--gemma-path", $GEMMA)
        if ($Quick) { $a += @("--limit", "100", "--teacher-limit", "20") }
        if ($sd -ne $seeds[0]) { $a += "--no-latency" }
        P3 $a
    }
    if ($seeds.Count -gt 1) {
        Log "G1 confirm aggregate (seeds 3,4,5, primary head:planck-rank)"
        P3 @("wikirace", "aggregate", "--seeds", "3,4,5", "--tag", $tt, "--primary", "head:planck-rank")
    }
}

# ── results back to git (readable from any machine) ──────────────────────
function Do-Push {
    if ($NoPush) { Log "-NoPush: results left uncommitted"; return }
    Log "committing results/planck3 (summaries, REPORT.md, logs, compressed trajectories)"
    foreach ($pat in @("summary.json", "cards.md", "results.jsonl", "train_log.json", "trajectories.jsonl.gz",
                       "pairs_*.jsonl", "REPORT.md", "doctor.json", "digest.md", "*.log")) {
        Get-ChildItem -Path $RES -Recurse -Filter $pat -ErrorAction SilentlyContinue |
            ForEach-Object { Invoke-Quiet git @("add", "-f", "--", $_.FullName) | Out-Null }
    }
    if ((Invoke-Quiet git @("diff", "--cached", "--quiet")).Code -eq 0) { Log "nothing new to commit"; return }
    Invoke-Checked git @("commit", "-m", "results(planck3): run$Tag $(Get-Date -Format yyyy-MM-dd_HHmm)")
    Invoke-Checked git @("pull", "--rebase", "origin", "main")
    Invoke-Checked git @("push", "origin", "main")
}

# ── scheduled continuous retrieval ───────────────────────────────────────
function Do-Schedule {
    if (-not (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue)) { Warn "Task Scheduler cmdlets not available on this system"; return }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" digest -Explore"
    $trigger = New-ScheduledTaskTrigger -Daily -At $At
    Register-ScheduledTask -TaskName $TASK_NAME -Action $action -Trigger $trigger -Force `
        -Description "Planck 3.0: read adjacent entities from trusted sources into the local knowledge graph; writes results/planck3/digest.md" | Out-Null
    Log "scheduled '$TASK_NAME' daily at $At (runs only while you are logged on). Remove: .\scripts\planck3.ps1 unschedule"
}

function Do-Unschedule {
    if (Get-Command Unregister-ScheduledTask -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false -ErrorAction SilentlyContinue
        Log "removed scheduled task '$TASK_NAME'"
    }
}

switch ($Command.ToLower()) {
    "setup"      { Do-Setup }
    "doctor"     { Start-RunLog "doctor"; Do-Setup; Do-Doctor $Deep.IsPresent }
    "smoke"      { Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q") }
    "searxng"    { Do-Searxng }
    "ask"        {
        if (-not $Question) { throw 'usage: .\scripts\planck3.ps1 ask "your question"' }
        P3 (@("ask", $Question, "--type", $Type, "--policy", (Get-TeacherPolicy), "--gemma-path", $GEMMA, "-v") + $Rest)
    }
    { $_ -in @("chat", "serve") } {
        Do-Searxng
        if ($Command -eq "serve") { Log "open http://127.0.0.1:8010 in a browser (Ctrl+C to stop)" }
        P3 (@($Command.ToLower(), "--policy", (Get-TeacherPolicy), "--gemma-path", $GEMMA) + $Rest)
    }
    "g0"         { Start-RunLog "g0"; Do-Searxng; Do-G0; P3 @("report") }
    "g1"         { Start-RunLog "g1"; Do-G1; P3 @("report") }
    "report"     { P3 @("report") }
    "digest"     { $a = @("digest"); if ($Explore) { Do-Searxng; $a += "--explore" }; P3 ($a + $Rest) }
    "schedule"   { Do-Schedule }
    "unschedule" { Do-Unschedule }
    "py"         { P3 @((@($Question) + $Rest) | Where-Object { $_ }) }
    "round3"     {
        Start-RunLog "round3"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Doctor $true
        Do-Searxng
        Do-G0Fresh
        Do-G2
        if (-not $Quick) { Do-G1Confirm } else { Log "-Quick: G1 confirmation skipped" }
        P3 @("report")
        Log ("round3$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    "all"        {
        Start-RunLog "all"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Doctor $true
        Do-Searxng; Do-G0; Do-G1
        P3 @("report")
        Log ("all$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    default      { Get-Help $PSCommandPath -Detailed | Out-String | Write-Host }
}
