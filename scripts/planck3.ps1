<#
 .SYNOPSIS
   Planck 3.0 one-stop runner for the Windows 4090 box.
   Step-by-step guide: SETUP_planck_20260901.md   Plan: SETUP_092026_planck3.md

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
     g0        G0: seed benchmark: Gemma on tools, Gemma closed-book (the base-chat rival), heuristic floor
     g1        G1: Wikiracing: graph -> tasks -> embed -> heads -> eval vs random/lexical/Gemma
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
    [int]$Limit = 0,
    [string]$At = "08:00",
    [switch]$Quick,
    [switch]$Deep,
    [switch]$NoPush,
    [switch]$Explore,
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
$GEMMA       = "models/gemma-4-e4b-it"
$WR          = "data/planck3/wikirace"
$RES         = "results/planck3"
$SEARX_NAME  = "planck3-searxng"
$SEARX_PORT  = 8888
$TASK_NAME   = "Planck3Explore"
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
    Invoke-Checked $PY @("-m", "pip", "install", "--quiet", "trafilatura", "requests", "scipy", "pytest", "sentencepiece")
}

function Do-Doctor([bool]$DeepRun) {
    $a = @("doctor"); if ($DeepRun) { $a += "--deep" }
    try { P3 $a } catch { throw "doctor found blocking problems (FAIL rows above). Fix them, then re-run." }
}

# ── searxng ──────────────────────────────────────────────────────────────
function Do-Searxng {
    if (-not (Test-Docker)) { Warn "Docker not found; search falls back to the Wikipedia API (still valid, narrower)"; return }
    if (Test-Searx) { Log "SearXNG already up on :$SEARX_PORT"; return }
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
    $runs = @(@{ Name = "g0_$pol$Tag"; Args = @("--policy", $pol); What = "G0 teacher on our tools: $pol" })
    if ($pol -ne "heuristic") {
        $runs += @{ Name = "g0_${pol}_closedbook$Tag"; Args = @("--policy", $pol, "--closed-book"); What = "G0 base-chat rival: $pol closed-book (no tools, no sources)" }
        $runs += @{ Name = "g0_heuristic$Tag"; Args = @("--policy", "heuristic"); What = "G0 heuristic floor" }
    }
    foreach ($r in $runs) {
        $out = "$RES/$($r.Name)"
        if ((Test-Path "$out/summary.json") -and $Command -eq "all") { Log "SKIP $($r.Name) (exists)"; continue }
        Log $r.What
        P3 (@("g0") + $r.Args + @("--out", $out, "--gemma-path", $GEMMA) + $sample)
    }
}

# ── G1 ───────────────────────────────────────────────────────────────────
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
    $td = if ($Quick) { "$WR/tasks_quick" } else { $WR }
    $tagArg = if ($Quick) { @("--tag", $Tag) } else { @() }   # never pass an empty --tag value
    $tasksArgs = if ($Quick) { @("--targets", "300") } else { @() }
    if (-not (Test-Path "$td/tasks_info.json")) { Log "G1 tasks${Tag}: BFS labels, split by target"; P3 (@("wikirace", "tasks") + $tagArg + $tasksArgs) }
    else { Log "SKIP tasks$Tag (exists)" }
    $encs = @("hash"); if ($havePlanck) { $encs += "planck" }
    $epochs = if ($Quick) { @("--epochs", "2") } else { @() }
    foreach ($e in $encs) {
        if (-not (Test-Path "$RES/g1_head_${e}_s$Seed$Tag/head.pt")) {
            Log "G1 train head:$e seed=$Seed$Tag"
            P3 (@("wikirace", "train", "--encoder", $e, "--seed", "$Seed") + $tagArg + $epochs)
        } else { Log "SKIP train head:$e s$Seed$Tag (exists)" }
    }
    $pols = @("random", "lexical", "head:hash")
    if ($havePlanck) { $pols += "head:planck" } else { Warn "no Planck checkpoint/tokenizer; G1 verdict will be INCOMPLETE" }
    if (Test-Path $GEMMA) { $pols += "gemma" } else { Warn "no Gemma teacher; G1 verdict will be INCOMPLETE" }
    Log "G1 eval${Tag}: $($pols -join ', ')"
    $a = @("wikirace", "eval", "--policies", ($pols -join ","), "--seed", "$Seed",
           "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK, "--gemma-path", $GEMMA) + $tagArg
    if ($Quick) { $a += @("--limit", "100", "--teacher-limit", "20") } elseif ($Limit -gt 0) { $a += @("--limit", "$Limit") }
    P3 $a
}

# ── results back to git (readable from any machine) ──────────────────────
function Do-Push {
    if ($NoPush) { Log "-NoPush: results left uncommitted"; return }
    Log "committing results/planck3 (summaries, REPORT.md, logs, compressed trajectories)"
    foreach ($pat in @("summary.json", "cards.md", "results.jsonl", "train_log.json", "trajectories.jsonl.gz",
                       "REPORT.md", "doctor.json", "digest.md", "*.log")) {
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
