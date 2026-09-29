<#
 .SYNOPSIS
   Planck 3.0 one-stop runner for the Windows 4090 box. Plan: SETUP_092026_planck3.md

 .DESCRIPTION
   Every stage is idempotent: finished outputs are skipped, so re-running after
   an interruption resumes. Start with `all` (or `setup` then `smoke`).

     all       setup -> smoke -> searxng -> g0 -> g1 -> report -> commit+push results
     setup     check/install deps into .venv, list what is missing (checkpoints, Gemma)
     smoke     offline unit + end-to-end tests (no network, no GPU), ~30s
     searxng   start the local SearXNG search container (Docker Desktop)
     chat      terminal chat with follow-ups ("and H&M?", "when was it founded?")
     serve     local web chat at http://127.0.0.1:8010 (open it in a browser)
     ask       one question -> answer card:   .\scripts\planck3.ps1 ask "Who founded SpaceX?" -Type entity
     g0        G0 teacher ceiling: seed benchmark with the Gemma teacher (-Policy heuristic|gemma|bedrock)
     g1        G1 Wikiracing: build graph -> tasks -> embed -> train heads -> eval (+ Gemma teacher)
     report    print every G0/G1 summary
     py        pass-through: .\scripts\planck3.ps1 py wikirace eval --limit 50

 .EXAMPLE
   powershell -ExecutionPolicy Bypass -File scripts\planck3.ps1 all
 .EXAMPLE
   .\scripts\planck3.ps1 g1 -Seed 1          # extra seed (reseed before believing a delta)
 .EXAMPLE
   .\scripts\planck3.ps1 all -NoPush         # compute only, don't commit results
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Command = "help",
    [Parameter(Position = 1)][string]$Question = "",
    [string]$Type = "entity",
    [string]$Policy = "gemma",
    [int]$Seed = 0,
    [int]$Limit = 0,
    [switch]$NoPush,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

$env:PYTHONUTF8 = "1"                  # page text is not cp1252
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$PLANCK_CKPT = "checkpoints/planck13/best.pt"
$PLANCK_TOK  = "data/wikipedia/tokenizer.model"
$GEMMA       = "models/gemma-4-e4b-it"
$WR          = "data/planck3/wikirace"
$RES         = "results/planck3"
$SEARX_NAME  = "planck3-searxng"
$SEARX_PORT  = 8888

function Log([string]$msg) { Write-Host ""; Write-Host "[planck3 $(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor Cyan }
function Warn([string]$msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }

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

# $Rest may be $null; drop empty elements so python never sees a stray "" argument
function P3 { param([string[]]$A) Invoke-Checked $PY (@("scripts/planck3.py") + @($A | Where-Object { $_ })) }

function Test-Docker { return [bool](Get-Command docker -ErrorAction SilentlyContinue) }
function Test-Searx { return (Invoke-Quiet $PY @("scripts/planck3.py", "search-check")).Code -eq 0 }

# ── setup ────────────────────────────────────────────────────────────────
function Do-Setup {
    Log "installing Planck 3.0 deps into .venv"
    Invoke-Checked $PY @("-m", "pip", "install", "--quiet", "--upgrade", "trafilatura", "requests", "scipy", "pytest")
    $torch = (Invoke-Quiet $PY @("-c", "import torch;print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'CPU-ONLY')")).Out
    $tf = (Invoke-Quiet $PY @("-c", "import transformers;print(transformers.__version__)")).Out
    Log "environment check"
    Write-Host ("  python        {0}" -f (& $PY --version))
    Write-Host ("  torch         {0}" -f $(if ($torch) { $torch } else { "MISSING (pip install torch --index-url https://download.pytorch.org/whl/cu124)" }))
    Write-Host ("  transformers  {0}" -f $(if ($tf) { $tf } else { "MISSING (needed for the Gemma teacher)" }))
    foreach ($p in @($PLANCK_CKPT, $PLANCK_TOK, $GEMMA)) {
        $ok = Test-Path $p
        Write-Host ("  {0,-34} {1}" -f $p, $(if ($ok) { "ok" } else { "MISSING" })) -ForegroundColor $(if ($ok) { "Green" } else { "Yellow" })
    }
    if (-not (Test-Path $PLANCK_CKPT)) { Warn "head:planck needs the Planck 1.3 checkpoint; G1 will still run hash/lexical/random + Gemma" }
    if (-not (Test-Path $GEMMA)) { Warn "Gemma teacher missing; download: huggingface-cli download google/gemma-4-E4B-it --local-dir $GEMMA" }
    Write-Host ("  docker        {0}" -f $(if (Test-Docker) { "ok" } else { "MISSING (install Docker Desktop for SearXNG; Wikipedia search is the fallback)" }))
}

# ── searxng ──────────────────────────────────────────────────────────────
function Do-Searxng {
    if (-not (Test-Docker)) { Warn "Docker not found; G0 will fall back to Wikipedia search (still valid, narrower)"; return }
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
    Warn "SearXNG did not answer in 60s; check: docker logs $SEARX_NAME  (G0 falls back to Wikipedia)"
}

# ── G0 ───────────────────────────────────────────────────────────────────
function Do-G0 {
    $out = "$RES/g0_$Policy"
    if ((Test-Path "$out/summary.json") -and $Command -eq "all") { Log "SKIP g0 (exists): $out/summary.json"; return }
    if ($Policy -eq "gemma" -and -not (Test-Path $GEMMA)) { Warn "no Gemma at $GEMMA; running heuristic baseline only"; $script:Policy = "heuristic"; $out = "$RES/g0_heuristic" }
    Log "G0 teacher ceiling: policy=$Policy"
    $a = @("g0", "--policy", $Policy, "--out", $out, "--gemma-path", $GEMMA)
    if ($Limit -gt 0) { $a += @("--limit", "$Limit") }
    P3 $a
    if ($Policy -ne "heuristic" -and -not (Test-Path "$RES/g0_heuristic/summary.json")) {
        Log "G0 heuristic baseline (for comparison)"
        P3 @("g0", "--policy", "heuristic", "--out", "$RES/g0_heuristic")
    }
}

# ── G1 ───────────────────────────────────────────────────────────────────
function Do-G1 {
    if (-not (Test-Path "$WR/graph.npz")) { Log "G1 build: Simple English Wikipedia link graph"; P3 @("wikirace", "build") }
    else { Log "SKIP build (exists): $WR/graph.npz" }
    if (-not (Test-Path "$WR/tasks_info.json")) { Log "G1 tasks: BFS labels, split by target"; P3 @("wikirace", "tasks") }
    else { Log "SKIP tasks (exists)" }
    if (-not (Test-Path "$WR/emb_hash.npy")) { Log "G1 embed: hash control"; P3 @("wikirace", "embed", "--encoder", "hash") }
    $havePlanck = Test-Path $PLANCK_CKPT
    if ($havePlanck -and -not (Test-Path "$WR/emb_planck.npy")) {
        Log "G1 embed: frozen Planck 1.3"
        P3 @("wikirace", "embed", "--encoder", "planck", "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK)
    }
    $encs = @("hash"); if ($havePlanck) { $encs += "planck" }
    foreach ($e in $encs) {
        if (-not (Test-Path "$RES/g1_head_${e}_s$Seed/head.pt")) {
            Log "G1 train head:$e seed=$Seed"
            P3 @("wikirace", "train", "--encoder", $e, "--seed", "$Seed")
        } else { Log "SKIP train head:$e s$Seed (exists)" }
    }
    $pols = @("random", "lexical", "head:hash")
    if ($havePlanck) { $pols += "head:planck" }
    if (Test-Path $GEMMA) { $pols += "gemma" } else { Warn "no Gemma teacher; G1 verdict will be INCOMPLETE" }
    Log "G1 eval: $($pols -join ', ')"
    $a = @("wikirace", "eval", "--policies", ($pols -join ","), "--seed", "$Seed",
           "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK, "--gemma-path", $GEMMA)
    if ($Limit -gt 0) { $a += @("--limit", "$Limit") }
    P3 $a
}

function Do-Push {
    if ($NoPush) { Log "-NoPush: results left uncommitted"; return }
    Log "committing results/planck3 summaries"
    foreach ($pat in @("summary.json", "cards.md", "results.jsonl", "train_log.json")) {
        Get-ChildItem -Path $RES -Recurse -Filter $pat -ErrorAction SilentlyContinue |
            ForEach-Object { Invoke-Quiet git @("add", "--", $_.FullName) | Out-Null }
    }
    if ((Invoke-Quiet git @("diff", "--cached", "--quiet")).Code -eq 0) { Log "nothing new to commit"; return }
    Invoke-Checked git @("commit", "-m", "results(planck3): G0/G1 run $(Get-Date -Format yyyy-MM-dd)")
    Invoke-Checked git @("pull", "--rebase", "origin", "main")
    Invoke-Checked git @("push", "origin", "main")
}

switch ($Command.ToLower()) {
    "setup"   { Do-Setup }
    "smoke"   { Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q") }
    "searxng" { Do-Searxng }
    "ask"     {
        if (-not $Question) { throw 'usage: .\scripts\planck3.ps1 ask "your question" -Type year|number|date|entity|text' }
        $p = if ($Policy -eq "gemma" -and -not (Test-Path $GEMMA)) { "heuristic" } else { $Policy }
        P3 (@("ask", $Question, "--type", $Type, "--policy", $p, "--gemma-path", $GEMMA, "-v") + $Rest)
    }
    { $_ -in @("chat", "serve") } {
        Do-Searxng
        $p = if ($Policy -eq "gemma" -and -not (Test-Path $GEMMA)) { "heuristic" } else { $Policy }
        if ($Command -eq "serve") { Log "open http://127.0.0.1:8010 in a browser (Ctrl+C to stop)" }
        P3 (@($Command.ToLower(), "--policy", $p, "--gemma-path", $GEMMA) + $Rest)
    }
    "g0"      { Do-Searxng; Do-G0 }
    "g1"      { Do-G1 }
    "report"  { P3 @("report") }
    "py"      { P3 @((@($Question) + $Rest) | Where-Object { $_ }) }
    "all"     {
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Searxng; Do-G0; Do-G1
        P3 @("report"); Do-Push
    }
    default   { Get-Help $PSCommandPath -Detailed | Out-String | Write-Host }
}
