<#
 .SYNOPSIS
   Planck 3.0 one-stop runner for the Windows 4090 box.
   Step-by-step guides: SETUP_planck_20260901.md (rounds 1-2), SETUP_planck_20260903.md (round 3),
   SETUP_planck_20261008.md (round 4), SETUP_planck_20261009.md (round 5), SETUP_planck_20261012.md (round 6),
   SETUP_planck_20261014.md (round 7), SETUP_planck_20261015.md (round 8)
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
     round4    round 4 (SETUP_planck_20261008.md): ddgs search (no Docker) on the fresh benchmark, answer
               tiers (confident / low confidence / evidence only), G2 v2 retrained on ddgs, system trust layer
     round5    round 5 (SETUP_planck_20261009.md): read -> weigh -> write on 3 benchmarks (fresh, news,
               general info); hand-scored 0-10 source trust; writers trained once (Hertz / Planck, from Gemma);
               worst-case personal profile; Brave vs ddgs; CPU latency (phone proxy)
     round6    round 6 (SETUP_planck_20261012.md): Brave + relevance check as the product search; the reader
               trained once as a cross-encoder (Planck and Hertz) on 2025 news, recent events and general
               information; writers retrained on fixed data with a forced lead, 3 tries and the KV cache
     round6b   round 6b (SETUP_planck_20261012.md section 10): the round 6 readers retrained on the top 16
               candidates as served (fixes the last-slot leak), same points, no new searches; ~1.5 h
     round7    round 7 (SETUP_planck_20261014.md): reading taught as a skill (SQuAD, NewsQA, TriviaQA,
               SearchQA, NQ, SQuAD 2.0 unanswerable -> reader pools), then our pools; Planck + Hertz;
               rebuild gate N (does Planck read news?); no new searches; ~4-6 h
     round8    round 8 (SETUP_planck_20261015.md): Hertz's reader on a phone proxy first (CPU, 4 threads,
               fp32 / int8, top 16 / top 8; ~1 h, pushed at once), then Planck 1.5 = Planck's size pretrained
               on Hertz's corpus (~25 h, resumable), round 7's reading stage on it, the benchmark; no new searches
     setkey    store an API key in .env (gitignored, never pushed), input hidden: .\scripts\planck3.ps1 setkey
               (default BRAVE_API_KEY; another name: setkey OTHER_KEY). Then checks the Brave API answers
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

# .env (gitignored; template .env.example): API keys for every child process. The real environment wins.
if (Test-Path (Join-Path $Root ".env")) {
    foreach ($line in Get-Content (Join-Path $Root ".env")) {
        if ($line.TrimStart().StartsWith("#")) { continue }
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $k = $Matches[1]; $v = $Matches[2].Trim().Trim('"').Trim("'")
            if ($v -and -not [Environment]::GetEnvironmentVariable($k)) { [Environment]::SetEnvironmentVariable($k, $v, "Process") }
        }
    }
}

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
    Invoke-PyLogged (@("scripts/planck3.py") + @($A | Where-Object { $_ }))
}

# Any python script, teed into the run log like P3 (round 8: the multi-day pretraining run).
function Invoke-PyLogged {
    param([string[]]$argv)
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
    # ddgs: round 4 search (metasearch library, no Docker); scikit-learn: the G2 v2 gate
    Invoke-Checked $PY @("-m", "pip", "install", "--quiet", "trafilatura", "lxml_html_clean", "requests", "scipy", "pytest", "sentencepiece", "ddgs", "scikit-learn", "pyarrow", "huggingface_hub")
}

function Do-Doctor([bool]$DeepRun) {
    $a = @("doctor"); if ($DeepRun) { $a += "--deep" }
    try { P3 $a } catch { throw "doctor found blocking problems (FAIL rows above). Fix them, then re-run." }
}

# ── searxng ──────────────────────────────────────────────────────────────
$script:SearxUp = $false
function Do-Searxng {
    if (Test-Docker) {
        $cfgHash = (Get-FileHash "config/searxng/settings.yml" -Algorithm SHA1).Hash
        $stamp = "data/planck3/searxng_settings.sha1"
        $applied = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { "" }
        if ($applied -ne $cfgHash) {
            $exists = (Invoke-Quiet docker @("ps", "-a", "--filter", "name=^$SEARX_NAME`$", "--format", "{{.Names}}")).Out -eq $SEARX_NAME
            if ($exists) { Log "config/searxng/settings.yml changed: recreating the SearXNG container"; Invoke-Quiet docker @("rm", "-f", $SEARX_NAME) | Out-Null }
            New-Item -ItemType Directory -Force -Path "data/planck3" | Out-Null
            Set-Content -Path $stamp -Value $cfgHash -Encoding ASCII
        }
    }
    $script:SearxUp = Test-Searx
    if (-not (Test-Docker)) { Warn "Docker not found; search falls back to the Wikipedia API (still valid, narrower)"; return }
    if (Test-Searx) { Log "SearXNG up and answering on :$SEARX_PORT"; return }
    $running = (Invoke-Quiet docker @("ps", "--filter", "name=^$SEARX_NAME`$", "--format", "{{.Names}}")).Out -eq $SEARX_NAME
    if ($running) {
        # up but returning nothing = upstream engines rate-limit it; blocks are usually temporary
        Warn "SearXNG is running but returns no results (upstream engines are rate-limiting it); waiting up to 10 min"
        for ($i = 0; $i -lt 10; $i++) {
            Start-Sleep -Seconds 60
            if (Test-Searx) { Log "SearXNG answers again"; $script:SearxUp = $true; return }
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
    # a fresh container pulls the image and warms its engines: give it up to 5 min (round 3 attempt 2
    # gave up after 60 s and the whole run fell back to Wikipedia search)
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 5
        if (Test-Searx) { Log "SearXNG is up on http://localhost:$SEARX_PORT"; $script:SearxUp = $true; return }
        if ($i % 12 -eq 11) { Log "  still waiting for SearXNG ($([int](($i + 1) * 5 / 60)) min)" }
    }
    Warn "SearXNG did not answer in 5 min (search falls back to Wikipedia). Last container log lines:"
    $logs = (Invoke-Quiet docker @("logs", "--tail", "15", $SEARX_NAME)).Out
    if ($logs) { $logs | ForEach-Object { Write-Host "    $_"; Write-Log "    $_" } }
}

# ── ddgs (round 4 search of record) ──────────────────────────────────────
$script:DdgsUp = $false
function Test-Ddgs { return (Invoke-Quiet $PY @("scripts/planck3.py", "search-check", "--backend", "ddgs")).Code -eq 0 }
function Do-Ddgs([bool]$Required) {
    if (Test-Ddgs) { Log "ddgs answering (no Docker needed)"; $script:DdgsUp = $true; return }
    Warn "ddgs returned nothing (search engines rate-limit scripted clients); re-checking every 2 min for up to 10 min"
    for ($i = 0; $i -lt 5; $i++) {
        Start-Sleep -Seconds 120
        if (Test-Ddgs) { Log "ddgs answers again"; $script:DdgsUp = $true; return }
    }
    if ($Required) { throw "ddgs is not answering. Round 4 measures ddgs search, so it stops instead of running on the Wikipedia fallback. Try again in an hour; finished work is kept." }
    Warn "ddgs still not answering: trying SearXNG"; Do-Searxng
}
# chat / serve / explore: ddgs if it answers, else SearXNG, else Wikipedia search
function Do-Search { if (Test-Ddgs) { Log "search: ddgs"; $script:DdgsUp = $true } else { Do-Searxng } }

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
    if (-not (($null -ne $j.search_health) -and [bool]$j.search_health.valid)) { return $false }
    # SearXNG is the retrieval layer of record: a tools run on the Wikipedia fallback is kept as
    # <dir>_wikipedia and redone once SearXNG answers (round 3 attempt 2 ran entirely on Wikipedia)
    # a "searxng" run that SearXNG did not actually serve (cache hits included): until round 3b the
    # fallback cache was read first, so most fresh-benchmark "SearXNG" runs were really Wikipedia
    $share = $j.search_health.primary_share
    if ($j.search -eq "searxng" -and $script:SearxUp -and (($null -eq $share) -or ($share -lt 0.8)) -and (Test-Searx)) {
        $aside = "${dir}_mixed"
        if (-not (Test-Path $aside)) { Move-Item $dir $aside; Log "kept $dir as $aside (SearXNG served too little of it); redoing it" }
        else { Remove-Item -Recurse -Force $dir }
        return $false
    }
    if ($j.search -ne "searxng" -and $script:SearxUp -and (Test-Searx)) {  # re-check NOW: it can be blocked mid-round
        $aside = "${dir}_wikipedia"
        if (-not (Test-Path $aside)) { Move-Item $dir $aside; Log "kept the Wikipedia-search run as $aside; redoing it on SearXNG" }
        else { Remove-Item -Recurse -Force $dir }
        return $false
    }
    return $true
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
    # G2 v2 (2026-10-08): choice head + calibrated answerability gate, trained on points collected
    # with the deployment search (SearXNG). v1 (round 3) stays on record; it is not re-run.
    if (-not (Test-Path $G2_TRAIN)) { throw "missing $G2_TRAIN (git pull: generated on the Mac from Wikidata and committed)" }
    $lim = if ($Quick) { @("--limit", "100") } else { @() }
    $backend = if ($script:SearxUp) { "searxng" } else { "wikipedia" }
    $pts = "data/planck3/g2/points_$backend.jsonl"
    if ($backend -ne "searxng") {
        Warn "G2 v2 wants SearXNG-collected training data (train = deploy); SearXNG is not answering, using Wikipedia"
        if ((Test-Path "data/planck3/g2/points.jsonl") -and -not (Test-Path $pts)) { Copy-Item "data/planck3/g2/points.jsonl" $pts }
    }
    $health = "data/planck3/g2/points_$backend.health.json"
    if ($backend -eq "searxng" -and (Test-Path $pts)) {
        $h = if (Test-Path $health) { Get-Content $health -Raw | ConvertFrom-Json } else { $null }
        if (($null -eq $h) -or ($null -eq $h.primary_share) -or ($h.primary_share -lt 0.8)) {
            # collected before the cache-order fix: mostly Wikipedia results under a SearXNG name
            Move-Item -Force $pts "data/planck3/g2/points_searxng_mixed.jsonl"
            if (Test-Path $health) { Move-Item -Force $health "data/planck3/g2/points_searxng_mixed.health.json" }
            Log "G2 points were not really SearXNG-served: kept as points_searxng_mixed.jsonl, collecting again"
        }
    }
    Log "G2 collect ($backend): labelled decision points from known answers (resumable)"
    P3 (@("g2", "collect", "--points", $pts) + $lim)
    $encs = @("hash"); if ((Test-Path $PLANCK_CKPT) -and (Test-Path $PLANCK_TOK)) { $encs += "planck" }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pts)
    foreach ($e in $encs) {
        $emb = "data/planck3/g2/emb_${e}_$stem.npy"
        $head = "$RES/g2v2_head_${e}_s0/head.pt"
        if ((Test-Newer $pts $emb) -or $Quick) {
            Log "G2 embed: $e ($stem)"; P3 @("g2", "embed", "--encoder", $e, "--points", $pts, "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK)
        }
        if ((Test-Newer $emb $head) -or $Quick) { Log "G2 v2 train head:$e"; P3 @("g2", "train2", "--encoder", $e, "--points", $pts) }
        foreach ($b in @(@{ P = "g0f"; T = $FRESH }, @{ P = "g0"; T = "scripts/assets/planck3_tasks.json" })) {
            $out = "$RES/$($b.P)_planck-g2v2-${e}$Tag"
            if ((Test-ValidRun $out) -and -not (Test-Newer $head "$out/summary.json") -and -not $Quick) {
                Log "SKIP $out (valid, head unchanged)"; continue
            }
            Log "G2 v2 eval head:$e on $($b.T)"
            $smp = if ($Quick) { @("--sample", "12") } else { @() }
            P3 (@("g0", "--policy", "planck", "--g2-head", $head, "--tasks", $b.T, "--out", $out,
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

# ── round 4 (SETUP_planck_20261008.md) ────────────────────────────────────
# A ddgs run is valid if ddgs served >= 80% of its searches and <= 20% came back empty. A run that
# misses that is set aside ONCE as <dir>_degraded and redone; a second miss is kept (DEGRADED in the report).
function Test-Round4Run([string]$dir) {
    if (-not (Test-Path "$dir/summary.json")) { return $false }
    $j = Get-Content "$dir/summary.json" -Raw | ConvertFrom-Json
    if ($j.mode -eq "closed_book") { return $true }
    $h = $j.search_health
    $ok = ($null -ne $h) -and [bool]$h.valid -and ($j.search -eq "ddgs") -and ($null -ne $h.primary_share) -and ($h.primary_share -ge 0.8)
    if ($ok) { return $true }
    if (Test-Path "${dir}_degraded") { Warn "$dir is below the bar again (ddgs served $($h.primary_share)); keeping it, reported as DEGRADED"; return $true }
    Move-Item $dir "${dir}_degraded"; Log "kept $dir as ${dir}_degraded (ddgs served too little of it); redoing it"
    return $false
}

function Invoke-R4Run([string]$name, [string[]]$a, [string]$what) {
    $out = "$RES/$name"
    if (Test-Round4Run $out) { Log "SKIP $name (valid)"; return }
    Log $what
    $smp = if ($Quick) { @("--sample", "12") } else { @() }
    P3 (@("g0") + $a + @("--search", "ddgs", "--out", $out, "--gemma-path", $GEMMA) + $smp)
}

function Do-Round4 {
    $pol = Get-TeacherPolicy
    # 1. the search change alone (system trust OFF = same reading order as round 3): rules S2, A1, P
    if ($pol -ne "heuristic" -and -not (Test-Path "$RES/g0f_${pol}_closedbook/summary.json")) {
        Log "fresh/long-tail: base chat ($pol closed-book)"
        P3 @("g0", "--policy", $pol, "--closed-book", "--tasks", $FRESH, "--out", "$RES/g0f_${pol}_closedbook", "--gemma-path", $GEMMA)
    }
    Invoke-R4Run "g0f_heuristic_snip_ddgs$Tag" @("--policy", "heuristic", "--tasks", $FRESH, "--system-trust", "none") "round 4: heuristic on ddgs (rule S2, P)"
    if ($pol -ne "heuristic") {
        Invoke-R4Run "g0f_${pol}_snip_ddgs$Tag" @("--policy", $pol, "--tasks", $FRESH, "--system-trust", "none") "round 4: $pol on our tools, ddgs (rules S2, A1, P)"
    }
    # 2. G2 v2 trained on what deployment sees (ddgs), then the system trust layer from the same points
    if (-not (Test-Path $G2_TRAIN)) { throw "missing $G2_TRAIN (git pull)" }
    $pts = "data/planck3/g2/points_ddgs$Tag.jsonl"
    $lim = if ($Quick) { @("--limit", "100") } else { @() }
    Log "G2 collect on ddgs: labelled decision points from known answers (resumable; ~2 h first time)"
    P3 (@("g2", "collect", "--search", "ddgs", "--points", $pts) + $lim)
    Log "system trust layer from $pts -> config/planck3/source_trust.json"
    P3 @("trust", "build", "--points", $pts)
    $encs = @("hash"); if ((Test-Path $PLANCK_CKPT) -and (Test-Path $PLANCK_TOK)) { $encs += "planck" }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pts)
    foreach ($e in $encs) {
        $emb = "data/planck3/g2/emb_${e}_$stem.npy"
        $head = "$RES/g2v2_head_${e}_s0_ddgs$Tag/head.pt"
        if ((Test-Newer $pts $emb) -or $Quick) {
            Log "G2 embed: $e ($stem)"; P3 @("g2", "embed", "--encoder", $e, "--points", $pts, "--checkpoint", $PLANCK_CKPT, "--tokenizer", $PLANCK_TOK)
        }
        if ((Test-Newer $emb $head) -or $Quick) { Log "G2 v2 train head:$e on ddgs points"; P3 @("g2", "train2", "--encoder", $e, "--points", $pts, "--tag", "_ddgs$Tag") }
        Invoke-R4Run "g0f_planck-g2v2-${e}_ddgs$Tag" @("--policy", "planck", "--g2-head", $head, "--tasks", $FRESH,
            "--planck-checkpoint", $PLANCK_CKPT, "--planck-tokenizer", $PLANCK_TOK) "round 4: G2 v2 head:$e (ddgs-trained) on the fresh benchmark (rules B, P)"
        Invoke-R4Run "g0_planck-g2v2-${e}_ddgs$Tag" @("--policy", "planck", "--g2-head", $head, "--tasks", "scripts/assets/planck3_tasks.json",
            "--planck-checkpoint", $PLANCK_CKPT, "--planck-tokenizer", $PLANCK_TOK) "round 4: G2 v2 head:$e on the 47-question seed benchmark (rule B)"
    }
    # 3. the same heuristic WITH the system trust layer: rule T (does measured source trust help?)
    Invoke-R4Run "g0f_heuristic_snip_ddgs_trust$Tag" @("--policy", "heuristic", "--tasks", $FRESH) "round 4: heuristic on ddgs + system trust layer (rule T)"
}

# ── round 5 (SETUP_planck_20261009.md) ────────────────────────────────────
$NEWS   = "scripts/assets/planck3_tasks_news.json"
$INFO   = "scripts/assets/planck3_tasks_geninfo.json"
$V2HEAD = "$RES/g2v2_head_planck_s0_ddgs/head.pt"     # the round 4 choice head (trained once): the Planck reader

function Test-R5Run([string]$dir, [string]$backend) {
    if (-not (Test-Path "$dir/summary.json")) { return $false }
    $j = Get-Content "$dir/summary.json" -Raw | ConvertFrom-Json
    $h = $j.search_health
    $ok = ($null -ne $h) -and [bool]$h.valid -and ($j.search -eq $backend) -and ($null -ne $h.primary_share) -and ($h.primary_share -ge 0.8)
    if ($ok) { return $true }
    if (Test-Path "${dir}_degraded") { Warn "$dir is below the search bar again; keeping it (reported as is)"; return $true }
    Move-Item $dir "${dir}_degraded"; Log "kept $dir as ${dir}_degraded ($backend served too little of it); redoing it"
    return $false
}

function Invoke-R5([string]$name, [string[]]$a, [string]$what, [string]$backend = "ddgs", [int]$sample = 0) {
    $out = "$RES/$name$Tag"
    if (Test-R5Run $out $backend) { Log "SKIP $name (valid)"; return }
    Log $what
    $smp = if ($Quick) { @("--sample", "12") } elseif ($sample -gt 0) { @("--sample", "$sample") } else { @() }
    P3 (@("r5") + $a + @("--tasks", "$FRESH,$NEWS,$INFO", "--search", $backend, "--out", $out, "--gemma-path", $GEMMA,
          "--g2-head", $V2HEAD, "--planck-checkpoint", $PLANCK_CKPT, "--planck-tokenizer", $PLANCK_TOK) + $smp)
}

function Do-Round5 {
    foreach ($f in @($NEWS, $INFO, $V2HEAD, $PLANCK_CKPT)) { if (-not (Test-Path $f)) { throw "missing $f (git pull; the G2 v2 ddgs head is from round 4)" } }
    $haveGemma = Test-Path $GEMMA
    $haveHertz = (Test-Path $HERTZ_CKPT) -and (Test-Path $HERTZ_TOK)
    # 1. no-model floor, and the Planck reader with the template writer (isolates reading from writing)
    Invoke-R5 "r5_heuristic_template" @("--reader", "heuristic", "--writer", "template") "round 5: no-model floor (heuristic reader, template writer)"
    Invoke-R5 "r5_planck_template" @("--reader", "planck", "--writer", "template") "round 5: Planck reader, template writer"
    # 2. one-time writer training set: Gemma writes for G2 TRAINING questions (never a benchmark), faithfulness-filtered
    $wdata = "data/planck3/writer/train$Tag.jsonl"
    if ($haveGemma) {
        $lim = if ($Quick) { @("--limit", "60") } else { @("--limit", "1200") }
        Log "writer data: Gemma writes answers for G2 training questions (resumable; ~1-1.5 h first time)"
        P3 (@("writer", "collect", "--out", $wdata, "--search", "ddgs", "--reader", "planck", "--g2-head", $V2HEAD,
              "--planck-checkpoint", $PLANCK_CKPT, "--planck-tokenizer", $PLANCK_TOK, "--gemma-path", $GEMMA) + $lim)
    } else { Warn "no Gemma at ${GEMMA}: no writer training data, so no Hertz / Planck writer this round" }
    # 3. train the small writers ONCE (never pass an empty --tag: PowerShell 5.1 drops empty arguments)
    $wtag = if ($Tag) { @("--tag", $Tag) } else { @() }
    $writers = @()
    if ((Test-Path $wdata) -and $haveHertz) {
        if (Test-Newer $wdata "checkpoints/writer_hertz$Tag/best.pt") {
            Log "train the Hertz 1.2 writer (640M), once"
            P3 (@("writer", "train", "--base", "hertz", "--out", $wdata, "--base-checkpoint", $HERTZ_CKPT, "--base-tokenizer", $HERTZ_TOK, "--batch-size", "4") + $wtag)
        }
        $writers += "hertz"
    } elseif (-not $haveHertz) { Warn "no Hertz checkpoint: the Hertz writer arm is skipped" }
    if (Test-Path $wdata) {
        if (Test-Newer $wdata "checkpoints/writer_planck$Tag/best.pt") {
            Log "train the Planck 1.3 writer (100M), once"
            P3 (@("writer", "train", "--base", "planck", "--out", $wdata, "--base-checkpoint", $PLANCK_CKPT, "--base-tokenizer", $PLANCK_TOK) + $wtag)
        }
        $writers += "planck"
    }
    # 4. the product candidates, the Gemma reference, and the checks around the product
    foreach ($w in $writers) {
        Invoke-R5 "r5_planck_$w" @("--reader", "planck", "--writer", $w, "--writer-checkpoint", "checkpoints/writer_$w$Tag/best.pt") "round 5: Planck reader + $w writer (small-model product)"
    }
    if ($haveGemma) { Invoke-R5 "r5_gemma_gemma" @("--reader", "gemma", "--writer", "gemma") "round 5: Gemma reader + Gemma writer (reference)" }
    $prod = if ($writers.Count) { $writers[0] } else { "template" }
    $pa = @("--reader", "planck", "--writer", $prod) + $(if ($prod -ne "template") { @("--writer-checkpoint", "checkpoints/writer_$prod$Tag/best.pt") } else { @() })
    Invoke-R5 "r5_planck_${prod}_contrarian" ($pa + @("--personal", "contrarian")) "round 5: worst-case personal profile (every source scored 10 - system): rule E"
    if ($env:BRAVE_API_KEY) { Invoke-R5 "r5_planck_${prod}_brave" $pa "round 5: the same on the Brave Search API (paired vs ddgs)" "brave" }
    else { Warn "no BRAVE_API_KEY (.\scripts\planck3.ps1 setkey): the Brave comparison is skipped" }
    # 5. CPU only, 4 threads: a phone proxy for the small models (searches and pages are cached by now)
    $old = $env:CUDA_VISIBLE_DEVICES; $env:CUDA_VISIBLE_DEVICES = "-1"; $env:OMP_NUM_THREADS = "4"
    try { Invoke-R5 "r5_planck_${prod}_cpu" $pa "round 5: CPU-only latency, 4 threads, 10 questions (phone proxy; no KV cache yet)" "ddgs" 10 }
    finally { if ($old) { $env:CUDA_VISIBLE_DEVICES = $old } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }; Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue }
}

# ── round 6 (SETUP_planck_20261012.md) ────────────────────────────────────
$READ_TASKS = "scripts/assets/planck3_read_train_tasks.json"
$BRAVE_RESERVE = "400"     # training-data collection leaves this many Brave searches in the month for the product

# A round 6 run is valid if Brave served >= 60% of its searches, <= 20% came back empty and >= 60% were on topic
# (web.search_health "valid"). A miss is set aside ONCE as <dir>_degraded and redone; a second miss is kept.
function Test-R6Run([string]$dir) {
    if (-not (Test-Path "$dir/summary.json")) { return $false }
    $j = Get-Content "$dir/summary.json" -Raw | ConvertFrom-Json
    $h = $j.search_health
    $ok = ($null -ne $h) -and [bool]$h.valid -and ($j.search -eq "brave") -and ($null -ne $h.primary_share) -and ($h.primary_share -ge 0.6)
    if ($ok) { return $true }
    if (Test-Path "${dir}_degraded") { Warn "$dir is below the search bar again; keeping it (reported as is)"; return $true }
    Move-Item $dir "${dir}_degraded"; Log "kept $dir as ${dir}_degraded (search below the bar); redoing it"
    return $false
}

function Invoke-R6([string]$name, [string[]]$a, [string]$what, [int]$sample = 0) {
    $out = "$RES/$name$Tag"
    if (Test-R6Run $out) { Log "SKIP $name (valid)"; return }
    Log $what
    $smp = if ($Quick) { @("--sample", "12") } elseif ($sample -gt 0) { @("--sample", "$sample") } else { @() }
    P3 (@("r5") + $a + @("--tasks", "$FRESH,$NEWS,$INFO", "--search", "brave", "--out", $out, "--gemma-path", $GEMMA,
          "--g2-head", $V2HEAD, "--planck-checkpoint", $PLANCK_CKPT, "--planck-tokenizer", $PLANCK_TOK) + $smp)
}

function Do-Round6 {
    foreach ($f in @($NEWS, $INFO, $READ_TASKS, $V2HEAD, $PLANCK_CKPT)) { if (-not (Test-Path $f)) { throw "missing $f (git pull)" } }
    if (-not $env:BRAVE_API_KEY) { throw "no BRAVE_API_KEY: round 6 runs on the Brave Search API (.\scripts\planck3.ps1 setkey)" }
    if ((Invoke-Quiet $PY @("scripts/planck3.py", "search-check", "--backend", "brave")).Code -ne 0) {
        throw "the Brave API does not answer (key, quota or network): check with .\scripts\planck3.ps1 setkey"
    }
    $haveGemma = Test-Path $GEMMA
    $haveHertz = (Test-Path $HERTZ_CKPT) -and (Test-Path $HERTZ_TOK)
    if (-not $haveGemma) { throw "round 6 needs Gemma at $GEMMA (it writes the news training questions and is the reference)" }
    $ttag = if ($Tag) { @("--tag", $Tag) } else { @() }
    $newsTasks = "data/planck3/reader/news_tasks$Tag.json"
    $points = "data/planck3/reader/points$Tag.jsonl"
    # 1. reader-training questions: Gemma turns 2025 Current Events items into questions (resumable)
    $lim = if ($Quick) { @("--limit", "40") } else { @() }
    Log "reader data 1/2: Gemma writes news questions from the 2025 Current Events items (resumable; ~1 h first time)"
    P3 (@("readtasks", "newsq", "--out", $newsTasks, "--gemma-path", $GEMMA) + $lim)
    # 2. labelled pools from known answers: Brave while the month allows, then ddgs (relevance-checked)
    $lim = if ($Quick) { @("--limit", "60") } else { @() }
    Log "reader data 2/2: search + label pools for the Wikidata and news questions (resumable; ~1.5-2 h first time)"
    P3 (@("xreader", "collect", "--tasks", "$READ_TASKS,$newsTasks", "--points", $points, "--search", "brave",
          "--secondary", "ddgs", "--brave-reserve", $BRAVE_RESERVE) + $lim)
    # 3. the readers, trained ONCE (round 4's labelled pools for the 1,455 round 3 questions are reused)
    $q = if ($Quick) { @("--max-points", "300", "--epochs", "1") } else { @() }
    if (Test-Newer $points "checkpoints/xreader_planck$Tag/reader.pt") {
        Log "train the Planck 1.3 reader (100M, cross-encoder), once"
        P3 (@("xreader", "train", "--base", "planck", "--points", $points, "--base-checkpoint", $PLANCK_CKPT, "--base-tokenizer", $PLANCK_TOK) + $ttag + $q)
    }
    if ($haveHertz -and (Test-Newer $points "checkpoints/xreader_hertz$Tag/reader.pt")) {
        Log "train the Hertz 1.2 reader (640M, cross-encoder), once"
        P3 (@("xreader", "train", "--base", "hertz", "--points", $points, "--base-checkpoint", $HERTZ_CKPT, "--base-tokenizer", $HERTZ_TOK) + $ttag + $q)
    } elseif (-not $haveHertz) { Warn "no Hertz checkpoint: the Hertz reader arm is skipped" }
    # 4. reading only (template writer), all on Brave: floor, round 5's reader, the new readers, Gemma
    Invoke-R6 "r6_heuristic_template" @("--reader", "heuristic", "--writer", "template") "round 6: no-model floor"
    Invoke-R6 "r6_planck_template" @("--reader", "planck", "--writer", "template") "round 6: round 5's Planck reader (frozen features), template writer"
    Invoke-R6 "r6_x-planck_template" @("--reader", "x-planck", "--reader-checkpoint", "checkpoints/xreader_planck$Tag/reader.pt", "--writer", "template") "round 6: the new Planck reader, template writer"
    if (Test-Path "checkpoints/xreader_hertz$Tag/reader.pt") {
        Invoke-R6 "r6_x-hertz_template" @("--reader", "x-hertz", "--reader-checkpoint", "checkpoints/xreader_hertz$Tag/reader.pt", "--writer", "template") "round 6: the new Hertz reader, template writer"
    }
    Invoke-R6 "r6_gemma_template" @("--reader", "gemma", "--writer", "template") "round 6: Gemma reader, template writer (reading reference)"
    $reader = @((Invoke-Quiet $PY (@("scripts/planck3.py", "r6", "reader") + $ttag)).Out | Where-Object { $_ })[-1]
    if (-not $reader) { $reader = "x-planck" }
    $rbase = $reader.Substring(2)
    $rck = @("--reader", $reader, "--reader-checkpoint", "checkpoints/xreader_$rbase$Tag/reader.pt")
    Log "reader that ships (pre-registered rule): $reader"
    # 5. writers: round 5's data re-judged (checker fix), new Gemma answers with the forced lead, retrain once
    $w5 = "data/planck3/writer/train.jsonl"
    $w6 = "data/planck3/writer/train6$Tag.jsonl"
    if (Test-Path $w5) { Log "writer data: re-judge round 5's examples with the fixed checker"; P3 @("writer", "rescore", "--out", $w5) }
    $lim = if ($Quick) { @("--limit", "40") } else { @("--limit", "1500") }
    Log "writer data: Gemma writes answers (forced lead) for the reader-training questions (resumable; ~1-1.5 h)"
    P3 (@("writer", "collect", "--tasks", "$READ_TASKS,$newsTasks", "--out", $w6, "--search", "brave", "--secondary", "ddgs",
          "--brave-reserve", $BRAVE_RESERVE, "--gemma-path", $GEMMA) + $rck + $lim)
    $wdata = if (Test-Path $w5) { "$w5,$w6" } else { $w6 }
    $writers = @("planck"); if ($haveHertz) { $writers += "hertz" }
    foreach ($w in $writers) {
        if (Test-Newer $w6 "checkpoints/writer_${w}6$Tag/best.pt") {
            Log "train the $w writer once more (round 6 data, forced lead)"
            $wa = @("writer", "train", "--base", $w, "--out", $wdata, "--require-lead", "--tag", "6$Tag")
            if ($w -eq "hertz") { $wa += @("--base-checkpoint", $HERTZ_CKPT, "--base-tokenizer", $HERTZ_TOK, "--batch-size", "4") }
            else { $wa += @("--base-checkpoint", $PLANCK_CKPT, "--base-tokenizer", $PLANCK_TOK) }
            P3 $wa
        }
        Log "KV cache check: $w writer (cached vs uncached greedy answers must be identical)"
        P3 @("writer", "kvcheck", "--checkpoint", "checkpoints/writer_${w}6$Tag/best.pt", "--out", $w6, "--limit", "20")
    }
    # 6. the product candidates and the full Gemma reference
    foreach ($w in $writers) {
        Invoke-R6 "r6_${reader}_$w" ($rck + @("--writer", $w, "--writer-checkpoint", "checkpoints/writer_${w}6$Tag/best.pt", "--tries", "3")) "round 6: $reader reader + $w writer (product candidate)"
    }
    Invoke-R6 "r6_gemma_gemma" @("--reader", "gemma", "--writer", "gemma") "round 6: Gemma reader + Gemma writer (reference)"
    # 7. what ships (pre-registered), then the checks around it
    P3 (@("r6", "product") + $ttag)
    $prod = Get-Content "$RES/r6_product$Tag.json" -Raw | ConvertFrom-Json
    $pa = $rck + @("--writer", $prod.writer)
    if ($prod.writer -ne "template") { $pa += @("--writer-checkpoint", "checkpoints/writer_$($prod.writer)6$Tag/best.pt", "--tries", "3") }
    Invoke-R6 "$($prod.run)_contrarian" ($pa + @("--personal", "contrarian")) "round 6: worst-case personal profile on the product (rule E)"
    if ($prod.writer -ne "template" -and -not (Test-Path "$RES/$($prod.run)$Tag/judge.json")) {
        Log "writer judge: Gemma compares the small writer's answers with the template, blind"
        P3 @("writer", "judge", "--run", "$RES/$($prod.run)$Tag", "--gemma-path", $GEMMA)
    }
    # 8. CPU only, 4 threads: the phone proxy, now with the KV cache (searches and pages are cached by now)
    $old = $env:CUDA_VISIBLE_DEVICES; $env:CUDA_VISIBLE_DEVICES = "-1"; $env:OMP_NUM_THREADS = "4"
    try {
        Invoke-R6 "$($prod.run)_cpu" $pa "round 6: the product on CPU only, 4 threads, 20 questions (phone proxy, KV cache)" 20
        foreach ($w in $writers) { P3 @("writer", "kvcheck", "--checkpoint", "checkpoints/writer_${w}6$Tag/best.pt", "--out", $w6, "--limit", "10") }
    }
    finally { if ($old) { $env:CUDA_VISIBLE_DEVICES = $old } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }; Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue }
}

# Round 6b (SETUP_planck_20261012.md section 10): the same readers retrained on the pools AS SERVED
# (round 6 swapped a deep right candidate into the last slot). Same points, same settings, no new
# searches (the 243 questions are cached); the template writes. Comparators are round 6's runs.
function Do-Round6b {
    $points = "data/planck3/reader/points$Tag.jsonl"
    foreach ($f in @($NEWS, $INFO, $V2HEAD, $PLANCK_CKPT, $points, "$RES/r6_planck_template$Tag/summary.json", "$RES/r6_gemma_template$Tag/summary.json")) {
        if (-not (Test-Path $f)) { throw "missing $f (round 6b reuses round 6: run .\scripts\planck3.ps1 round6 first)" }
    }
    $haveHertz = (Test-Path $HERTZ_CKPT) -and (Test-Path $HERTZ_TOK)
    $ttag = if ($Tag) { @("--tag", $Tag) } else { @() }
    $q = if ($Quick) { @("--max-points", "300", "--epochs", "1") } else { @() }
    $bases = @("planck"); if ($haveHertz) { $bases += "hertz" }
    foreach ($b in $bases) {
        $ck = "checkpoints/xreader_${b}_v2$Tag/reader.pt"
        if (-not (Test-Path $ck)) {
            Log "round 6b: train the $b reader once more, on the top 16 candidates as served"
            $bc = if ($b -eq "hertz") { @("--base-checkpoint", $HERTZ_CKPT, "--base-tokenizer", $HERTZ_TOK) } else { @("--base-checkpoint", $PLANCK_CKPT, "--base-tokenizer", $PLANCK_TOK) }
            P3 (@("xreader", "train", "--base", $b, "--points", $points, "--tag", "_v2$Tag") + $bc + $q)
        }
        Invoke-R6 "r6b_x-${b}_template" @("--reader", "x-$b", "--reader-checkpoint", $ck, "--writer", "template") "round 6b: the $b reader (as served), template writer"
    }
    P3 (@("r6", "productb") + $ttag)
    $prod = Get-Content "$RES/r6b_product$Tag.json" -Raw | ConvertFrom-Json
    $pa = @("--reader", $prod.reader, "--reader-checkpoint", $prod.reader_checkpoint, "--writer", "template")
    Invoke-R6 "$($prod.run)_contrarian" ($pa + @("--personal", "contrarian")) "round 6b: worst-case personal profile on the product (rule E)"
    $old = $env:CUDA_VISIBLE_DEVICES; $env:CUDA_VISIBLE_DEVICES = "-1"; $env:OMP_NUM_THREADS = "4"
    try { Invoke-R6 "$($prod.run)_cpu" $pa "round 6b: the product on CPU only, 4 threads, 20 questions (phone proxy)" 20 }
    finally { if ($old) { $env:CUDA_VISIBLE_DEVICES = $old } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }; Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue }
}

# Round 7 (SETUP_planck_20261014.md): reading taught as a SKILL first (public reading-comprehension sets
# turned into the reader's own pools), then round 6b's training on our pools. SGS only (Planck, Hertz).
# No new searches: the 243 questions are cached on Brave. The product reader is fixed: x-planck.
function Do-Round7 {
    $points = "data/planck3/reader/points$Tag.jsonl"
    foreach ($f in @($NEWS, $INFO, $V2HEAD, $PLANCK_CKPT, $points, "$RES/r6_planck_template$Tag/summary.json", "$RES/r6_gemma_template$Tag/summary.json")) {
        if (-not (Test-Path $f)) { throw "missing $f (round 7 reuses round 6: run .\scripts\planck3.ps1 round6 first)" }
    }
    $haveHertz = (Test-Path $HERTZ_CKPT) -and (Test-Path $HERTZ_TOK)
    $ttag = if ($Tag) { @("--tag", $Tag) } else { @() }
    # 1. the reading sets -> pools (downloads ~390 MB of parquet once; ~15-30 min)
    $rc = "data/planck3/reader/rc_points$Tag.jsonl"
    $sz = if ($Quick) { @("--per-set", "300", "--neg", "150") } else { @() }
    Log "round 7 data: public reading-comprehension sets -> reader pools (SQuAD, NewsQA, TriviaQA, SearchQA, NQ, SQuAD 2.0 unanswerable)"
    P3 (@("xreader", "rcdata", "--points", $rc) + $sz)
    $bases = @("planck"); if ($haveHertz) { $bases += "hertz" }
    foreach ($b in $bases) {
        $bc = if ($b -eq "hertz") { @("--base-checkpoint", $HERTZ_CKPT, "--base-tokenizer", $HERTZ_TOK) } else { @("--base-checkpoint", $PLANCK_CKPT, "--base-tokenizer", $PLANCK_TOK) }
        # 2. stage A: reading as a skill (1 epoch; Planck ~30 min, Hertz ~3-4 h)
        $ckA = "checkpoints/xreader_${b}_rc$Tag/reader.pt"
        if (-not (Test-Path $ckA)) {
            Log "round 7 stage A: the $b reader learns to read (public reading sets, 1 epoch)"
            $qa = if ($Quick) { @("--max-points", "1500") } else { @() }
            P3 (@("xreader", "train", "--base", $b, "--points", $rc, "--stable-points", "none", "--epochs", "1", "--max-eval", "6000", "--tag", "_rc$Tag") + $bc + $qa)
        }
        # 3. stage B: stage A, then round 6b's training on our pools (same points and settings)
        $ckB = "checkpoints/xreader_${b}_r7$Tag/reader.pt"
        if (-not (Test-Path $ckB)) {
            Log "round 7 stage B: the $b reader, from stage A, on our pools (as round 6b)"
            $qb = if ($Quick) { @("--max-points", "300", "--epochs", "1") } else { @() }
            P3 (@("xreader", "train", "--base", $b, "--points", $points, "--init", $ckA, "--tag", "_r7$Tag") + $bc + $qb)
        }
        Invoke-R6 "r7_x-${b}-rc_template" @("--reader", "x-$b", "--reader-checkpoint", $ckA, "--writer", "template") "round 7: the $b reader after stage A only (reading sets), template writer"
        Invoke-R6 "r7_x-${b}_template" @("--reader", "x-$b", "--reader-checkpoint", $ckB, "--writer", "template") "round 7: the $b reader (stage A + B), template writer"
    }
    # 4. the product (fixed: x-planck r7 + template) under the worst-case profile and on CPU, then the rules
    $pa = @("--reader", "x-planck", "--reader-checkpoint", "checkpoints/xreader_planck_r7$Tag/reader.pt", "--writer", "template")
    Invoke-R6 "r7_x-planck_template_contrarian" ($pa + @("--personal", "contrarian")) "round 7: worst-case personal profile on the product (rule E)"
    $old = $env:CUDA_VISIBLE_DEVICES; $env:CUDA_VISIBLE_DEVICES = "-1"; $env:OMP_NUM_THREADS = "4"
    try { Invoke-R6 "r7_x-planck_template_cpu" $pa "round 7: the product on CPU only, 4 threads, 20 questions (phone proxy)" 20 }
    finally { if ($old) { $env:CUDA_VISIBLE_DEVICES = $old } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }; Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue }
    P3 (@("r7", "decide") + $ttag)
}

# CPU only, 4 threads: the phone proxy (rule L, rule M). Restores the GPU afterwards.
function Invoke-OnCpu([scriptblock]$Body) {
    $old = $env:CUDA_VISIBLE_DEVICES; $env:CUDA_VISIBLE_DEVICES = "-1"; $env:OMP_NUM_THREADS = "4"
    try { & $Body }
    finally { if ($old) { $env:CUDA_VISIBLE_DEVICES = $old } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }; Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue }
}

# Round 8 (SETUP_planck_20261015.md): (1) round 7's Hertz reader on a phone proxy; (2) Planck 1.5 =
# Planck 1.3's architecture pretrained on Hertz 1.2's corpus; (3) round 7's reading stage on it; (4) the
# benchmark and the pre-registered rules. No new searches: the 243 questions are cached on Brave.
function Do-Round8 {
    $rc = "data/planck3/reader/rc_points$Tag.jsonl"
    foreach ($f in @($NEWS, $INFO, $V2HEAD, $PLANCK_CKPT, $rc, "$RES/r6_planck_template$Tag/summary.json",
                     "$RES/r6_gemma_template$Tag/summary.json", "$RES/r7_x-planck-rc_template$Tag/summary.json")) {
        if (-not (Test-Path $f)) { throw "missing $f (round 8 reuses rounds 6 and 7: run .\scripts\planck3.ps1 round7$(if ($Quick) { ' -Quick' }) first)" }
    }
    $ttag = if ($Tag) { @("--tag", $Tag) } else { @() }
    # 1. Hertz on the phone proxy (~1 h): CPU, 4 threads; fp32 for the latency reference, int8 for the rule
    $ckH = "checkpoints/xreader_hertz_rc$Tag/reader.pt"
    if (Test-Path $ckH) {
        $ha = @("--reader", "x-hertz", "--reader-checkpoint", $ckH, "--writer", "template")
        Invoke-OnCpu {
            Invoke-R6 "r8_x-hertz-rc_cpu-fp32-top16" $ha "round 8 mobile: Hertz's reader on CPU, fp32, top 16 candidates, 20 questions (latency reference)" 20
            Invoke-R6 "r8_x-hertz-rc_cpu-int8-top16" ($ha + @("--reader-quant", "int8")) "round 8 mobile: Hertz's reader on CPU, int8, top 16 candidates, all questions"
            Invoke-R6 "r8_x-hertz-rc_cpu-int8-top8" ($ha + @("--reader-quant", "int8", "--reader-cands", "8")) "round 8 mobile: Hertz's reader on CPU, int8, top 8 candidates, all questions"
        }
        P3 (@("r8", "decide") + $ttag)
        P3 @("report")
        Do-Push   # the mobile result lands before the multi-day pretraining starts
    } else { Warn "no Hertz reading reader at $ckH (round 7): the mobile check is skipped" }

    # 2. Planck 1.5: Planck 1.3's architecture (100M) on Hertz 1.2's corpus, its tokenizer, 1 epoch
    $data = "data/hertz12_data"
    foreach ($f in @("train.bin", "val.bin", "tokenizer.model")) {
        if (-not (Test-Path "$data/$f")) { throw "Hertz 1.2's corpus is missing ($data/$f). Round 8 reuses it as is; preparing a new 10B corpus would change the data (see SETUP_hertz_202606_1.md)" }
    }
    $gb = (Get-Item "$data/train.bin").Length / 2e9
    if ($gb -lt 9) { throw ("$data/train.bin holds {0:N2}B tokens, not Hertz's ~9.7B: a stale or partial corpus" -f $gb) }
    $save = "checkpoints/planck15$Tag"
    if (-not (Test-Path "$save/final_metrics.json")) {
        Log ("round 8: pretraining Planck 1.5 on {0:N2}B tokens of Hertz's corpus (~25 h at the measured ~110k tok/s; keep the box awake; re-run resumes)" -f $gb)
        $pa = @("-u", "scripts/train_hertz.py", "--dataset", "fineweb-wiki-mix", "--data-dir", $data,
                "--d-s", "128", "--d-f", "1000", "--n-passes", "3", "--n-heads", "4", "--context-len", "512",
                "--epochs", "1", "--batch-size", "16", "--grad-accum", "4", "--mixed-precision", "bf16",
                "--lr", "3e-4", "--warmup-steps", "2000", "--save-interval", "2000", "--keep-last", "3",
                "--bf16-milestone-interval", "0", "--eval-interval", "2000", "--save-dir", $save)
        if ($Quick) { $pa += @("--max-opt-steps", "300", "--warmup-steps", "30", "--eval-interval", "100", "--save-interval", "100") }
        $last = Get-ChildItem -Path $save -Filter "step_*.pt" -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.BaseName -replace '\D', '') } -Descending | Select-Object -First 1
        if ($last) { Log "resuming from $($last.Name)"; $pa += @("--resume", $last.FullName) }
        Invoke-PyLogged $pa
    }
    $baseCk = if (Test-Path "$save/best.pt") { "$save/best.pt" } else { "$save/final.pt" }

    # 3. round 7's stage A (the reading sets, 1 epoch) on Planck 1.5; no stage B
    $ckP = "checkpoints/xreader_planck_p15rc$Tag/reader.pt"
    if (-not (Test-Path $ckP)) {
        Log "round 8: Planck 1.5's reader learns to read (round 7's reading sets, 1 epoch)"
        $qa = if ($Quick) { @("--max-points", "1500") } else { @() }
        P3 (@("xreader", "train", "--base", "planck", "--points", $rc, "--stable-points", "none", "--epochs", "1",
              "--max-eval", "6000", "--tag", "_p15rc$Tag", "--base-checkpoint", $baseCk, "--base-tokenizer", "$data/tokenizer.model") + $qa)
    }

    # 4. the product (fixed: x-planck15 rc + template), its worst-case profile and CPU runs, then the rules
    $pa = @("--reader", "x-planck", "--reader-checkpoint", $ckP, "--writer", "template")
    Invoke-R6 "r8_x-planck15-rc_template" $pa "round 8: Planck 1.5's reader, template writer"
    Invoke-R6 "r8_x-planck15-rc_template_contrarian" ($pa + @("--personal", "contrarian")) "round 8: worst-case personal profile on the product (rule E)"
    Invoke-OnCpu { Invoke-R6 "r8_x-planck15-rc_template_cpu" $pa "round 8: the product on CPU only, 4 threads, 20 questions (phone proxy)" 20 }
    P3 (@("r8", "decide") + $ttag)
}

# ── secrets ──────────────────────────────────────────────────────────────
function Do-SetKey {
    $name = if ($Question) { $Question } else { "BRAVE_API_KEY" }
    $sec = Read-Host "Paste $name (input hidden)" -AsSecureString
    $val = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
    if (-not $val) { throw "nothing entered; .env unchanged" }
    $path = Join-Path $Root ".env"
    $lines = @()
    if (Test-Path $path) { $lines = @(Get-Content $path | Where-Object { $_ -notmatch "^\s*$name\s*=" }) }
    $lines += "$name=$($val.Trim())"
    Set-Content -Path $path -Value $lines -Encoding UTF8
    [Environment]::SetEnvironmentVariable($name, $val.Trim(), "Process")
    Log "$name saved to .env (gitignored, never pushed)"
    if ($name -eq "BRAVE_API_KEY") {
        try { P3 @("search-check", "--backend", "brave") } catch { Warn "the Brave API did not answer with this key (typo, or over quota?)" }
    }
}

# ── results back to git (readable from any machine) ──────────────────────
function Do-Push {
    if ($NoPush) { Log "-NoPush: results left uncommitted"; return }
    Log "committing results/planck3 (summaries, REPORT.md, logs, compressed trajectories)"
    foreach ($pat in @("summary.json", "cards.md", "results.jsonl", "train_log.json", "trajectories.jsonl.gz",
                       "pairs_*.jsonl", "REPORT.md", "doctor.json", "digest.md", "*.log", "judge.json", "judge_pairs.jsonl",
                       "r*_product*.json")) {
        Get-ChildItem -Path $RES -Recurse -Filter $pat -ErrorAction SilentlyContinue |
            ForEach-Object { Invoke-Quiet git @("add", "-f", "--", $_.FullName) | Out-Null }
    }
    # the system trust layer is shared with every user: it ships in git
    if (Test-Path "config/planck3/source_trust.json") { Invoke-Quiet git @("add", "--", "config/planck3/source_trust.json") | Out-Null }
    Get-ChildItem -Path "data/planck3/g2" -Filter "points_*.health.json" -ErrorAction SilentlyContinue |
        ForEach-Object { Invoke-Quiet git @("add", "-f", "--", $_.FullName) | Out-Null }
    # round 6: Gemma's news questions (and the rejected ones, with reasons) are training data: keep them auditable
    Get-ChildItem -Path "data/planck3/reader" -Filter "news_tasks*.json" -ErrorAction SilentlyContinue |
        ForEach-Object { Invoke-Quiet git @("add", "-f", "--", $_.FullName) | Out-Null }
    Get-ChildItem -Path "data/planck3/reader" -Filter "*points*.health.json" -ErrorAction SilentlyContinue |
        ForEach-Object { Invoke-Quiet git @("add", "-f", "--", $_.FullName) | Out-Null }
    # round 8: the rebuilt base's metrics (weights stay local)
    Get-ChildItem -Path "checkpoints" -Filter "final_metrics.json" -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { Invoke-Quiet git @("add", "-f", "--", $_.FullName) | Out-Null }
    Invoke-Quiet git @("reset", "-q", "--", ".env") | Out-Null   # belt and braces: the repo is public
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
    "setkey"     { Do-SetKey }
    "ask"        {
        if (-not $Question) { throw 'usage: .\scripts\planck3.ps1 ask "your question"' }
        P3 (@("ask", $Question, "--type", $Type, "--policy", (Get-TeacherPolicy), "--gemma-path", $GEMMA, "-v") + $Rest)
    }
    { $_ -in @("chat", "serve") } {
        Do-Search
        if ($Command -eq "serve") { Log "open http://127.0.0.1:8010 in a browser (Ctrl+C to stop)" }
        P3 (@($Command.ToLower(), "--policy", (Get-TeacherPolicy), "--gemma-path", $GEMMA) + $Rest)
    }
    "g0"         { Start-RunLog "g0"; Do-Searxng; Do-G0; P3 @("report") }
    "g1"         { Start-RunLog "g1"; Do-G1; P3 @("report") }
    "report"     { P3 @("report") }
    "digest"     { $a = @("digest"); if ($Explore) { Do-Search; $a += "--explore" }; P3 ($a + $Rest) }
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
    "round5"     {
        Start-RunLog "round5"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Doctor $true
        Do-Ddgs $true
        Do-Round5
        P3 @("report")
        Log ("round5$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    "round6"     {
        Start-RunLog "round6"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Doctor $true
        Do-Round6
        P3 @("report")
        Log ("round6$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    "round7"     {
        Start-RunLog "round7"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Round7
        P3 @("report")
        Log ("round7$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    "round8"     {
        Start-RunLog "round8"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Round8
        P3 @("report")
        Log ("round8$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    "round6b"    {
        Start-RunLog "round6b"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Round6b
        P3 @("report")
        Log ("round6b$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
        Do-Push
    }
    "round4"     {
        Start-RunLog "round4"
        $t0 = Get-Date
        Do-Setup
        Log "offline smoke tests"; Invoke-Checked $PY @("-m", "pytest", "tests/test_planck3.py", "-q")
        Do-Doctor $true
        Do-Ddgs $true
        Do-Round4
        P3 @("report")
        Log ("round4$Tag finished in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
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
