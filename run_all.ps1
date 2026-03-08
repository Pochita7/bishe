<#
.SYNOPSIS
    GAIA Level 1 one-click test script
.DESCRIPTION
    Auto: conda activate -> set API Key -> solve all 53 tasks -> evaluation report
.USAGE
    .\run_all.ps1              # all 53 tasks
    .\run_all.ps1 -Num 5      # first 5 tasks
    .\run_all.ps1 -Single 0   # single task
    .\run_all.ps1 -EvalOnly   # evaluate only
    .\run_all.ps1 -Resume     # resume from last checkpoint
#>

param(
    [int]$Num = 0,
    [int]$Single = -1,
    [switch]$EvalOnly,
    [switch]$Resume,
    [string]$Output = "gaia_results.jsonl",
    [int]$MaxRound = 15
)

# ======================== Config ========================
$API_KEY   = "d9452cdf-f0ec-41f7-9029-115170830afc"
$CONDA_ENV = "autogen-env"
# ========================================================

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Banner
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GAIA Level 1 Auto Test" -ForegroundColor Cyan
Write-Host "  Model: DeepSeek-V3.2 (VolcEngine)" -ForegroundColor Cyan
$nowStr = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "  Time: $nowStr" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set API Key
$env:ARK_API_KEY = $API_KEY

# Activate conda env
Write-Host "[1/3] Activating conda env: $CONDA_ENV ..." -ForegroundColor Yellow
try {
    conda activate $CONDA_ENV 2>$null
} catch {
    Write-Host "  conda activate failed, trying direct run..." -ForegroundColor DarkYellow
}

# Validate env
$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
Write-Host "  Python: $pyVersion" -ForegroundColor Gray
$hasAutogen = python -c "import autogen_agentchat; print('OK')" 2>$null
if ($hasAutogen -ne "OK") {
    Write-Host "  [ERROR] autogen_agentchat not installed!" -ForegroundColor Red
    Write-Host "    conda activate $CONDA_ENV" -ForegroundColor Red
    Write-Host "    pip install autogen-agentchat autogen-ext" -ForegroundColor Red
    exit 1
}
Write-Host "  autogen_agentchat: OK" -ForegroundColor Gray

# Evaluate only mode
if ($EvalOnly) {
    Write-Host ""
    Write-Host "[Eval Mode] Analyzing results..." -ForegroundColor Yellow
    python run_gaia.py --evaluate --output $Output
    exit 0
}

# Resume mode
$startFrom = 0
if ($Resume -and (Test-Path $Output)) {
    $doneLines = Get-Content $Output | Where-Object { $_.Trim() }
    $startFrom = $doneLines.Count
    $doneCorrect = ($doneLines | ForEach-Object { (ConvertFrom-Json $_).is_correct } | Where-Object { $_ -eq $true }).Count
    $nextTask = $startFrom + 1
    Write-Host "[Resume] Done: $startFrom tasks (correct: $doneCorrect), continuing from task $nextTask" -ForegroundColor Green
} elseif (-not $Resume) {
    # Non-resume: backup old results
    if (Test-Path $Output) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backup = "${Output}.bak_${timestamp}"
        Copy-Item $Output $backup
        Write-Host "  Old results backed up to: $backup" -ForegroundColor Gray
        Remove-Item $Output
    }
}

# Build run command
Write-Host ""
Write-Host "[2/3] Starting solver ..." -ForegroundColor Yellow

$runArgs = @("run_gaia.py", "--output", $Output, "--max-round", $MaxRound)

if ($Resume) {
    $runArgs += "--resume"
}

if ($Single -ge 0) {
    $runArgs += @("--single", $Single)
    Write-Host "  Mode: Single task (index $Single)" -ForegroundColor Gray
} elseif ($Num -gt 0) {
    $runArgs += @("--num", $Num)
    Write-Host "  Mode: First $Num tasks" -ForegroundColor Gray
} else {
    $runArgs += "--all"
    Write-Host "  Mode: All Level 1 (53 tasks)" -ForegroundColor Gray
}

Write-Host "  Output: $Output" -ForegroundColor Gray
Write-Host "  Max rounds: $MaxRound" -ForegroundColor Gray
Write-Host ""

# Record start time
$startTime = Get-Date

# Run solver with colored output (temporarily allow errors from stderr)
$ErrorActionPreference = "Continue"
python $runArgs 2>&1 | ForEach-Object {
    $line = "$_"
    if ($line -match "Running accuracy") {
        Write-Host $line -ForegroundColor Cyan
    } elseif ($line -match "Correct:") {
        Write-Host $line -ForegroundColor Green
    } elseif ($line -match "ERROR|Traceback") {
        Write-Host $line -ForegroundColor Red
    } elseif ($line -match "Solving Task") {
        Write-Host $line -ForegroundColor Yellow
    } else {
        Write-Host $line
    }
}
$ErrorActionPreference = "Stop"

$elapsed = (Get-Date) - $startTime
$hours = [math]::Floor($elapsed.TotalHours)
$mins  = $elapsed.Minutes
$secs  = $elapsed.Seconds
$elapsedStr = "${hours}h ${mins}m ${secs}s"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[3/3] Done! Elapsed: $elapsedStr" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Auto evaluate
if (Test-Path $Output) {
    python run_gaia.py --evaluate --output $Output
} else {
    Write-Host "[WARN] Results file not found: $Output" -ForegroundColor Red
}

Write-Host ""
Write-Host "Done! Results: $Output" -ForegroundColor Green
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
