# scripts/run_backtest.ps1 -- Run walk-forward backtest

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run setup first."
    exit 1
}
. $venvActivate

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting walk-forward backtest..."
python -m quanti.backtest.engine
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Backtest finished. Exit code: $exitCode"
exit $exitCode
