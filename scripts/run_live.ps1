# scripts/run_live.ps1 -- Start live trading (invoked via NSSM Windows Service)
# DO NOT run this directly in a terminal for live trading.
# For paper trading, use run_paper.ps1 instead.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run setup first."
    exit 1
}
. $venvActivate

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting live trading engine (NSSM managed)..."
python -m quanti.main_live
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Live trading stopped. Exit code: $exitCode"
exit $exitCode
