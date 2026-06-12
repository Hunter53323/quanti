# scripts/run_paper.ps1 -- Start paper trading session

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run setup first."
    exit 1
}
. $venvActivate

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting paper trading session..."
Write-Host "Press Ctrl+C to stop."
python -m quanti.main_paper
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Paper trading stopped. Exit code: $exitCode"
exit $exitCode
