# scripts/ingest_daily.ps1 -- Fetch and validate daily ETF data
# Schedule: Windows Task Scheduler, 16:30 on trading days

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting daily data ingestion..."

# Activate venv
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Run setup first."
    exit 1
}
. $venvActivate

# Run python ingestion
python -c @"
from quanti.data.ingestion import run_daily_ingest
ok, errors = run_daily_ingest()
for e in errors:
    print(f'WARN: {e}')
if not ok:
    print('ERROR: Ingestion completed with errors, check logs')
    exit(1)
print('OK: Daily ingestion complete')
"@
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Ingestion finished. Exit code: $exitCode"
exit $exitCode
