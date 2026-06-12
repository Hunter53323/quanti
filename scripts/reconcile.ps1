# scripts/reconcile.ps1 -- Manual position reconciliation (local vs. broker)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
. $venvActivate

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running position reconciliation..."
python -c @"
from quanti.execution.order_manager import OrderManager
from quanti.state.journal import Journal

j = Journal()
om = OrderManager(j)
discrepancies = om.reconcile_positions()
if not discrepancies:
    print('OK: Local positions match broker.')
else:
    print(f'WARN: {len(discrepancies)} discrepancies found:')
    for d in discrepancies:
        print(f'  {d}')
"@
exit $LASTEXITCODE
