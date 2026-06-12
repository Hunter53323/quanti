# scripts/install_nssm_service.ps1 -- Install quanti as Windows Service
# Run as Administrator. Requires NSSM installed from https://nssm.cc/
$Action = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."

# Check NSSM
$NSSM = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $NSSM) {
    Write-Error "NSSM not found. Download from https://nssm.cc/ and add to PATH."
    exit 1
}

# Stop existing service if running
$Existing = Get-Service QuantiTrading -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "Stopping existing service..."
    nssm stop QuantiTrading 2>$null
    nssm remove QuantiTrading confirm 2>$null
}

# Install
$PythonPath = "$ProjectRoot\venv\Scripts\python.exe"
$MainScript = "$ProjectRoot\quanti\main_live.py"
Write-Host "Installing: nssm install QuantiTrading $PythonPath $MainScript"
nssm install QuantiTrading $PythonPath $MainScript

# Configure
nssm set QuantiTrading AppDirectory "$ProjectRoot"
nssm set QuantiTrading AppStdout "$ProjectRoot\logs\service_stdout.log"
nssm set QuantiTrading AppStderr "$ProjectRoot\logs\service_stderr.log"
nssm set QuantiTrading AppRotateFiles 1
nssm set QuantiTrading AppRotateBytes 10485760
nssm set QuantiTrading Start SERVICE_AUTO_START

Write-Host "QuantiTrading service installed."
Write-Host "Start:  nssm start QuantiTrading"
Write-Host "Status: nssm status QuantiTrading"
Write-Host "Logs:   Get-Content $ProjectRoot\logs\service_stdout.log -Wait"
