# scripts/healthcheck.ps1 -- Daily system health check
# Schedule: Windows Task Scheduler, 09:00 daily

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
. $venvActivate

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running health check..."

# Check data freshness
$dataDir = Join-Path $ProjectRoot "data\clean"
$latestFile = Get-ChildItem -Path $dataDir -Filter "*.parquet" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latestFile) {
    Write-Host "WARN: No data files found in $dataDir"
} else {
    $hoursAgo = (Get-Date) - $latestFile.LastWriteTime
    Write-Host "Data freshness: last file $($latestFile.Name) written $([math]::Round($hoursAgo.TotalHours, 1)) hours ago"
    if ($hoursAgo.TotalHours -gt 24) {
        Write-Host "CRITICAL: Data is more than 24 hours stale!"
    }
}

# Check process (if live)
$process = Get-Process -Name "python" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*quanti*" }
if ($process) {
    Write-Host "OK: Quanti process running (PID: $($process.Id), CPU: $([math]::Round($process.CPU,1)))"
} else {
    Write-Host "INFO: No quanti process running (expected if not in live/paper session)"
}

# Check disk space on C:
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "Disk free: ${freeGB}GB on C:"
if ($freeGB -lt 5) {
    Write-Host "WARN: Low disk space!"
}

# Check log directory
$logDir = Join-Path $ProjectRoot "logs"
if (Test-Path $logDir) {
    $logCount = (Get-ChildItem -Path $logDir -Filter "*.log" -ErrorAction SilentlyContinue).Count
    Write-Host "Log files: $logCount"
} else {
    Write-Host "INFO: No logs directory yet"
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Health check complete."
