# activate.ps1 -- Activate the virtual environment for this project
# Usage: .\activate.ps1

$venvPath = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating: $venvPath"
    . $venvPath
    Write-Host "Environment ready. Python: $(python --version)"
    Write-Host "Project root: $PSScriptRoot"
} else {
    Write-Error "Virtual environment not found at $venvPath"
    Write-Host "Create it first: python -m venv venv"
    exit 1
}
