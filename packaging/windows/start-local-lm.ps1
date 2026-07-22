$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
if (-not (Test-Path ".venv\Scripts\local-lm.exe")) {
  throw "Local LM is not installed. Create .venv and install services/api first."
}
if (-not (Test-Path "apps\web\dist")) { npm run build }
& ".venv\Scripts\local-lm.exe"
