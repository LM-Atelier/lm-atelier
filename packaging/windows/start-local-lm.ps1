$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
if (-not (Test-Path ".venv\Scripts\lm-atelier.exe")) {
  throw "LM Atelier is not installed. Create .venv and install services/api first."
}
if (-not (Test-Path "apps\web\dist")) { npm run build }
& ".venv\Scripts\lm-atelier.exe"
