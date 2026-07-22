$ErrorActionPreference = "Stop"
$InstallRoot = if ($env:LM_ATELIER_INSTALL_ROOT) { $env:LM_ATELIER_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "LMAtelier" }
$Version = (Get-Content (Join-Path $InstallRoot "current-version.txt") -Raw).Trim()
$AppRoot = Join-Path $InstallRoot "versions\$Version"
if (-not $env:LOCAL_LM_DATA_DIR) { $env:LOCAL_LM_DATA_DIR = Join-Path $InstallRoot "data" }
if (-not $env:LOCAL_LM_HOST) { $env:LOCAL_LM_HOST = "127.0.0.1" }
Set-Location $AppRoot
& (Join-Path $AppRoot ".venv\Scripts\lm-atelier.exe")
