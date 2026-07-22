param([switch]$PurgeData)
$ErrorActionPreference = "Stop"
$InstallRoot = if ($env:LM_ATELIER_INSTALL_ROOT) { $env:LM_ATELIER_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "LMAtelier" }
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $InstallRoot "versions")
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $InstallRoot "current-version.txt"), (Join-Path $InstallRoot "previous-version.txt"), (Join-Path $InstallRoot "start-lm-atelier.ps1")
if ($PurgeData) {
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $InstallRoot "data")
  Write-Host "LM Atelier and its local data were removed."
} else {
  Write-Host "LM Atelier was removed. Local data remains at $InstallRoot\data"
}
