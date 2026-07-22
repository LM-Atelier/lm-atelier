$ErrorActionPreference = "Stop"
$InstallRoot = if ($env:LM_ATELIER_INSTALL_ROOT) { $env:LM_ATELIER_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "LMAtelier" }
$CurrentFile = Join-Path $InstallRoot "current-version.txt"
$PreviousFile = Join-Path $InstallRoot "previous-version.txt"
if (-not (Test-Path $PreviousFile)) { throw "No previous LM Atelier version is available." }
$Current = (Get-Content $CurrentFile -Raw).Trim()
$Previous = (Get-Content $PreviousFile -Raw).Trim()
Set-Content -Path $CurrentFile -Value $Previous -NoNewline
Set-Content -Path $PreviousFile -Value $Current -NoNewline
Write-Host "Rolled back to $Previous. User data was unchanged."
