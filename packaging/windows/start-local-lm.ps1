$ErrorActionPreference = "Stop"
$InstallRoot = if ($env:LM_ATELIER_INSTALL_ROOT) { $env:LM_ATELIER_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "LMAtelier" }
$Version = (Get-Content (Join-Path $InstallRoot "current-version.txt") -Raw).Trim()
$AppRoot = Join-Path $InstallRoot "versions\$Version"
if (-not $env:LOCAL_LM_DATA_DIR) { $env:LOCAL_LM_DATA_DIR = Join-Path $InstallRoot "data" }
if (-not $env:LOCAL_LM_HOST) { $env:LOCAL_LM_HOST = "127.0.0.1" }
$Port = if ($env:LOCAL_LM_PORT) { $env:LOCAL_LM_PORT } else { "12340" }
$ApplicationUrl = "http://$($env:LOCAL_LM_HOST):$Port"
Set-Location $AppRoot
$Python = Join-Path $AppRoot ".venv\Scripts\python.exe"
$BrowserJob = Start-Job -ScriptBlock {
  param($Url)
  for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    try {
      Invoke-WebRequest "$Url/api/health" -UseBasicParsing -TimeoutSec 1 | Out-Null
      Start-Process $Url
      return
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
} -ArgumentList $ApplicationUrl
try {
  & $Python -c "from local_lm.main import run; run()"
} finally {
  Stop-Job $BrowserJob -ErrorAction SilentlyContinue
  Remove-Job $BrowserJob -Force -ErrorAction SilentlyContinue
}
