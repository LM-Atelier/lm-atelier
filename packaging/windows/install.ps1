$ErrorActionPreference = "Stop"
$SourceRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VersionLine = Select-String -Path (Join-Path $SourceRoot "services\api\local_lm\__init__.py") -Pattern '^__version__ = "([^"]+)"$'
$Version = $VersionLine.Matches[0].Groups[1].Value
$InstallRoot = if ($env:LM_ATELIER_INSTALL_ROOT) { $env:LM_ATELIER_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "LMAtelier" }
$VersionRoot = Join-Path $InstallRoot "versions\$Version"
$PartialRoot = "$VersionRoot.partial"

if (-not (Test-Path (Join-Path $SourceRoot "apps\web\dist"))) {
  throw "This archive has no prebuilt web application. Use an official release archive."
}

New-Item -ItemType Directory -Force -Path (Join-Path $InstallRoot "versions"), (Join-Path $InstallRoot "data") | Out-Null
if (Test-Path $PartialRoot) { Remove-Item -Recurse -Force $PartialRoot }
Copy-Item -Recurse -Force $SourceRoot $PartialRoot
py -3.12 -m venv (Join-Path $PartialRoot ".venv")
& (Join-Path $PartialRoot ".venv\Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $PartialRoot ".venv\Scripts\python.exe") -m pip install (Join-Path $PartialRoot "services\api")
if (Test-Path $VersionRoot) { Remove-Item -Recurse -Force $VersionRoot }
Move-Item $PartialRoot $VersionRoot

$CurrentFile = Join-Path $InstallRoot "current-version.txt"
if (Test-Path $CurrentFile) {
  Copy-Item $CurrentFile (Join-Path $InstallRoot "previous-version.txt") -Force
}
Set-Content -Path $CurrentFile -Value $Version -NoNewline
Copy-Item (Join-Path $VersionRoot "packaging\windows\start-local-lm.ps1") (Join-Path $InstallRoot "start-lm-atelier.ps1") -Force

$ProgramsRoot = [Environment]::GetFolderPath("Programs")
$ShortcutPath = Join-Path $ProgramsRoot "LM Atelier.lnk"
$PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
$ShortcutShell = New-Object -ComObject WScript.Shell
$Shortcut = $ShortcutShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PowerShell
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$InstallRoot\start-lm-atelier.ps1`""
$Shortcut.WorkingDirectory = $InstallRoot
$Shortcut.Description = "Open LM Atelier"
$Shortcut.IconLocation = "$VersionRoot\.venv\Scripts\lm-atelier.exe,0"
$Shortcut.Save()

Write-Host "LM Atelier $Version installed. Open LM Atelier from the Windows Start menu."
