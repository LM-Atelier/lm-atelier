[CmdletBinding()]
param(
    [switch]$SkipWebBuild,
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$VersionFile = Join-Path $RepositoryRoot "services\api\local_lm\__init__.py"
$VersionMatch = Select-String -Path $VersionFile -Pattern '^__version__ = "([^"]+)"$'
$Version = $VersionMatch.Matches[0].Groups[1].Value
$OutputRoot = if ($OutputDirectory) {
    if ([IO.Path]::IsPathRooted($OutputDirectory)) {
        [IO.Path]::GetFullPath($OutputDirectory)
    } else {
        [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $OutputDirectory))
    }
} else {
    Join-Path $RepositoryRoot "release"
}
$IconRoot = Join-Path $RepositoryRoot "build\installer-assets"
$DistRoot = Join-Path $RepositoryRoot "build\pyinstaller-windows"
$WorkRoot = Join-Path $RepositoryRoot "build\pyinstaller-work-windows"

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Label,
        [Parameter(Mandatory)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    Write-Host "==> $Label"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing $Python. Create .venv and install services/api[dev,package] first."
}

$InnoCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
$InnoCompiler = $InnoCandidates | Select-Object -First 1
if (-not $InnoCompiler) {
    $InnoCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    $InnoCompiler = if ($InnoCommand) { $InnoCommand.Source } else { $null }
}
if (-not $InnoCompiler) {
    throw "Inno Setup 6 is required to build the Windows installer."
}

Push-Location $RepositoryRoot
try {
    if (-not $SkipWebBuild) {
        Invoke-Checked "Production web build" (Get-Command npm.cmd -ErrorAction Stop).Source @(
            "run", "build"
        )
    }
    Invoke-Checked "Installer icons" $Python @(
        "scripts/build-icons.py", "--output-dir", $IconRoot
    )
    Invoke-Checked "Frozen Windows application" $Python @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath", $DistRoot,
        "--workpath", $WorkRoot,
        "packaging/LMAtelier.spec"
    )
    Invoke-Checked "Frozen application smoke test" $Python @(
        "scripts/smoke-frozen.py",
        (Join-Path $DistRoot "LM Atelier\LM Atelier.exe"),
        "--version", $Version
    )
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    Invoke-Checked "Windows installer" $InnoCompiler @(
        "/DMyAppVersion=$Version",
        "/DMySourceDir=$(Join-Path $DistRoot 'LM Atelier')",
        "/DMyOutputDir=$OutputRoot",
        "packaging\windows\LMAtelier.iss"
    )

    $Installer = Join-Path $OutputRoot "LM-Atelier-Setup-$Version-windows-x86_64.exe"
    if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
        throw "The expected installer was not created: $Installer"
    }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Installer).Hash.ToLowerInvariant()
    Write-Host "Created $Installer"
    Write-Host "SHA-256: $Hash"
}
finally {
    Pop-Location
}
