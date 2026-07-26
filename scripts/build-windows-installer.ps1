[CmdletBinding()]
param(
    [switch]$SkipWebBuild,
    [string]$OutputDirectory,
    [string]$InnoCompilerPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$VersionFile = Join-Path $RepositoryRoot "services\api\local_lm\__init__.py"
$VersionMatch = Select-String -Path $VersionFile -Pattern '^__version__ = "([^"]+)"$'
$Version = $VersionMatch.Matches[0].Groups[1].Value
$CoreVersion = ($Version -split "-", 2)[0]
if ($CoreVersion -notmatch '^\d+\.\d+\.\d+$') {
    throw "Application version must start with three numeric components: $Version"
}
$FileVersionParts = @($CoreVersion -split "\.")
foreach ($Part in $FileVersionParts) {
    if ([uint64]$Part -gt 65535) {
        throw "Windows version components must not exceed 65535: $Version"
    }
}
$FileVersion = "$CoreVersion.0"
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
$ExpectedInnoVersion = "6.7.1"
$ExpectedInnoCompilerSha256 = (
    "eb6f4410c8db367a5f74127e8025ad2ccacc0afabbe783959d237df3050f97fb"
)

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

$InnoCompiler = if ($InnoCompilerPath) {
    if (-not (Test-Path -LiteralPath $InnoCompilerPath -PathType Leaf)) {
        throw "The requested Inno Setup compiler does not exist: $InnoCompilerPath"
    }
    (Resolve-Path -LiteralPath $InnoCompilerPath).Path
} else {
    $InnoCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
    $Candidate = $InnoCandidates | Select-Object -First 1
    if (-not $Candidate) {
        $InnoCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        $Candidate = if ($InnoCommand) { $InnoCommand.Source } else { $null }
    }
    $Candidate
}
if (-not $InnoCompiler) {
    throw "Inno Setup 6 is required to build the Windows installer."
}
$InnoVersionInfo = (Get-Item -LiteralPath $InnoCompiler).VersionInfo
$InnoVersionCandidates = @(
    $InnoVersionInfo.ProductVersion,
    $InnoVersionInfo.FileVersion
)
$InnoUninstaller = Join-Path (Split-Path -Parent $InnoCompiler) "unins000.exe"
if (Test-Path -LiteralPath $InnoUninstaller -PathType Leaf) {
    $UninstallerVersionInfo = (Get-Item -LiteralPath $InnoUninstaller).VersionInfo
    $InnoVersionCandidates += @(
        $UninstallerVersionInfo.ProductVersion,
        $UninstallerVersionInfo.FileVersion
    )
}
$InnoVersion = $InnoVersionCandidates |
    Where-Object { $_ -and $_ -notmatch '^0(?:\.0)*$' } |
    Select-Object -First 1
if (-not $InnoVersion) {
    throw "Could not determine the Inno Setup compiler version."
}
$InnoVersion = $InnoVersion.Trim()
if ($InnoVersion -ne $ExpectedInnoVersion) {
    throw "Expected Inno Setup $ExpectedInnoVersion; found $InnoVersion."
}
$InnoSignature = Get-AuthenticodeSignature -LiteralPath $InnoCompiler
if ($InnoSignature.Status -ne "Valid") {
    throw (
        "The Inno Setup compiler must have a valid Authenticode signature; " +
        "found $($InnoSignature.Status)."
    )
}
$InnoSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $InnoCompiler
).Hash.ToLowerInvariant()
if ($InnoSha256 -ne $ExpectedInnoCompilerSha256) {
    throw "The Inno Setup compiler does not match the reviewed SHA-256 digest."
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
    $MetadataArguments = @(
        "scripts/build-release-metadata.py",
        "--installer-tool", "Inno Setup Compiler",
        "--installer-tool-version", $InnoVersion,
        "--installer-tool-sha256", $InnoSha256
    )
    if ($env:RELEASE_TAG) {
        $MetadataArguments += "--require-release-tag"
    }
    Invoke-Checked "Release licenses, notices, and SBOM" $Python $MetadataArguments
    Invoke-Checked "Frozen Windows application" $Python @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath", $DistRoot,
        "--workpath", $WorkRoot,
        "packaging/LMAtelier.spec"
    )
    $FrozenApplicationRoot = Join-Path $DistRoot "LM Atelier"
    Invoke-Checked "Frozen payload inventory" $Python @(
        "scripts/inventory-frozen-payload.py",
        "--payload-root", $FrozenApplicationRoot,
        "--analysis-toc", (Join-Path $WorkRoot "LMAtelier\Analysis-00.toc")
    )
    Invoke-Checked "Frozen payload verification" $Python @(
        "scripts/inventory-frozen-payload.py",
        "--payload-root", $FrozenApplicationRoot,
        "--verify-only"
    )
    Invoke-Checked "Frozen application smoke test" $Python @(
        "scripts/smoke-frozen.py",
        (Join-Path $FrozenApplicationRoot "LM Atelier.exe"),
        "--version", $Version
    )
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    Invoke-Checked "Windows installer" $InnoCompiler @(
        "/DMyAppVersion=$Version",
        "/DMyFileVersion=$FileVersion",
        "/DMySourceDir=$FrozenApplicationRoot",
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
