[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Installer,
    [Parameter(Mandatory)]
    [string]$Version,
    [string]$PreviousInstaller,
    [string]$PreviousVersion,
    [int]$Port = 12443,
    [switch]$AllowPurge
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Label,
        [Parameter(Mandatory)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [switch]$WaitProcess
    )

    Write-Host "==> $Label"
    if ($WaitProcess) {
        $Process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -Wait `
            -PassThru `
            -WindowStyle Hidden
        if ($Process.ExitCode -ne 0) {
            throw "$Label failed with exit code $($Process.ExitCode)."
        }
        return
    }
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if (-not $env:RUNNER_TEMP) {
    throw "RUNNER_TEMP is required for an isolated installer smoke test."
}
if ($AllowPurge -and $env:GITHUB_ACTIONS -ne "true") {
    throw "The purge test is restricted to an ephemeral GitHub-hosted runner."
}

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$InstallerPath = (Resolve-Path -LiteralPath $Installer).Path
$PreviousInstallerPath = if ($PreviousInstaller) {
    (Resolve-Path -LiteralPath $PreviousInstaller).Path
}
else {
    $null
}
if ([bool]$PreviousInstallerPath -ne [bool]$PreviousVersion) {
    throw "PreviousInstaller and PreviousVersion must be provided together."
}
$RunnerRoot = [IO.Path]::GetFullPath($env:RUNNER_TEMP)
$TestRoot = [IO.Path]::GetFullPath(
    (Join-Path $RunnerRoot "lm-atelier-installer-smoke-$PID")
)
if (-not $TestRoot.StartsWith(
        $RunnerRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) +
            [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )) {
    throw "The installer smoke-test root escaped RUNNER_TEMP."
}

$InstallRoot = Join-Path $TestRoot "application"
$InstallLog = Join-Path $TestRoot "install.log"
$DefaultData = Join-Path $env:LOCALAPPDATA "LMAtelier\data"
$DataMarker = Join-Path $DefaultData "installer-smoke-preserve.txt"
$StartMenuShortcut = Join-Path $env:APPDATA (
    "Microsoft\Windows\Start Menu\Programs\LM Atelier\LM Atelier.lnk"
)

if (Test-Path -LiteralPath $TestRoot) {
    throw "Refusing to reuse an installer smoke-test root: $TestRoot"
}
if (Test-Path -LiteralPath $DefaultData) {
    throw "Refusing to alter a pre-existing LM Atelier data directory: $DefaultData"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "The repository package environment is missing: $Python"
}

$VersionInfo = (Get-Item -LiteralPath $InstallerPath).VersionInfo
if ($VersionInfo.ProductVersion.Trim() -ne $Version) {
    throw "Installer product version does not match $Version."
}
$SignatureStatus = (
    Get-AuthenticodeSignature -LiteralPath $InstallerPath
).Status.ToString()
if ($SignatureStatus -notin @("Valid", "NotSigned")) {
    throw "Installer signature status is unexpected: $SignatureStatus"
}
if ($PreviousInstallerPath) {
    $PreviousVersionInfo = (
        Get-Item -LiteralPath $PreviousInstallerPath
    ).VersionInfo
    if ($PreviousVersionInfo.ProductVersion.Trim() -ne $PreviousVersion) {
        throw "Previous installer product version does not match $PreviousVersion."
    }
    $PreviousSignatureStatus = (
        Get-AuthenticodeSignature -LiteralPath $PreviousInstallerPath
    ).Status.ToString()
    if ($PreviousSignatureStatus -notin @("Valid", "NotSigned")) {
        throw (
            "Previous installer signature status is unexpected: " +
            $PreviousSignatureStatus
        )
    }
}

New-Item -ItemType Directory -Path $TestRoot | Out-Null
$InstallArguments = @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/CLOSEAPPLICATIONS=0",
    "/RESTARTAPPLICATIONS=0",
    "/DIR=$InstallRoot",
    "/LOG=$InstallLog"
)

try {
    $InitialInstaller = if ($PreviousInstallerPath) {
        $PreviousInstallerPath
    }
    else {
        $InstallerPath
    }
    $InitialVersion = if ($PreviousVersion) {
        $PreviousVersion
    }
    else {
        $Version
    }
    Invoke-Checked `
        "Silent installer execution" `
        $InitialInstaller `
        $InstallArguments `
        -WaitProcess
    $Application = Join-Path $InstallRoot "LM Atelier.exe"
    $Uninstaller = Join-Path $InstallRoot "unins000.exe"
    if (-not (Test-Path -LiteralPath $Application -PathType Leaf)) {
        throw "The installed application is missing: $Application"
    }
    if (-not (Test-Path -LiteralPath $Uninstaller -PathType Leaf)) {
        throw "The installed uninstaller is missing: $Uninstaller"
    }
    if (-not (Test-Path -LiteralPath $StartMenuShortcut -PathType Leaf)) {
        throw "The Start-menu shortcut was not created."
    }

    Invoke-Checked "Installed payload verification" $Python @(
        "scripts/inventory-frozen-payload.py",
        "--payload-root", $InstallRoot,
        "--verify-only",
        "--installer-extras", "windows"
    )
    Invoke-Checked "Installed application smoke test" $Python @(
        "scripts/smoke-frozen.py",
        $Application,
        "--version", $InitialVersion,
        "--port", "$Port"
    )

    New-Item -ItemType Directory -Force -Path $DefaultData | Out-Null
    [IO.File]::WriteAllText(
        $DataMarker,
        "LM Atelier installer preservation smoke test",
        [Text.UTF8Encoding]::new($false)
    )
    if ($PreviousInstallerPath) {
        Invoke-Checked `
            "Upgrade from $PreviousVersion to $Version" `
            $InstallerPath `
            $InstallArguments `
            -WaitProcess
        if (-not (Test-Path -LiteralPath $DataMarker -PathType Leaf)) {
            throw "Version upgrade did not preserve local data."
        }
        Invoke-Checked "Upgraded payload verification" $Python @(
            "scripts/inventory-frozen-payload.py",
            "--payload-root", $InstallRoot,
            "--verify-only",
            "--installer-extras", "windows"
        )
        Invoke-Checked "Upgraded application smoke test" $Python @(
            "scripts/smoke-frozen.py",
            $Application,
            "--version", $Version,
            "--port", "$Port"
        )
    }
    Invoke-Checked `
        "In-place reinstall" `
        $InstallerPath `
        $InstallArguments `
        -WaitProcess
    if (-not (Test-Path -LiteralPath $DataMarker -PathType Leaf)) {
        throw "In-place reinstall did not preserve local data."
    }
    $Uninstaller = Join-Path $InstallRoot "unins000.exe"
    Invoke-Checked "Data-preserving uninstall" $Uninstaller @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART"
    ) -WaitProcess
    if (Test-Path -LiteralPath $InstallRoot) {
        throw "The application directory remains after uninstall."
    }
    if (-not (Test-Path -LiteralPath $DataMarker -PathType Leaf)) {
        throw "Ordinary uninstall did not preserve local data."
    }
    if (Test-Path -LiteralPath $StartMenuShortcut) {
        throw "The Start-menu shortcut remains after uninstall."
    }

    if ($AllowPurge) {
        Invoke-Checked `
            "Reinstall for explicit purge" `
            $InstallerPath `
            $InstallArguments `
            -WaitProcess
        $Uninstaller = Join-Path $InstallRoot "unins000.exe"
        Invoke-Checked "Explicit data purge" $Uninstaller @(
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/PURGEDATA"
        ) -WaitProcess
        if (Test-Path -LiteralPath $DefaultData) {
            throw "Explicit purge left the default data directory in place."
        }
    }

    Write-Host "Windows installer smoke test passed: $InstallerPath"
}
finally {
    if ((Test-Path -LiteralPath $DataMarker) -and -not $AllowPurge) {
        Remove-Item -LiteralPath $DataMarker -Force
        if (
            (Test-Path -LiteralPath $DefaultData -PathType Container) -and
            -not (Get-ChildItem -LiteralPath $DefaultData -Force)
        ) {
            Remove-Item -LiteralPath $DefaultData -Force
        }
    }
}
