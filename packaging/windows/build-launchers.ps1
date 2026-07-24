param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$LauncherRoot = Join-Path $PSScriptRoot "launcher"
$CompilerCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)
$Compiler = $CompilerCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $Compiler) {
    throw "The Windows .NET Framework C# compiler is required to build the launchers."
}

$ResolvedOutput = New-Item -ItemType Directory -Force -Path $OutputDirectory

function Build-Launcher {
    param(
        [string]$Source,
        [string]$Name
    )

    $Output = Join-Path $ResolvedOutput.FullName $Name
    & $Compiler `
        /nologo `
        /target:winexe `
        /platform:x64 `
        /optimize+ `
        /reference:System.dll `
        /reference:System.Windows.Forms.dll `
        "/out:$Output" `
        (Join-Path $LauncherRoot $Source)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not build $Name."
    }
}

Build-Launcher -Source "SetupLauncher.cs" -Name "Setup LM Atelier.exe"
Build-Launcher -Source "LMAtelierLauncher.cs" -Name "LM Atelier.exe"
