[CmdletBinding()]
param(
    [switch]$SkipLinuxPackagingSyntax
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonTools = Join-Path $RepositoryRoot ".venv\Scripts"

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

function Resolve-PythonTool {
    param([Parameter(Mandatory)][string]$Name)

    $Path = Join-Path $PythonTools "$Name.exe"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Path. Create .venv and install services/api[dev] first."
    }
    return $Path
}

Push-Location $RepositoryRoot
try {
    $Ruff = Resolve-PythonTool "ruff"
    $Mypy = Resolve-PythonTool "mypy"
    $Bandit = Resolve-PythonTool "bandit"
    $Pytest = Resolve-PythonTool "pytest"
    $Python = Resolve-PythonTool "python"
    $Npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $Git = (Get-Command git.exe -ErrorAction Stop).Source

    Invoke-Checked "Ruff format" $Ruff @("format", "--check", "services/api")
    Invoke-Checked "Ruff lint" $Ruff @("check", "services/api")
    Invoke-Checked "Strict mypy" $Mypy @("services/api/local_lm")
    Invoke-Checked "Bandit high-severity scan" $Bandit @(
        "-q", "-lll", "-r", "services/api/local_lm"
    )
    $VerificationTemp = Join-Path $RepositoryRoot "temp"
    New-Item -ItemType Directory -Force -Path $VerificationTemp | Out-Null
    $PytestTemp = Join-Path $VerificationTemp "verify-pytest-$PID"
    Invoke-Checked "API tests" $Pytest @(
        "services/api/tests",
        "-q",
        "--basetemp=$PytestTemp",
        "-p",
        "no:cacheprovider"
    )
    Invoke-Checked "Web lint" $Npm @("run", "lint")
    Invoke-Checked "Web typecheck" $Npm @("run", "typecheck")
    Invoke-Checked "Web tests" $Npm @("test")
    Invoke-Checked "Production web build" $Npm @("run", "build")

    Write-Host "==> GitHub workflow YAML"
    $WorkflowCheck = @'
from pathlib import Path
import yaml

paths = sorted(Path('.github/workflows').glob('*.yml'))
paths += sorted(Path('.github/workflows').glob('*.yaml'))
if not paths:
    raise SystemExit('No GitHub workflow files found')
for path in paths:
    with path.open('r', encoding='utf-8') as handle:
        yaml.safe_load(handle)
    print(path)
'@
    Invoke-Checked "GitHub workflow YAML" $Python @("-c", $WorkflowCheck)

    Write-Host "==> Windows packaging syntax"
    $PowerShellErrors = @()
    Get-ChildItem -LiteralPath "packaging/windows" -Filter "*.ps1" | ForEach-Object {
        $Tokens = $null
        $Errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $_.FullName,
            [ref]$Tokens,
            [ref]$Errors
        ) | Out-Null
        if ($Errors.Count -gt 0) {
            $PowerShellErrors += "$($_.Name): $($Errors -join '; ')"
        }
    }
    if ($PowerShellErrors.Count -gt 0) {
        throw "Windows packaging syntax failed: $($PowerShellErrors -join ' | ')"
    }

    if (-not $SkipLinuxPackagingSyntax) {
        $BashCommand = Get-Command bash.exe -ErrorAction SilentlyContinue
        $BashPath = if ($BashCommand) { $BashCommand.Source } else { $null }
        if (-not $BashPath) {
            $GitBash = Join-Path $env:ProgramFiles "Git\bin\bash.exe"
            if (Test-Path -LiteralPath $GitBash -PathType Leaf) {
                $BashPath = $GitBash
            }
        }
        if (-not $BashPath) {
            throw "Git Bash is required for Linux packaging syntax. Use " +
                "-SkipLinuxPackagingSyntax only when an Ubuntu CI run will cover this gate."
        }
        Invoke-Checked "Linux packaging syntax" $BashPath @(
            "-n",
            "scripts/package.sh",
            "packaging/linux/install.sh",
            "packaging/linux/rollback.sh",
            "packaging/linux/start-installed.sh",
            "packaging/linux/uninstall.sh"
        )
    }

    Write-Host "==> Repository hygiene"
    $Tracked = & $Git ls-files
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed with exit code $LASTEXITCODE."
    }
    $RuntimePattern =
        '(^|/)(\.env|\.private|data|models|downloads|artifacts)(/|$)|' +
        '\.(gguf|safetensors|ckpt|sqlite3?)$'
    $UnsafeTracked = @($Tracked | Where-Object { $_ -match $RuntimePattern })
    if ($UnsafeTracked.Count -gt 0) {
        throw "Private or runtime artifacts are tracked: $($UnsafeTracked -join ', ')"
    }

    & $Git grep -I -E `
        '(hf_[A-Za-z0-9]{20,}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY)' `
        -- . ':!package-lock.json' | Out-Null
    $SecretScanExit = $LASTEXITCODE
    if ($SecretScanExit -eq 0) {
        throw "A likely credential is present in tracked source."
    }
    if ($SecretScanExit -ne 1) {
        throw "Tracked-source credential scan failed with exit code $SecretScanExit."
    }

    Invoke-Checked "Git whitespace check" $Git @("diff", "--check", "HEAD", "--")
    Write-Host "All LM Atelier local verification gates passed."
}
finally {
    Pop-Location
}
