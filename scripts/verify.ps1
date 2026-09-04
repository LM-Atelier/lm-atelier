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

    # No stage runs unleased: the leased-stage runner asserts this process
    # still holds its handle before launching the child, and the child
    # inherits the handle, so the exclusion spans the stage for as long
    # as the child can still act.
    Invoke-LeasedStage -Label $Label -FilePath $FilePath `
        -ArgumentList $ArgumentList -Lease $script:MachineLease
}

function Resolve-PythonTool {
    param([Parameter(Mandatory)][string]$Name)

    $Path = Join-Path $PythonTools "$Name.exe"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Path. Create .venv and install services/api[dev,package] first."
    }
    return $Path
}

Push-Location $RepositoryRoot
$MachineLease = $null
$LeaseReleaseFailed = $false
$GateBodyPassed = $false
try {
    $Ruff = Resolve-PythonTool "ruff"
    $Mypy = Resolve-PythonTool "mypy"
    $Bandit = Resolve-PythonTool "bandit"
    $Pytest = Resolve-PythonTool "pytest"
    $Python = Resolve-PythonTool "python"
    $Npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $Git = (Get-Command git.exe -ErrorAction Stop).Source

    # The whole gate is MACHINE-EXCLUSIVE: a package-wide mypy or a mutation
    # battery running beside it can hang it outright, and convention alone
    # does not prevent that. The gate holds a KERNEL-OWNED HANDLE for its
    # whole run: the open is the acquisition, every stage child inherits
    # the handle, and the kernel frees the machine the instant the last
    # holder dies - there is nothing to renew and nothing to recover.
    Write-Host "==> Machine-exclusive lease"
    . (Join-Path $RepositoryRoot "scripts\machine-lease.ps1")
    . (Join-Path $RepositoryRoot "scripts\held-pytest-scratch.ps1")
    $MachineLease = Enter-MachineLease -RepositoryRoot $RepositoryRoot -Purpose "verify.ps1"
    if (-not $MachineLease) {
        # The refusal above says WHY (a live holder, an unreadable record, a
        # broken repo resolution); this throw only stops the gate - labeling
        # every failure "held" would send people hunting a missing holder.
        throw "The machine-exclusive lease was not acquired; the line above says why. Investigate with 'python scripts/machine_lock.py status'."
    }

    # Direct stages run OUTSIDE Invoke-Checked, so each carries the barrier
    # itself: real work behind an unguarded banner is exactly where a lost
    # lease slips a stage past the exclusivity claim.
    Assert-MachineLeaseHeld -Lease $MachineLease
    Write-Host "==> Import identity"
    $ImportedDirectory = & $Python @(
        "-c",
        "import local_lm, pathlib; print(pathlib.Path(local_lm.__file__).resolve().parent)"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Could not import local_lm, so which tree these gates would measure is unknown."
    }
    Assert-MachineLeaseHeld -Lease $MachineLease
    $Imported = (Resolve-Path -LiteralPath $ImportedDirectory).Path
    $ExpectedPackage = (Resolve-Path -LiteralPath (
        Join-Path $RepositoryRoot "services\api\local_lm"
    )).Path
    # EXACT identity, not containment. Linked worktrees can live below the main
    # checkout, so a nested worktree's package sits under the main root and any prefix or
    # containment test calls it "inside this one". That is the wrong answer,
    # and it is the one a run from the main checkout with PYTHONPATH pointed at
    # a worktree would get.
    if (-not $Imported.Equals($ExpectedPackage, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw (
            "local_lm imports from a different tree than this one, so every gate " +
            "below would test the wrong code.`n" +
            "  imported from : $Imported`n" +
            "  expected      : $ExpectedPackage`n" +
            "Set PYTHONPATH to this worktree before running:`n" +
            "  `$env:PYTHONPATH = `"$(Join-Path $RepositoryRoot 'services\api')`""
        )
    }

    Invoke-Checked "Ruff format" $Ruff @("format", "--check", "services/api")
    Invoke-Checked "Ruff lint" $Ruff @("check", "services/api")
    # Without the explicit config mypy does not discover the nested API
    # pyproject from the repository root, so this ran with default settings
    # while being labelled strict. The label is the promise; the flag is
    # what keeps it.
    Invoke-Checked "Strict mypy" $Mypy @(
        "--config-file", "services/api/pyproject.toml", "services/api/local_lm"
    )
    Invoke-Checked "Strict mypy (Linux platform)" $Mypy @(
        "--platform", "linux",
        "--config-file", "services/api/pyproject.toml", "services/api/local_lm"
    )
    Invoke-Checked "Bandit high-severity scan" $Bandit @(
        "-q", "-lll", "-r", "services/api/local_lm"
    )
    Invoke-Checked "Version metadata" $Python @("scripts/sync-version.py")
    $PytestTemp = New-HeldPytestScratch `
        -RepositoryRoot $RepositoryRoot -Lease $MachineLease
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
    Invoke-Checked "Browser suite typecheck" $Npm @("run", "e2e:typecheck")
    Invoke-Checked "Browser suite discovery" $Npm @("run", "e2e:list")
    Invoke-Checked "Production web build" $Npm @("run", "build")

    Invoke-Checked "GitHub workflow policy" $Python @(
        "scripts/validate-workflows.py"
    )

    Assert-MachineLeaseHeld -Lease $MachineLease
    Write-Host "==> Windows packaging syntax"
    $PowerShellErrors = @()
    Get-ChildItem -LiteralPath "packaging/windows", "scripts" -Filter "*.ps1" | ForEach-Object {
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
    Assert-MachineLeaseHeld -Lease $MachineLease

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
            "scripts/build-linux-installer.sh",
            "scripts/smoke-linux-installer.sh",
            "packaging/linux/self-extracting-installer.sh",
            "packaging/linux/frozen-uninstall.sh"
        )
    }

    Invoke-Checked "Repository hygiene" $Python @(
        "scripts/check-repository-hygiene.py"
    )

    Invoke-Checked "Unstaged whitespace check" $Git @("diff", "--check", "--")
    Invoke-Checked "Staged whitespace check" $Git @(
        "diff", "--cached", "--check", "--"
    )
    if ($env:GITHUB_BASE_REF) {
        Invoke-Checked "Pull-request whitespace check" $Git @(
            "diff",
            "--check",
            "origin/$($env:GITHUB_BASE_REF)...HEAD",
            "--"
        )
    } else {
        Invoke-Checked "Latest commit whitespace check" $Git @(
            "log", "--check", "--format=", "-1", "HEAD", "--"
        )
    }
    # The success banner is printed by the lease completion below, only
    # after the release succeeded: a green line followed by a release
    # error would claim a clean run the exit code then contradicts.
    $GateBodyPassed = $true
}
finally {
    if ($MachineLease) {
        # A leaked lease means the machine stops being provably exclusive,
        # and a LASTEXITCODE reset here would hide exactly that: a green
        # gate whose release failed reporting clean. The failure
        # PROPAGATES below - though a failure thrown by the body itself
        # still wins, because this block only records. The success banner
        # and the browser note are printed here, after the release, and
        # only for a body that completed.
        $Completed = Complete-MachineLeaseRun -Lease $MachineLease -BodyPassed $GateBodyPassed `
            -SuccessMessage "All LM Atelier local verification gates passed." `
            -Epilogue "Not run here: the browser golden path. Use 'npm run e2e'; CI runs it on Ubuntu."
        if (-not $Completed) {
            $LeaseReleaseFailed = $true
        }
    }
    Pop-Location
}
if ($LeaseReleaseFailed) {
    exit 2
}
