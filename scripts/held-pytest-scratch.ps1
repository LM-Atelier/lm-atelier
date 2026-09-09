function Test-PytestScratchContainedBy {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root
    )

    $Trim = [char[]]@(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $Candidate = [IO.Path]::GetFullPath($Path).TrimEnd($Trim)
    $Boundary = [IO.Path]::GetFullPath($Root).TrimEnd($Trim)
    return (
        $Candidate.Equals($Boundary, [StringComparison]::OrdinalIgnoreCase) -or
        $Candidate.StartsWith(
            $Boundary + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    )
}

function New-HeldPytestScratch {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)]$Lease
    )

    Assert-MachineLeaseHeld -Lease $Lease
    $NamedRoot = [Environment]::GetEnvironmentVariable("RUNNER_TEMP")
    if ([String]::IsNullOrWhiteSpace($NamedRoot)) {
        $NamedRoot = [IO.Path]::GetTempPath()
    }
    $Probe = Open-MachineLeaseDirectory -Path $NamedRoot
    if (Test-MachineLeaseHandleInvalid $Probe) {
        throw "The pytest scratch root could not be opened: $NamedRoot."
    }
    try {
        $RootIdentity = Get-MachineLeaseIdentity -Handle $Probe
        $RootName = Get-MachineLeaseFinalPath -Handle $Probe
    } finally {
        [LeaseNative.Kernel]::CloseHandle($Probe) | Out-Null
    }
    if (-not $RootIdentity -or -not $RootName) {
        throw "The pytest scratch root could not be identified."
    }

    $HeldRoot = ConvertTo-MachineLeasePlainName $RootName
    $HeldRepository = ConvertTo-MachineLeasePlainName $Lease.Binding.Anchor
    if (Test-PytestScratchContainedBy -Path $HeldRoot -Root $HeldRepository) {
        throw "The pytest scratch root resolves inside the repository."
    }

    $Pins = @()
    try {
        foreach ($Link in Get-MachineLeaseReparseLinks `
            -Path $NamedRoot -Role "the pytest scratch root" -Itself) {
            $Pins += Open-MachineLeasePin -Path $Link.Path -Role $Link.Role
        }
        $Pins += Open-MachineLeasePin `
            -Path $HeldRoot -Role "the pytest scratch root directory"

        $Again = Open-MachineLeaseDirectory -Path $NamedRoot
        if (Test-MachineLeaseHandleInvalid $Again) {
            throw "The pinned pytest scratch root could not be reopened."
        }
        try {
            $AgainIdentity = Get-MachineLeaseIdentity -Handle $Again
            $AgainName = Get-MachineLeaseFinalPath -Handle $Again
        } finally {
            [LeaseNative.Kernel]::CloseHandle($Again) | Out-Null
        }
        if (
            $AgainIdentity -ne $RootIdentity -or
            $AgainName -ne $RootName
        ) {
            throw "The pytest scratch root changed while it was being held."
        }

        $Parent = Join-Path $HeldRoot (
            "lm-atelier-verify-" + [Guid]::NewGuid().ToString("N")
        )
        if (Test-Path -LiteralPath $Parent) {
            throw "The new pytest scratch directory already exists."
        }
        New-Item -ItemType Directory -Path $Parent -ErrorAction Stop | Out-Null
        $ParentPin = Open-MachineLeasePin `
            -Path $Parent -Role "the pytest scratch directory"
        $Pins += $ParentPin
        $ParentItem = Get-Item -LiteralPath $Parent -Force -ErrorAction Stop
        $ParentName = Get-MachineLeaseFinalPath -Handle $ParentPin.Handle
        if (
            ($ParentItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            (ConvertTo-MachineLeasePlainName $ParentName) -ne $Parent
        ) {
            throw "The new pytest scratch directory is not the held directory."
        }

        Assert-MachineLeaseHeld -Lease $Lease
        foreach ($Pin in $Pins) {
            if (-not (Get-MachineLeaseIdentity -Handle $Pin.Handle)) {
                throw "A pytest scratch hold was lost during selection."
            }
        }
        $PytestPath = Join-Path $Parent "pytest"
        $Lease.Binding.Pins = @($Lease.Binding.Pins) + @($Pins)
        return $PytestPath
    } catch {
        if (-not (Close-MachineLeaseAcquired -Pins $Pins -During "a refused pytest scratch selection")) {
            exit 4
        }
        throw
    }
}
