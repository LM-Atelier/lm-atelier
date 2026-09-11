# Machine-lease helpers for the gate: a kernel-held, inheritable handle.
#
# Enter-MachineLease OPENS the lease file with write access while sharing
# only reads; the kernel enforces the exclusion, so a second write-access
# open fails while this process - or any stage child that inherited the
# handle - is alive. There is nothing to renew and nothing to recover: the
# machine frees the instant the last holder dies. Exit-MachineLease closes
# the handle. The bytes in the file are diagnostics, never authority.
#
# The lease file is opened under the repository's COMMON git directory while
# that directory is HELD: the repository directory is opened as an object
# first, the common directory is resolved through its held name and held the
# same way, every link of the resolution chain - the .git entry, a pointer's
# private directory and commondir file, the common directory itself - is
# then pinned read-share only for the lease lifetime, and the resolution is
# repeated through the pinned chain before the lease is opened. A directory
# replaced under its name or a pointer moved elsewhere before the pins took
# hold is refused rather than leased; once they hold, the kernel refuses the
# writes that would move the binding, for as long as the lease and any stage
# child it covers can act. The barrier at every stage checks the pins.
#
# Dot-source this file; it defines functions only and runs nothing.

if (-not ("LeaseNative.Kernel" -as [type])) {
    Add-Type -Namespace LeaseNative -Name Kernel -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)]
public struct FileInformation {
    public uint FileAttributes;
    public uint CreationLow;
    public uint CreationHigh;
    public uint AccessLow;
    public uint AccessHigh;
    public uint WriteLow;
    public uint WriteHigh;
    public uint VolumeSerialNumber;
    public uint FileSizeHigh;
    public uint FileSizeLow;
    public uint NumberOfLinks;
    public uint FileIndexHigh;
    public uint FileIndexLow;
}
[DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern IntPtr CreateFileW(
    string name, uint access, uint share, IntPtr security,
    uint disposition, uint flags, IntPtr template);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool CloseHandle(IntPtr handle);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetFileInformationByHandle(IntPtr handle, out FileInformation information);
[DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern uint GetFinalPathNameByHandleW(
    IntPtr handle, System.Text.StringBuilder path, uint length, uint flags);
'@
}

function Test-MachineLeaseHandleInvalid {
    param($Handle)
    return ($Handle -eq [IntPtr]::Zero -or $Handle -eq [IntPtr]::new(-1))
}

function Read-MachineLeaseEnvironment {
    # The one place this file reads the process environment back, so that the
    # restoration check has a seam a control can stand in front of.
    #
    # Without it that branch cannot be reached at all: the real setter
    # round-trips a value exactly, so no test can arrange for the read to
    # differ from what was just written, and the comparison below would be
    # asserted only by reading it.
    param([Parameter(Mandatory = $true)][string] $Name)
    return [Environment]::GetEnvironmentVariable($Name)
}

function Get-MachineLeaseCommonDir {
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    # The COMMON git dir is shared by every linked worktree, so every
    # checkout of this repository resolves the same lease file. Git runs
    # with its redirection environment scrubbed so inherited GIT_DIR state
    # cannot point the lease elsewhere.
    $Scrubbed = @(
        "GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES", "GIT_PREFIX"
    )
    # REMOVED, not emptied, and VERIFIED in both directions. Binding $null
    # to a [string] parameter yields the empty string in some PowerShell
    # hosts, and git reads an empty GIT_DIR as a repository path, so a
    # variable that is merely set empty still redirects. Absence is
    # therefore tested rather than requested.
    #
    # The restoration comparison asks for the SAME BYTES, which needs an
    # explicit ordinal comparison rather than any PowerShell operator. `-ne` is
    # case-insensitive, so a changed case read as restored; `-cne` is
    # case-sensitive but still cultural, so two different strings the current
    # culture considers equal also read as restored. Measured rather than
    # assumed: -cne calls such a pair equal and [string]::Equals with
    # StringComparison.Ordinal does not.
    #
    # Every scrubbed name is path-valued, so no caller was pointed at the wrong
    # repository by either gap - but the guard did not hold the property it is
    # here for, which is that the environment is what it was.
    #
    # Both halves fail closed, and the resolution is only returned when
    # both held. A removal that did not take effect would leave a variable
    # naming another repository, and git would resolve THAT repository and
    # the gate would take its lease. A restoration that did not take effect
    # would hand the caller a correct answer while leaving its environment
    # redirected for every later git command, which is the same failure one
    # step further on.
    $Saved = @{}
    $Cleared = $true
    foreach ($Name in $Scrubbed) {
        $Saved[$Name] = Read-MachineLeaseEnvironment -Name $Name
        Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath "Env:$Name") {
            Write-Host "ERROR: $Name could not be removed from the environment; refusing to resolve the repository with git redirection in force."
            $Cleared = $false
        }
    }
    $Common = $null
    $Resolved = $false
    $Restored = $true
    try {
        if ($Cleared) {
            $Common = & git -C $RepositoryRoot rev-parse --path-format=absolute --git-common-dir
            $Resolved = ($LASTEXITCODE -eq 0 -and [bool]$Common)
        }
    } finally {
        foreach ($Name in $Scrubbed) {
            if ($null -eq $Saved[$Name]) {
                Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
                if (Test-Path -LiteralPath "Env:$Name") {
                    Write-Host "ERROR: $Name was absent before this resolution and could not be removed again; refusing the resolution rather than leaving this process redirected."
                    $Restored = $false
                }
            } else {
                [Environment]::SetEnvironmentVariable($Name, $Saved[$Name])
                $Current = Read-MachineLeaseEnvironment -Name $Name
                if (-not [string]::Equals($Current, $Saved[$Name], [StringComparison]::Ordinal)) {
                    Write-Host "ERROR: $Name could not be restored to the value it had; refusing the resolution rather than leaving this process redirected."
                    $Restored = $false
                }
            }
        }
    }
    if (-not $Resolved -or -not $Restored) {
        return $null
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Common "HEAD") -PathType Leaf)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Common "config") -PathType Leaf)) {
        return $null
    }
    return $Common
}

function Get-MachineLeasePath {
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    $Common = Get-MachineLeaseCommonDir -RepositoryRoot $RepositoryRoot
    if (-not $Common) {
        return $null
    }
    return (Join-Path $Common "machine-exclusive.lease")
}

function Get-MachineLeaseIdentity {
    param([Parameter(Mandatory)]$Handle)

    # (volume serial, file index) of the object behind a live handle; $null
    # when the handle is not live - a closed or reused handle answers for
    # nothing.
    $Information = New-Object LeaseNative.Kernel+FileInformation
    if (-not [LeaseNative.Kernel]::GetFileInformationByHandle($Handle, [ref]$Information)) {
        return $null
    }
    return "$($Information.VolumeSerialNumber):$($Information.FileIndexHigh):$($Information.FileIndexLow)"
}

function Get-MachineLeaseFinalPath {
    param([Parameter(Mandatory)]$Handle)

    # The held object's current name, read from the handle rather than
    # looked up by a name that may since have been given to something else.
    $Buffer = New-Object System.Text.StringBuilder 32768
    $Length = [LeaseNative.Kernel]::GetFinalPathNameByHandleW($Handle, $Buffer, [uint32]$Buffer.Capacity, [uint32]0)
    if ($Length -eq 0 -or $Length -ge $Buffer.Capacity) {
        return $null
    }
    return $Buffer.ToString()
}

function Open-MachineLeaseDirectory {
    param([Parameter(Mandatory)][string]$Path)

    # FILE_READ_ATTRIBUTES, share read|write|delete, OPEN_EXISTING,
    # FILE_FLAG_BACKUP_SEMANTICS (directories need it): a hold on the
    # directory OBJECT that blocks nobody. The Win32 error is read on the
    # statement after the call: anything the engine runs in between - a
    # function call, a variable lookup - can overwrite it.
    $Handle = [LeaseNative.Kernel]::CreateFileW(
        $Path, [uint32]128, [uint32]7, [IntPtr]::Zero,
        [uint32]3, [uint32]33554432, [IntPtr]::Zero
    )
    $script:MachineLeaseLastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    return $Handle
}

function ConvertTo-MachineLeasePlainName {
    param([Parameter(Mandatory)][string]$Name)

    # A final name carries the \\?\ prefix, which the path cmdlets and git
    # do not take; the plain spelling names the same object.
    if ($Name.StartsWith('\\?\UNC\')) { return '\\' + $Name.Substring(8) }
    if ($Name.StartsWith('\\?\')) { return $Name.Substring(4) }
    return $Name
}

function Get-MachineLeaseDirectoryIdentity {
    param([Parameter(Mandatory)][string]$Path)

    $Probe = Open-MachineLeaseDirectory -Path $Path
    if (Test-MachineLeaseHandleInvalid $Probe) {
        return $null
    }
    $Failure = $null
    try {
        return Get-MachineLeaseIdentity -Handle $Probe
    } catch {
        $Failure = $_
        throw
    } finally {
        Close-MachineLeaseDirectoryProbe -Handle $Probe -Failure $Failure -During "directory identity lookup"
    }
}

function Add-MachineLeasePathPins {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Role,
        [Parameter(Mandatory)][AllowEmptyCollection()][Collections.ArrayList]$Pins,
        [Parameter(Mandatory)][AllowEmptyCollection()][Collections.Generic.HashSet[string]]$Seen,
        [switch]$Itself
    )

    # Hold every component before inspecting it, including ordinary directories.
    $Full = [IO.Path]::GetFullPath($Path)
    $Prefixes = @()
    $Current = Split-Path -Parent $Full
    while ($Current -and (Split-Path -Parent $Current)) {
        $Prefixes = @($Current) + $Prefixes
        $Current = Split-Path -Parent $Current
    }
    if ($Itself) { $Prefixes += $Full }
    foreach ($Prefix in $Prefixes) {
        if (-not $Seen.Add($Prefix)) { continue }
        $PinRole = if ($Itself -and $Prefix -eq $Full) { $Role } else { "a component on the way to $Role" }
        [void]$Pins.Add((Open-MachineLeasePin -Path $Prefix -Role $PinRole))
        $Item = Get-Item -LiteralPath $Prefix -Force -ErrorAction Stop
        if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $Probe = Open-MachineLeaseDirectory -Path $Prefix
            if (Test-MachineLeaseHandleInvalid $Probe) {
                throw "the link target could not be held: $Prefix"
            }
            $Failure = $null
            try {
                $Identity = Get-MachineLeaseIdentity -Handle $Probe
                $Target = ConvertTo-MachineLeasePlainName (Get-MachineLeaseFinalPath -Handle $Probe)
                if (-not $Identity -or -not $Target) { throw "the link target could not be identified: $Prefix" }
            } catch {
                $Failure = $_
                throw
            } finally {
                Close-MachineLeaseDirectoryProbe -Handle $Probe -Failure $Failure -During "link target lookup"
            }
            Add-MachineLeasePathPins -Path $Target -Role $Role -Pins $Pins -Seen $Seen -Itself
            if ((Get-MachineLeaseDirectoryIdentity -Path $Prefix) -ne $Identity -or
                (Get-MachineLeaseDirectoryIdentity -Path $Target) -ne $Identity) {
                throw "a link target changed while its path was being held: $Prefix"
            }
        }
    }
}

function Get-MachineLeaseNamedDirectory {
    param(
        [Parameter(Mandatory)][string]$Base,
        [Parameter(Mandatory)][string]$File
    )
    $Text = [IO.File]::ReadAllText($File).Trim()
    if ([IO.Path]::IsPathRooted($Text)) { return $Text }
    return (Join-Path $Base $Text)
}

function Get-MachineLeaseResolutionChain {
    param(
        [Parameter(Mandatory)][string]$Anchor,
        [Parameter(Mandatory)][AllowEmptyCollection()][Collections.ArrayList]$Pins
    )

    $Seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    Add-MachineLeasePathPins -Path $Anchor -Role "the repository directory" -Pins $Pins -Seen $Seen -Itself
    $Entry = Join-Path $Anchor ".git"
    Add-MachineLeasePathPins -Path $Entry -Role "the repository's .git entry" -Pins $Pins -Seen $Seen -Itself
    $Item = Get-Item -LiteralPath $Entry -Force -ErrorAction Stop
    if ($Item.PSIsContainer) {
        $Private = $Entry
    } else {
        $Text = [IO.File]::ReadAllText($Entry)
        if (-not $Text.StartsWith("gitdir:")) { throw "the .git file is not a git pointer: $Entry" }
        $Private = $Text.Substring(7).Trim()
        if (-not [IO.Path]::IsPathRooted($Private)) { $Private = Join-Path $Anchor $Private }
        Add-MachineLeasePathPins -Path $Private -Role "the checkout's private git directory" -Pins $Pins -Seen $Seen -Itself
    }
    $CommonDir = Join-Path $Private "commondir"
    if (Test-Path -LiteralPath $CommonDir -PathType Leaf) {
        Add-MachineLeasePathPins -Path $CommonDir -Role "the commondir file" -Pins $Pins -Seen $Seen -Itself
        $Named = Get-MachineLeaseNamedDirectory -Base $Private -File $CommonDir
        Add-MachineLeasePathPins -Path $Named -Role "the common git directory" -Pins $Pins -Seen $Seen -Itself
    }
}

function Open-MachineLeasePin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Role
    )
    return Open-MachineLeaseSharedPin -Path $Path -Role $Role -Share 1
}

function Open-MachineLeaseSharedPin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Role,
        [Parameter(Mandatory)][uint32]$Share
    )

    # Hold the component itself, with the caller's sharing mode. Neither
    # strict nor parent sharing permits rename or deletion of this name.
    $Flags = [uint32]33554432 -bor [uint32]2097152
    # FILE_READ_ATTRIBUTES | GENERIC_READ, selected sharing, OPEN_EXISTING.
    $Handle = [LeaseNative.Kernel]::CreateFileW(
        $Path, [uint32]2147483776, $Share, [IntPtr]::Zero,
        [uint32]3, $Flags, [IntPtr]::Zero
    )
    $script:MachineLeaseLastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    if (Test-MachineLeaseHandleInvalid $Handle) {
        if ($script:MachineLeaseLastError -eq 32) {
            throw "$Role is open for writing elsewhere: $Path"
        }
        throw "$Role could not be held: $Path (error $script:MachineLeaseLastError)"
    }
    # Inheritable, exactly as the lease is: a stage child launched while the
    # lease is held carries every pin for as long as it can act, so the
    # chain stays held for the child's lifetime even when the holder dies
    # first. A pin that cannot be marked is let go, and a close the kernel
    # refuses on that path strands this process.
    if (-not [LeaseNative.Kernel]::SetHandleInformation($Handle, 0x1, 0x1)) {
        $MarkError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        # Hand the outcome back rather than exiting here. Exiting from inside
        # the pin opener leaves the caller's earlier pins unattempted and
        # unreported, so the single report the design promises would name this
        # pin and nothing else. The flag rides out with the throw; the caller
        # closes what it holds, then exits stranded knowing about both.
        if (-not (Close-MachineLeaseAcquired -Pins @([pscustomobject]@{ Path = $Path; Role = $Role; Handle = $Handle }) -During "a refused pinning")) {
            $script:MachineLeaseStranded = $true
        }
        throw "$Role could not be held for a child's lifetime: $Path (error $MarkError)"
    }
    return [pscustomobject]@{ Path = $Path; Role = $Role; Handle = $Handle }
}


function Enable-MachineLeaseParentWrites {
    param([Parameter(Mandatory)][AllowEmptyCollection()][Collections.ArrayList]$Pins)

    # A held direct child cannot be removed, keeping an ordinary parent
    # nonempty and unable to become a junction. Keep files, links and leaves
    # write-exclusive; every replacement still excludes rename and delete.
    $Parents = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($Pin in $Pins) {
        [void]$Parents.Add([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Pin.Path)))
    }
    for ($Index = 0; $Index -lt $Pins.Count; $Index++) {
        $Original = $Pins[$Index]
        if (-not $Parents.Contains([IO.Path]::GetFullPath($Original.Path))) { continue }
        $Information = New-Object LeaseNative.Kernel+FileInformation
        if (-not [LeaseNative.Kernel]::GetFileInformationByHandle($Original.Handle, [ref]$Information)) {
            throw "a held parent could not be identified"
        }
        if (-not ($Information.FileAttributes -band 16) -or ($Information.FileAttributes -band 1024)) {
            continue
        }
        $Replacement = Open-MachineLeaseSharedPin -Path $Original.Path -Role $Original.Role -Share 3
        [void]$Pins.Add($Replacement)
        $OriginalIdentity = Get-MachineLeaseIdentity -Handle $Original.Handle
        if (-not $OriginalIdentity -or (Get-MachineLeaseIdentity -Handle $Replacement.Handle) -ne $OriginalIdentity) {
            throw "a held parent changed while its sharing was adjusted"
        }
        $Pins[$Index] = $Replacement
        $Pins.RemoveAt($Pins.Count - 1)
        if (-not (Close-MachineLeaseAcquired -Pins @($Original) -During "parent sharing adjustment")) {
            $script:MachineLeaseStranded = $true
            throw "a strict parent hold could not be retired"
        }
    }
}

function Close-MachineLeaseAcquired {
    param($Handle, $Pins, $Probes, [Parameter(Mandatory)][string]$During)

    # Close everything a boundary acquired, each exactly once - the lease
    # handle when there is one, then every pin - and report every close the
    # kernel refused: nothing is skipped because an earlier close refused.
    # $true when all of them closed; otherwise this process is stranded
    # holding what would not close, and the caller says so.
    $Ok = $true
    if ($null -ne $Handle -and -not (Test-MachineLeaseHandleInvalid $Handle)) {
        if (-not [LeaseNative.Kernel]::CloseHandle($Handle)) {
            $LastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            Write-Host "ERROR: CloseHandle refused the lease handle $Handle after $During (error $LastError); the handle was NOT closed and this process is stranded holding the machine."
            $Ok = $false
        }
    }
    foreach ($Pin in @($Pins)) {
        if ($null -eq $Pin) { continue }
        if (-not [LeaseNative.Kernel]::CloseHandle($Pin.Handle)) {
            $LastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            Write-Host "ERROR: CloseHandle refused the pin on $($Pin.Role) $($Pin.Path) after $During (error $LastError); this process is stranded holding the checkout binding."
            $Ok = $false
        }
    }
    foreach ($Probe in @($Probes)) {
        if ($null -eq $Probe -or (Test-MachineLeaseHandleInvalid $Probe)) { continue }
        if (-not [LeaseNative.Kernel]::CloseHandle($Probe)) {
            $LastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            Write-Host "ERROR: CloseHandle refused the directory probe $Probe after $During (error $LastError); this process retains a temporary kernel handle."
            $Ok = $false
        }
    }
    return $Ok
}

function Close-MachineLeaseDirectoryProbe {
    param($Handle, $Failure, [Parameter(Mandatory)][string]$During)

    if (-not (Close-MachineLeaseAcquired -Probes @($Handle) -During $During)) {
        $script:MachineLeaseStranded = $true
        $Message = "A temporary directory probe could not be closed after $During."
        if ($Failure) {
            throw [System.InvalidOperationException]::new($Message, $Failure.Exception)
        }
        throw [System.InvalidOperationException]::new($Message)
    }
}

function Test-MachineLeaseBinding {
    param([Parameter(Mandatory)]$Binding)

    # $null while the repository still names the held common directory:
    # the repository directory opened by the name it had when the lease was
    # taken is the same object, and the common directory resolved through it
    # is the object the lease lives in. Otherwise the reason the binding no
    # longer holds.
    $AnchorPlain = ConvertTo-MachineLeasePlainName $Binding.Anchor
    $Anchor = Open-MachineLeaseDirectory -Path $AnchorPlain
    if (Test-MachineLeaseHandleInvalid $Anchor) {
        return "the repository directory $AnchorPlain could not be opened"
    }
    $Failure = $null
    try {
        if ((Get-MachineLeaseIdentity -Handle $Anchor) -ne $Binding.AnchorIdentity) {
            return "$AnchorPlain is another object"
        }
        $Resolved = Get-MachineLeaseCommonDir -RepositoryRoot $AnchorPlain
    } catch {
        $Failure = $_
        throw
    } finally {
        Close-MachineLeaseDirectoryProbe -Handle $Anchor -Failure $Failure -During "binding verification"
    }
    if (-not $Resolved) {
        return "$AnchorPlain no longer resolves a git repository"
    }
    if ((Get-MachineLeaseDirectoryIdentity -Path $Resolved) -ne $Binding.CommonIdentity) {
        return "its common git directory is now $Resolved, not the one held"
    }
    return $null
}

function Open-MachineLeaseHandle {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][uint32]$Access,
        [Parameter(Mandatory)][uint32]$Disposition
    )

    # Own every acquired object until both temporary probes have closed.
    # A refusal leaves no successful result over unreported live handles.
    $Anchor = $null
    $Directory = $null
    $Handle = $null
    $Pins = [Collections.ArrayList]::new()
    $Transferred = $false
    $script:MachineLeaseStranded = $false
    try {
        $Anchor = Open-MachineLeaseDirectory -Path $RepositoryRoot
        if (Test-MachineLeaseHandleInvalid $Anchor) {
            Write-Host "ERROR: the repository directory could not be held: $RepositoryRoot (error $script:MachineLeaseLastError)."
            return $null
        }
        $AnchorIdentity = Get-MachineLeaseIdentity -Handle $Anchor
        $AnchorName = Get-MachineLeaseFinalPath -Handle $Anchor
        if (-not $AnchorIdentity -or -not $AnchorName) {
            Write-Host "ERROR: the held repository directory could not be identified."
            return $null
        }
        $AnchorPlain = ConvertTo-MachineLeasePlainName $AnchorName
        $Common = Get-MachineLeaseCommonDir -RepositoryRoot $AnchorPlain
        if (-not $Common) {
            Write-Host "ERROR: $RepositoryRoot does not resolve a git repository for the lease."
            return $null
        }
        $Directory = Open-MachineLeaseDirectory -Path $Common
        if (Test-MachineLeaseHandleInvalid $Directory) {
            Write-Host "ERROR: the common git directory could not be held: $Common (error $script:MachineLeaseLastError)."
            return $null
        }
        $Identity = Get-MachineLeaseIdentity -Handle $Directory
        $Name = Get-MachineLeaseFinalPath -Handle $Directory
        if (-not $Identity -or -not $Name) {
            Write-Host "ERROR: the held common git directory could not be identified."
            return $null
        }
        if ((Get-MachineLeaseFinalPath -Handle $Anchor) -ne $AnchorName -or
            (Get-MachineLeaseIdentity -Handle $Anchor) -ne $AnchorIdentity) {
            Write-Host "ERROR: the repository directory moved while it was being resolved."
            return $null
        }
        $Again = Get-MachineLeaseCommonDir -RepositoryRoot $AnchorPlain
        if (-not $Again -or (Get-MachineLeaseDirectoryIdentity -Path $Again) -ne $Identity) {
            Write-Host "ERROR: the repository's common git directory changed while it was being held: now $Again."
            return $null
        }
        $Plain = ConvertTo-MachineLeasePlainName $Name
        try {
            Get-MachineLeaseResolutionChain -Anchor $AnchorPlain -Pins $Pins
            Enable-MachineLeaseParentWrites -Pins $Pins
            [void]$Pins.Add((Open-MachineLeasePin -Path $Plain -Role "the common git directory"))
        } catch {
            Write-Host "ERROR: the resolution chain could not be pinned. $_"
            return $null
        }
        $Pinned = Get-MachineLeaseCommonDir -RepositoryRoot $AnchorPlain
        if (-not $Pinned -or (Get-MachineLeaseDirectoryIdentity -Path $Pinned) -ne $Identity) {
            Write-Host "ERROR: the repository's common git directory changed while it was being pinned: now $Pinned."
            return $null
        }
        if (-not (Test-Path -LiteralPath (Join-Path $Plain "HEAD") -PathType Leaf) -or
            -not (Test-Path -LiteralPath (Join-Path $Plain "config") -PathType Leaf)) {
            Write-Host "ERROR: the held common git directory is not a git dir: $Plain."
            return $null
        }
        $LeasePath = Join-Path $Plain "machine-exclusive.lease"
        $Handle = [LeaseNative.Kernel]::CreateFileW(
            $LeasePath, $Access, [uint32]1, [IntPtr]::Zero,
            $Disposition, [uint32]128, [IntPtr]::Zero
        )
        $Error32 = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        if (Test-MachineLeaseHandleInvalid $Handle) {
            $Reason = "the lease file could not be opened (error $Error32)"
            if ($Error32 -eq 32) {
                $Recorded = ""
                try { $Recorded = [IO.File]::ReadAllText($LeasePath) } catch {}
                $Reason = "the machine is held: $Recorded"
            }
            Write-Host "ERROR: $Reason"
            return $null
        }
        $Binding = [pscustomobject]@{
            Anchor = $AnchorName
            AnchorIdentity = $AnchorIdentity
            Common = $Name
            CommonIdentity = $Identity
            Pins = $Pins.ToArray()
        }
        $Drift = Test-MachineLeaseBinding -Binding $Binding
        if ($Drift) {
            Write-Host "ERROR: the repository's common git directory changed during acquisition: $Drift."
            return $null
        }
        # Mark each probe transferred to its closer before attempting it.
        # A refused handle value must never be retried during outer cleanup.
        $Retired = $Directory
        $Directory = $null
        Close-MachineLeaseDirectoryProbe -Handle $Retired -During "lease acquisition"
        $Retired = $Anchor
        $Anchor = $null
        Close-MachineLeaseDirectoryProbe -Handle $Retired -During "repository resolution"
        $Transferred = $true
        return [pscustomobject]@{
            Handle = $Handle
            Path = $LeasePath
            Binding = $Binding
        }
    } catch {
        Write-Host "ERROR: the lease acquisition was refused. $_"
        return $null
    } finally {
        if (-not $Transferred) {
            $Closed = Close-MachineLeaseAcquired -Handle $Handle -Pins $Pins -Probes @($Directory, $Anchor) -During "a refused acquisition"
            if (-not $Closed -or $script:MachineLeaseStranded) {
                exit 4
            }
        }
    }
}

function Enter-MachineLease {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$Purpose
    )

    # GENERIC_READ|GENERIC_WRITE (0xC0000000 as decimal: PowerShell 5.1
    # reads the hex literal as a negative Int32), OPEN_ALWAYS.
    $Opened = Open-MachineLeaseHandle -RepositoryRoot $RepositoryRoot `
        -Access ([uint32]3221225472) -Disposition ([uint32]4)
    if (-not $Opened) {
        return $null
    }
    $Handle = $Opened.Handle
    $LeasePath = $Opened.Path
    # Everything after the successful kernel open runs under a cleanup
    # guard: the open IS the acquisition, so a failure in initialization
    # would otherwise leave a kernel handle this host can never release -
    # the machine would stay excluded with no holder able to let go.
    $Stream = $null
    try {
        # Inheritable: every stage child launched while this lease is held
        # carries the exclusion for exactly as long as it can still act.
        if (-not [LeaseNative.Kernel]::SetHandleInformation($Handle, 0x1, 0x1)) {
            throw "the hold could not be marked inheritable."
        }
        $Identity = Get-MachineLeaseIdentity -Handle $Handle
        if (-not $Identity) {
            throw "the lease handle could not be identified."
        }
        $Safe = [Microsoft.Win32.SafeHandles.SafeFileHandle]::new($Handle, $false)
        $Stream = [System.IO.FileStream]::new($Safe, [System.IO.FileAccess]::ReadWrite)
        $Record = @{
            schema = 2
            purpose = $Purpose
            holder_pid = $PID
            acquired_at = (Get-Date).ToUniversalTime().ToString("o")
        } | ConvertTo-Json
        $Bytes = [Text.Encoding]::UTF8.GetBytes($Record)
        $Stream.SetLength(0)
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush()
    } catch {
        # The initialization failure is reported first; every cleanup step
        # then reports its own result. The handle is called closed exactly
        # when CloseHandle said so: a stream that would not dispose is a
        # cleanup error in its own right, not a retained exclusion.
        Write-Host "ERROR: the lease acquisition could not be initialized. $_"
        if ($Stream) {
            try {
                $Stream.Dispose()
            } catch {
                Write-Host "ERROR: stream cleanup error after the failed acquisition (the kernel handle is closed separately below). $_"
            }
        }
        # The lease handle and every pin are closed, each exactly once, and
        # every refused close is reported; only when all of them closed is
        # nothing held.
        if (Close-MachineLeaseAcquired -Handle $Handle -Pins $Opened.Binding.Pins -During "the failed acquisition") {
            Write-Host "ERROR: the lease handle was closed; nothing is held."
        } else {
            Write-Host "ERROR: this process is stranded after the failed acquisition; exit it to free what would not close."
        }
        return $null
    }

    return [pscustomobject]@{
        Handle = $Handle
        Identity = $Identity
        Stream = $Stream
        Path = $LeasePath
        Purpose = $Purpose
        Binding = $Opened.Binding
    }
}

function Exit-MachineLease {
    param([Parameter(Mandatory)]$Lease)

    # A close failure means the exclusion may still stand, and a release
    # that reports success over a live handle lets the gate print green
    # while the machine stays held - both failure modes carry into the
    # returned result, and each is named for what it is: a stream cleanup
    # error is not a retained exclusion. Only the record removal is
    # best-effort.
    $Ok = $true
    try {
        $Lease.Stream.Dispose()
    } catch {
        Write-Host "ERROR: the lease stream did not dispose cleanly (a stream cleanup error; the kernel handle is closed separately below). $_"
        $Ok = $false
    }
    # The lease handle and every pin are closed, each exactly once, and
    # every refused close is reported into the result.
    if (-not (Close-MachineLeaseAcquired -Handle $Lease.Handle -Pins $Lease.Binding.Pins -During "the release")) {
        $Ok = $false
    }
    if ($Ok) {
        try {
            Remove-Item -LiteralPath $Lease.Path -Force -ErrorAction Stop
        } catch {
            # A contender may already have re-acquired and be writing its own
            # record; the exclusion never depended on that removal.
        }
    }
    return $Ok
}

function Assert-MachineLeaseHeld {
    param([Parameter(Mandatory)]$Lease)

    # The kernel owns the exclusion, so the local failure mode is this
    # process discarding its own handle. The barrier asks the kernel: the
    # handle must still answer for the same object it was opened on - a
    # closed handle answers for nothing, and a reused handle value answers
    # for something else - and the stream must still be open beside it.
    if (-not $Lease -or -not $Lease.Stream -or -not $Lease.Stream.CanWrite) {
        throw (
            "The machine lease stream is no longer open in this process; " +
            "refusing to run further machine-exclusive stages."
        )
    }
    $Live = Get-MachineLeaseIdentity -Handle $Lease.Handle
    if (-not $Live -or $Live -ne $Lease.Identity) {
        throw (
            "The machine lease kernel handle is no longer held by this process; " +
            "refusing to run further machine-exclusive stages."
        )
    }
    # Every pin must still answer for the link it holds: a pin this process
    # let go is a hold that is not intact, whatever the pointer says now.
    foreach ($Pin in @($Lease.Binding.Pins)) {
        if (-not (Get-MachineLeaseIdentity -Handle $Pin.Handle)) {
            throw (
                "$($Pin.Role) is no longer held by this process ($($Pin.Path)); " +
                "refusing to run further machine-exclusive stages."
            )
        }
    }
    # The hold is on a directory object; the repository must still name it.
    # With the chain pinned this is a self-check of what the kernel already
    # forbids from changing; a mismatch means a link was let go.
    $Drift = Test-MachineLeaseBinding -Binding $Lease.Binding
    if ($Drift) {
        throw (
            "The repository moved under the machine lease ($Drift); " +
            "refusing to run further machine-exclusive stages."
        )
    }
}

function Invoke-LeasedStage {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        $Lease
    )

    Write-Host "==> $Label"
    if ($Lease) {
        Assert-MachineLeaseHeld -Lease $Lease
    }
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Complete-MachineLeaseRun {
    param(
        [Parameter(Mandatory)]$Lease,
        [Parameter(Mandatory)][bool]$BodyPassed,
        [string]$SuccessMessage = "",
        [string]$Epilogue = ""
    )

    # The success message is printed only after the release succeeded: a
    # green banner followed by a release error would claim a clean run the
    # exit code then contradicts.
    if (-not (Exit-MachineLease $Lease)) {
        Write-Host "ERROR: the machine lease did not release cleanly; run 'python scripts/machine_lock.py status'."
        return $false
    }
    if ($BodyPassed) {
        if ($SuccessMessage) {
            Write-Host $SuccessMessage
        }
        if ($Epilogue) {
            Write-Host $Epilogue
        }
    }
    return $true
}
