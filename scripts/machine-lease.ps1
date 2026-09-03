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
        $Saved[$Name] = [Environment]::GetEnvironmentVariable($Name)
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
                if ([Environment]::GetEnvironmentVariable($Name) -ne $Saved[$Name]) {
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
    try {
        return Get-MachineLeaseIdentity -Handle $Probe
    } finally {
        [LeaseNative.Kernel]::CloseHandle($Probe) | Out-Null
    }
}

function Get-MachineLeaseReparseLinks {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Role,
        [switch]$Itself
    )

    # Every reparse point among the components of a textual path, root
    # first - and the path itself when -Itself is given and it is one. Git
    # resolves the text at each invocation, so retargeting one of these
    # changes what the text names while the object at its end stays held;
    # each is pinned as itself. ".." components collapse lexically, as
    # Win32 collapses them before the kernel sees the name.
    $Full = [IO.Path]::GetFullPath($Path)
    $Prefixes = @()
    $Current = Split-Path -Parent $Full
    while ($Current -and (Split-Path -Parent $Current)) {
        $Prefixes = @($Current) + $Prefixes
        $Current = Split-Path -Parent $Current
    }
    if ($Itself) {
        $Prefixes += $Full
    }
    $Links = @()
    foreach ($Prefix in $Prefixes) {
        $Item = Get-Item -LiteralPath $Prefix -Force -ErrorAction Stop
        if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $Links += @{ Path = $Prefix; Role = "a link on the way to $Role" }
        }
    }
    return $Links
}

function Get-MachineLeaseNamedDirectory {
    param(
        [Parameter(Mandatory)][string]$Base,
        [Parameter(Mandatory)][string]$File
    )

    # The directory a commondir file names, as git reads it: its text,
    # relative to the directory holding the file.
    $Text = [IO.File]::ReadAllText($File).Trim()
    if ([IO.Path]::IsPathRooted($Text)) {
        return $Text
    }
    return (Join-Path $Base $Text)
}

function Get-MachineLeaseResolutionChain {
    param([Parameter(Mandatory)][string]$Anchor)

    # The links git follows from the repository directory to its common
    # directory, each a path whose change would change what the repository
    # names: the .git entry; for a pointer file, the private git directory
    # it names and that directory's commondir file; for a directory, its
    # commondir file if it has one; and every reparse point among the
    # components of the paths the pointer and the commondir file name. A
    # reparse point is a link as itself.
    $Entry = Join-Path $Anchor ".git"
    $Item = Get-Item -LiteralPath $Entry -Force -ErrorAction Stop
    $Links = @(@{ Path = $Entry; Role = "the repository's .git entry" })
    if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        return $Links
    }
    if ($Item.PSIsContainer) {
        $CommonDir = Join-Path $Entry "commondir"
        if (Test-Path -LiteralPath $CommonDir -PathType Leaf) {
            $Links += @{ Path = $CommonDir; Role = "the commondir file" }
            $Named = Get-MachineLeaseNamedDirectory -Base $Entry -File $CommonDir
            $Links += @(Get-MachineLeaseReparseLinks -Path $Named -Role "the common git directory" -Itself)
        }
        return $Links
    }
    $Text = [IO.File]::ReadAllText($Entry)
    if (-not $Text.StartsWith("gitdir:")) {
        throw "the .git file is not a git pointer: $Entry"
    }
    $Target = $Text.Substring(7).Trim()
    if (-not [IO.Path]::IsPathRooted($Target)) {
        $Target = Join-Path $Anchor $Target
    }
    $Links += @(Get-MachineLeaseReparseLinks -Path $Target -Role "the checkout's private git directory")
    $Links += @{ Path = $Target; Role = "the checkout's private git directory" }
    $CommonDir = Join-Path $Target "commondir"
    if (Test-Path -LiteralPath $CommonDir -PathType Leaf) {
        $Links += @{ Path = $CommonDir; Role = "the commondir file" }
        $Named = Get-MachineLeaseNamedDirectory -Base $Target -File $CommonDir
        $Links += @(Get-MachineLeaseReparseLinks -Path $Named -Role "the common git directory" -Itself)
    }
    return $Links
}

function Open-MachineLeasePin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Role
    )

    # Hold one link against change: read share only. A directory is held
    # with backup semantics, a reparse point as itself, a file for reading;
    # the kernel then refuses every other open that would write, rename,
    # delete or retarget the link.
    $Item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $Flags = [uint32]128
    if ($Item.PSIsContainer) { $Flags = [uint32]33554432 }
    if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) { $Flags = $Flags -bor [uint32]2097152 }
    # FILE_READ_ATTRIBUTES | GENERIC_READ, share read, OPEN_EXISTING.
    $Handle = [LeaseNative.Kernel]::CreateFileW(
        $Path, [uint32]2147483776, [uint32]1, [IntPtr]::Zero,
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
        if (-not (Close-MachineLeaseAcquired -Pins @([pscustomobject]@{ Path = $Path; Role = $Role; Handle = $Handle }) -During "a refused pinning")) {
            exit 4
        }
        throw "$Role could not be held for a child's lifetime: $Path (error $MarkError)"
    }
    return [pscustomobject]@{ Path = $Path; Role = $Role; Handle = $Handle }
}

function Close-MachineLeaseAcquired {
    param($Handle, $Pins, [Parameter(Mandatory)][string]$During)

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
    return $Ok
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
    try {
        if ((Get-MachineLeaseIdentity -Handle $Anchor) -ne $Binding.AnchorIdentity) {
            return "$AnchorPlain is another object"
        }
        $Resolved = Get-MachineLeaseCommonDir -RepositoryRoot $AnchorPlain
    } finally {
        [LeaseNative.Kernel]::CloseHandle($Anchor) | Out-Null
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

    # The repository directory is held as an OBJECT before anything is
    # resolved; the common directory is resolved through its held name and
    # held the same way; the repository is re-read to prove it did not move
    # while its name was used, and the resolution is repeated through it to
    # prove the held common directory is still what the repository names.
    # The lease is opened under the held name, the directory is read back
    # from its handle after the open, and the binding is re-verified: a
    # directory renamed or replaced under its name, or a pointer moved
    # elsewhere, is refused with the lease closed - and a close the kernel
    # refuses on that path strands this process, which then exits 4 rather
    # than report an ordinary refusal over a live hold.
    $Anchor = Open-MachineLeaseDirectory -Path $RepositoryRoot
    if (Test-MachineLeaseHandleInvalid $Anchor) {
        Write-Host "ERROR: the repository directory could not be held: $RepositoryRoot (error $script:MachineLeaseLastError)."
        return $null
    }
    try {
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
        try {
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
            # Every link between the repository and the held common
            # directory, and that directory itself, is now pinned; the
            # resolution is repeated once more through the pinned chain, so
            # a change slipped in before the pins took hold is refused and
            # nothing can change after them.
            $Pins = @()
            try {
                foreach ($Link in Get-MachineLeaseResolutionChain -Anchor $AnchorPlain) {
                    $Pins += Open-MachineLeasePin -Path $Link.Path -Role $Link.Role
                }
                $Pins += Open-MachineLeasePin -Path $Plain -Role "the common git directory"
            } catch {
                Write-Host "ERROR: the resolution chain could not be pinned. $_"
                if (-not (Close-MachineLeaseAcquired -Pins $Pins -During "a refused pinning")) {
                    exit 4
                }
                return $null
            }
            $Pinned = Get-MachineLeaseCommonDir -RepositoryRoot $AnchorPlain
            if (-not $Pinned -or (Get-MachineLeaseDirectoryIdentity -Path $Pinned) -ne $Identity) {
                Write-Host "ERROR: the repository's common git directory changed while it was being pinned: now $Pinned."
                if (-not (Close-MachineLeaseAcquired -Pins $Pins -During "a refused acquisition")) {
                    exit 4
                }
                return $null
            }
            if (-not (Test-Path -LiteralPath (Join-Path $Plain "HEAD") -PathType Leaf) -or
                -not (Test-Path -LiteralPath (Join-Path $Plain "config") -PathType Leaf)) {
                Write-Host "ERROR: the held common git directory is not a git dir: $Plain."
                if (-not (Close-MachineLeaseAcquired -Pins $Pins -During "a refused acquisition")) {
                    exit 4
                }
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
                if (-not (Close-MachineLeaseAcquired -Pins $Pins -During "a refused acquisition")) {
                    exit 4
                }
                return $null
            }
            $Bound = $false
            try {
                $Binding = [pscustomobject]@{
                    Anchor = $AnchorName
                    AnchorIdentity = $AnchorIdentity
                    Common = $Name
                    CommonIdentity = $Identity
                    Pins = $Pins
                }
                $Drift = Test-MachineLeaseBinding -Binding $Binding
                if ($Drift) {
                    Write-Host "ERROR: the repository's common git directory changed during acquisition: $Drift."
                    return $null
                }
                $Bound = $true
            } finally {
                if (-not $Bound) {
                    # The lease handle and every pin are closed, each exactly
                    # once, and every refused close is reported before the
                    # process exits stranded.
                    if (-not (Close-MachineLeaseAcquired -Handle $Handle -Pins $Pins -During "the refused acquisition")) {
                        exit 4
                    }
                }
            }
            return [pscustomobject]@{
                Handle = $Handle
                Path = $LeasePath
                Binding = $Binding
            }
        } finally {
            [LeaseNative.Kernel]::CloseHandle($Directory) | Out-Null
        }
    } finally {
        [LeaseNative.Kernel]::CloseHandle($Anchor) | Out-Null
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
