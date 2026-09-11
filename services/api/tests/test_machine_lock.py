"""The machine lease is only worth having if its refusals hold.

The module under test is loaded through runpy. The behaviour tests anchor
the lease in their own scratch git repository, because the location
contract is "beside the COMMON git directory, shared by every worktree, of
an explicitly identified repo"; the one wiring test reads the gate script
instead of running it.
"""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_NAMESPACE = runpy.run_path(str(ROOT / "scripts" / "machine_lock.py"))
LeaseRefused = _NAMESPACE["LeaseRefused"]

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="the machine lease holds Windows kernel handles"
)


@pytest.fixture
def anchor(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path / "repo")], check=True, capture_output=True)
    return tmp_path / "repo"


def _lease_file(anchor: Path) -> Path:
    return anchor / ".git" / _NAMESPACE["LEASE_BASENAME"]


def test_acquire_holds_and_release_frees(anchor: Path) -> None:
    lease = _NAMESPACE["acquire"]("first-holder", repo=anchor)
    try:
        with pytest.raises(LeaseRefused, match="first-holder"):
            _NAMESPACE["acquire"]("second", repo=anchor)
    finally:
        _NAMESPACE["release"](lease)
    successor = _NAMESPACE["acquire"]("second", repo=anchor)
    _NAMESPACE["release"](successor)


def test_the_refusal_names_the_recorded_holder(anchor: Path) -> None:
    lease = _NAMESPACE["acquire"]("named-purpose", repo=anchor)
    try:
        with pytest.raises(LeaseRefused) as caught:
            _NAMESPACE["acquire"]("contender", repo=anchor)
        assert "named-purpose" in str(caught.value)
        assert str(os.getpid()) in str(caught.value)
    finally:
        _NAMESPACE["release"](lease)


def test_reads_are_never_blocked_while_held(anchor: Path) -> None:
    lease = _NAMESPACE["acquire"]("reader-friendly", repo=anchor)
    try:
        text = _lease_file(anchor).read_text(encoding="utf-8")
        assert "reader-friendly" in text
        assert str(_NAMESPACE["status"](anchor)).startswith("held by pid")
    finally:
        _NAMESPACE["release"](lease)


def test_status_reports_freedom_after_release(anchor: Path) -> None:
    lease = _NAMESPACE["acquire"]("status-check", repo=anchor)
    _NAMESPACE["release"](lease)
    assert _NAMESPACE["status"](anchor) == "free"


def test_hold_lease_context_releases_on_every_exit(anchor: Path) -> None:
    hold_lease = _NAMESPACE["hold_lease"]
    with (
        hold_lease("context-green", repo=anchor),
        pytest.raises(LeaseRefused, match="context-green"),
    ):
        _NAMESPACE["acquire"]("contender", repo=anchor)
    follow = _NAMESPACE["acquire"]("after-green", repo=anchor)
    _NAMESPACE["release"](follow)

    with (
        pytest.raises(ValueError, match="the real failure"),
        hold_lease("context-red", repo=anchor),
    ):
        raise ValueError("the real failure")
    follow = _NAMESPACE["acquire"]("after-red", repo=anchor)
    _NAMESPACE["release"](follow)


def test_a_dead_holder_frees_instantly(anchor: Path) -> None:
    """The kernel frees the handle with its owner: a dead holder's
    exclusion ends the moment the process is gone, with no waiting
    period for a contender to sit out."""

    holder = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "machine_lock.py"),
            "--repo",
            str(anchor),
            "hold",
            "--purpose",
            "doomed",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "READY"
        with pytest.raises(LeaseRefused, match="doomed"):
            _NAMESPACE["acquire"]("contender", repo=anchor)
        holder.kill()
        holder.wait(timeout=30)
        deadline = time.monotonic() + 10
        while True:
            try:
                lease = _NAMESPACE["acquire"]("successor", repo=anchor)
                break
            except LeaseRefused:
                assert time.monotonic() < deadline, (
                    "the machine stayed held after its only holder died"
                )
                time.sleep(0.1)
        _NAMESPACE["release"](lease)
    finally:
        with suppress(Exception):
            holder.kill()


_PARENT_THAT_DIES = """
import os
import pathlib
import runpy
import subprocess
import sys

namespace = runpy.run_path(sys.argv[3])
lease = namespace["acquire"]("parent-that-dies", repo=pathlib.Path(sys.argv[1]))
child = subprocess.Popen(
    [sys.executable, "-c", "import sys, time; time.sleep(120)"],
    close_fds=False,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
pathlib.Path(sys.argv[2]).write_text(str(child.pid), encoding="utf-8")
# Die WITHOUT releasing: the child inherited the handle and carries the
# exclusion for as long as it can still act.
os._exit(0)
"""


def test_a_child_extends_the_exclusion_beyond_its_parent(anchor: Path, tmp_path: Path) -> None:
    """The scenario the exclusion exists for: the holder dies mid-stage.

    The parent acquires, launches a child that inherits the handle, and
    dies without releasing. While that child can still act, a contender's
    acquire must refuse; the machine frees only when the child is gone.
    """

    script = tmp_path / "parent_that_dies.py"
    script.write_text(_PARENT_THAT_DIES, encoding="utf-8")
    pid_file = tmp_path / "child_pid.txt"
    # No pipes: with handle inheritance on, a pipe handed to the parent
    # would flow to the sleeper too and outlive this call.
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(anchor),
            str(pid_file),
            str(ROOT / "scripts" / "machine_lock.py"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        with pytest.raises(LeaseRefused, match="parent-that-dies"):
            _NAMESPACE["acquire"]("contender", repo=anchor)
        subprocess.run(
            ["taskkill", "/PID", str(child_pid), "/F"],
            capture_output=True,
            check=False,
        )
        deadline = time.monotonic() + 15
        while True:
            try:
                lease = _NAMESPACE["acquire"]("successor", repo=anchor)
                break
            except LeaseRefused:
                assert time.monotonic() < deadline, (
                    "the machine stayed held after the whole holder tree died"
                )
                time.sleep(0.2)
        _NAMESPACE["release"](lease)
    finally:
        subprocess.run(
            ["taskkill", "/PID", str(child_pid), "/F"],
            capture_output=True,
            check=False,
        )


def test_platform_refusal_precedes_any_windows_call(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolving = _NAMESPACE["acquire"].__globals__
    monkeypatch.setitem(resolving["os"].__dict__, "name", "posix")
    try:
        with pytest.raises(LeaseRefused, match="Windows only"):
            _NAMESPACE["acquire"]("nowhere", repo=anchor)
    finally:
        monkeypatch.undo()


def test_repo_identity_is_validated_not_trusted(anchor: Path, tmp_path: Path) -> None:
    bogus = tmp_path / "not-a-repo"
    bogus.mkdir()
    # A broken gitdir pointer makes this directory unresolvable WHEREVER the
    # test's scratch space lives: a bare non-repository directory would
    # resolve an ancestor repository when the scratch space itself sits
    # below one.
    (bogus / ".git").write_text("gitdir: does-not-exist\n", encoding="utf-8")
    with pytest.raises(LeaseRefused, match="does not resolve"):
        _NAMESPACE["acquire"]("nowhere", repo=bogus)
    with pytest.raises(LeaseRefused, match="absolute"):
        _NAMESPACE["lease_path"](Path("relative/path"))

    # A resolvable answer is still validated: a common dir that has lost its
    # config is corruption, not a place to put the machine's lease.
    (anchor / ".git" / "config").unlink()
    with pytest.raises(LeaseRefused, match="not a git dir"):
        _NAMESPACE["lease_path"](anchor)


def test_git_redirection_environment_is_scrubbed(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "init", "-q", str(decoy)], check=True, capture_output=True)
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    assert _NAMESPACE["lease_path"](anchor) == _lease_file(anchor)


def test_purpose_and_pid_are_validated(anchor: Path) -> None:
    with pytest.raises(LeaseRefused, match="purpose"):
        _NAMESPACE["acquire"]("", repo=anchor)
    with pytest.raises(LeaseRefused, match="holder_pid"):
        _NAMESPACE["acquire"]("typed", repo=anchor, holder_pid=True)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_round_trip_holds_and_releases(anchor: Path, tmp_path: Path) -> None:
    """Enter, exclude a contender, run a leased stage, Exit, and free."""

    contender = tmp_path / "contender.py"
    contender.write_text(
        "import pathlib\n"
        "import runpy\n"
        "import sys\n"
        "namespace = runpy.run_path(sys.argv[1])\n"
        "try:\n"
        '    namespace["acquire"]("contender", repo=pathlib.Path(sys.argv[2]))\n'
        "except namespace['LeaseRefused'] as exc:\n"
        "    print(f'REFUSED:{exc}')\n"
        "    raise SystemExit(0)\n"
        "print('TOOK-THE-MACHINE')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "shell-round-trip"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
& "{sys.executable}" @("{contender}", "{ROOT / "scripts" / "machine_lock.py"}", "{anchor}")
if ($LASTEXITCODE -ne 0) {{ Write-Output "CONTENDER-WON"; exit 4 }}
$Second = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "double-enter"
if ($Second) {{ Write-Output "DOUBLE-ENTER"; exit 6 }}
try {{
    Invoke-LeasedStage -Label "leased stage" -FilePath "{sys.executable}" `
        -ArgumentList @("-c", "print('stage ran')") -Lease $Lease
    Write-Output "STAGE-GREEN"
}} catch {{
    Write-Output "STAGE-REFUSED"
    exit 3
}}
if (Exit-MachineLease $Lease) {{ Write-Output "RELEASED" }}
else {{ Write-Output "RELEASE-FAILED"; exit 5 }}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REFUSED:" in result.stdout, "the contender was not excluded"
    assert "STAGE-GREEN" in result.stdout
    assert "RELEASED" in result.stdout
    lease = _NAMESPACE["acquire"]("after-shell", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_a_shell_childs_inherited_handle_outlives_its_parent(anchor: Path, tmp_path: Path) -> None:
    """The shell hold is inheritable: a child outlives the parent's release.

    Enter, launch a detached child that inherits the handle and sleeps, then
    Exit - releasing the PARENT's handle. While the child is alive a rival
    must still be refused; only when the child dies does the machine free.
    Without the inherit flag the child holds nothing and the rival takes the
    machine the instant Exit returns.
    """

    pid_file = tmp_path / "child_pid.txt"
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "inherit-child"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
$StartInfo = New-Object System.Diagnostics.ProcessStartInfo
$StartInfo.FileName = "{sys.executable}"
$StartInfo.Arguments = '-c "import sys, time; time.sleep(120)"'
$StartInfo.UseShellExecute = $false
$StartInfo.RedirectStandardInput = $true
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
$StartInfo.CreateNoWindow = $true
$Child = [System.Diagnostics.Process]::Start($StartInfo)
Set-Content -LiteralPath "{pid_file}" -Value $Child.Id
Exit-MachineLease $Lease | Out-Null
Set-Content -LiteralPath "{tmp_path / "released.txt"}" -Value "PARENT-RELEASED"
"""
    # The PowerShell runs with no pipes at all: any pipe handed to it would
    # be inherited by the sleeper and outlive this call.
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0
    assert (tmp_path / "released.txt").exists(), "the parent never released"
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        with pytest.raises(LeaseRefused, match="inherit-child"):
            _NAMESPACE["acquire"]("rival", repo=anchor)
        subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], capture_output=True, check=False)
        deadline = time.monotonic() + 15
        while True:
            try:
                lease = _NAMESPACE["acquire"]("successor", repo=anchor)
                break
            except LeaseRefused:
                assert time.monotonic() < deadline, (
                    "the machine stayed held after the whole holder tree died"
                )
                time.sleep(0.2)
        _NAMESPACE["release"](lease)
    finally:
        subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], capture_output=True, check=False)


def test_a_stage_child_keeps_the_machine_excluded(anchor: Path, tmp_path: Path) -> None:
    """Non-overlap, asserted while a real stage child is RUNNING.

    The stage child inherited the handle at launch, so even the moment a
    rival probes mid-stage, its write-access open fails at the kernel.
    """

    child = tmp_path / "stage_child.py"
    child.write_text(
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        "base = pathlib.Path(sys.argv[1])\n"
        'base.joinpath("running.txt").write_text("r", encoding="utf-8")\n'
        "deadline = time.time() + 60\n"
        'while not base.joinpath("checked.txt").exists():\n'
        "    if time.time() > deadline:\n"
        "        raise SystemExit(5)\n"
        "    time.sleep(0.2)\n",
        encoding="utf-8",
    )
    running = tmp_path / "running.txt"
    checked = tmp_path / "checked.txt"
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "mid-stage"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
try {{
    Invoke-LeasedStage -Label "long stage" -FilePath "{sys.executable}" `
        -ArgumentList @("{child}", "{tmp_path}") -Lease $Lease
    Write-Output "STAGE-GREEN"
}} catch {{
    Write-Output "STAGE-REFUSED"
    exit 3
}}
Exit-MachineLease $Lease | Out-Null
"""
    process = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not running.exists():
            assert time.monotonic() < deadline, "the stage child never started"
            assert process.poll() is None, process.communicate()[0]
            time.sleep(0.2)
        with pytest.raises(LeaseRefused, match="mid-stage"):
            _NAMESPACE["acquire"]("rival", repo=anchor)
        checked.write_text("checked", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=120)
    finally:
        with suppress(Exception):
            process.kill()
    assert process.returncode == 0, stdout + stderr
    assert "STAGE-GREEN" in stdout
    lease = _NAMESPACE["acquire"]("after-stage", repo=anchor)
    _NAMESPACE["release"](lease)


def test_the_gate_wiring_holds_the_handle_for_its_whole_run() -> None:
    """The gate takes the lease before its first stage, releases it inside
    its finally block, and prints its success banner only through the
    completion helper that runs after the release."""

    source = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")
    enter_at = source.index("$MachineLease = Enter-MachineLease")
    assert "machine-lease.ps1" in source[:enter_at], (
        "the helper must be dot-sourced before it is called"
    )
    first_stage_at = source.index('Invoke-Checked "')
    identity_stage_at = source.index('"==> Import identity"')
    assert enter_at < first_stage_at and enter_at < identity_stage_at, (
        "the lease is taken after the first gate stage, so the stage it "
        "exists to protect runs unprotected"
    )
    finally_at = source.rindex("finally {")
    finally_block_end = source.index("\n}", finally_at)
    assert "$LeaseReleaseFailed" in source and "exit 2" in source[finally_block_end:], (
        "a failed release must PROPAGATE after a green body, not be reset away"
    )
    complete_at = source.rindex("Complete-MachineLeaseRun")
    assert finally_at < complete_at < finally_block_end, (
        "the success banner is printed by the completion helper inside the finally block"
    )
    assert 'Write-Host "All LM Atelier' not in source, (
        "the body must not print success before the release"
    )
    assert "$GateBodyPassed = $true" in source[:finally_at], (
        "the body records completion; the completion helper decides what to print"
    )
    assert "$global:LASTEXITCODE = 0" not in source, "a masking reset would hide a failed release"
    assert "Invoke-LeasedStage -Label $Label" in source, (
        "Invoke-Checked must delegate to the leased-stage runner"
    )
    helper_source = (ROOT / "scripts" / "machine-lease.ps1").read_text(encoding="utf-8")
    assert "Stop-Process" not in helper_source, (
        "termination must never target a bare pid a stranger may have reused"
    )
    assert "SetHandleInformation" in helper_source, (
        "the hold must be marked inheritable or stage children stop extending it"
    )
    stage_fn_at = helper_source.index("function Invoke-LeasedStage")
    stage_run_at = helper_source.index("& $FilePath @ArgumentList", stage_fn_at)
    pre_guard_at = helper_source.index("Assert-MachineLeaseHeld -Lease $Lease", stage_fn_at)
    assert pre_guard_at < stage_run_at, (
        "the leased-stage runner must sanity-check the handle before its child"
    )
    for banner, name in (
        ('"==> Import identity"', "import identity"),
        ('"==> Windows packaging syntax"', "windows packaging"),
    ):
        banner_at = source.index(banner)
        guard_at = source.rfind("Assert-MachineLeaseHeld", 0, banner_at)
        assert guard_at != -1 and banner_at - guard_at < 400, (
            f"the {name} direct stage runs without the handle sanity check"
        )


def test_a_failed_initialization_frees_the_kernel_handle(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial acquisition must not strand the exclusion.

    The kernel open succeeds and initialization then fails. Without the
    cleanup guard the handle stays open in this process with no lease
    object anyone can release - in a long-lived host the machine would be
    excluded until the host dies. The guard closes the handle on the way
    out, so the failure is loud AND the very next acquire succeeds.
    """

    def boom_write(fd: int, data: bytes) -> int:
        raise OSError("the record write failed after the descriptor took the handle")

    # Fail at the record write: by then a real descriptor owns the kernel
    # handle, so without the cleanup guard the descriptor leaks and the
    # machine stays held in this same process - a successor is refused. The
    # guard closes the descriptor, so the failure is loud AND the very next
    # acquire succeeds.
    monkeypatch.setattr(os, "write", boom_write)
    with pytest.raises(OSError, match="the record write failed"):
        _NAMESPACE["acquire"]("doomed", repo=anchor)
    monkeypatch.undo()

    assert "free" in _NAMESPACE["status"](anchor), "the failed initialization stranded the hold"
    lease = _NAMESPACE["acquire"]("successor", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_failed_close_is_loud(anchor: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """release must not report success over a handle that did not close.

    A close the kernel refuses may leave the exclusion standing; suppressing
    it lets the gate print green while the machine stays held. The failure
    propagates from release and out of the hold_lease context alike, as a
    strand that keeps the kernel's refusal as its cause - and the pins are
    still closed behind it rather than abandoned with the exclusion.
    """

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_close = os.close

    def refusing(descriptor: int) -> None:
        raise OSError("the kernel refused the close")

    lease = _NAMESPACE["acquire"]("loud-close", repo=anchor)
    held_descriptor = lease.descriptor
    pins = lease.binding.pins if lease.binding is not None else ()
    monkeypatch.setattr(os, "close", refusing)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="may still hold") as caught:
        _NAMESPACE["release"](lease)
    monkeypatch.undo()
    real_close(held_descriptor)
    assert caught.value.kind == "descriptor" and caught.value.number == held_descriptor
    assert isinstance(caught.value.__cause__, OSError), "the kernel's refusal was lost"
    assert "refused the close" in str(caught.value.__cause__)
    # The pins were still closed behind the refused descriptor, each once.
    for pin in pins:
        assert not kernel32.GetFileInformationByHandle(
            ctypes.c_void_p(pin.handle), ctypes.byref(_NAMESPACE["_ByHandleFileInformation"]())
        ), "a pin stayed open behind the refused descriptor close"

    captured = -1
    with (
        pytest.raises(_NAMESPACE["LeaseStranded"], match="may still hold"),
        _NAMESPACE["hold_lease"]("loud-context", repo=anchor) as held,
    ):
        captured = held.descriptor
        monkeypatch.setattr(os, "close", refusing)
    monkeypatch.undo()
    real_close(captured)

    successor = _NAMESPACE["acquire"]("after-loud", repo=anchor)
    _NAMESPACE["release"](successor)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_exit_reports_a_failed_close(anchor: Path) -> None:
    """Exit-MachineLease carries a close failure into its result.

    A real Enter/Exit releases cleanly and answers True. A lease-shaped
    object carrying a handle the kernel rejects - value 3, never a multiple
    of four and so never a real handle - makes CloseHandle answer FALSE,
    and the returned result must carry that: an unconditional true would
    leave the gate's release-failure branch unreachable no matter what the
    kernel reported.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "exit-report"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
$Clean = Exit-MachineLease $Lease
$Bad = [pscustomobject]@{{
    Handle = [IntPtr]::new(3)
    Stream = [System.IO.MemoryStream]::new()
    Path = "{anchor}\\.git\\never-written"
    Purpose = "bad-handle"
}}
$Failed = Exit-MachineLease $Bad
Write-Output "CLEAN:$Clean"
Write-Output "FAILED:$Failed"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLEAN:True" in result.stdout, result.stdout
    assert "FAILED:False" in result.stdout, (
        "a close the kernel refused was reported as a clean release"
    )


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_a_shell_partial_initialization_frees_the_handle(anchor: Path) -> None:
    """The shell acquisition cleans up after its own initialization failure.

    The record build is made to fail after the kernel open by shadowing the
    JSON cmdlet with a throwing function; Enter must answer null with the
    handle closed, and the very next Enter in the same host must succeed -
    a leaked handle would exclude the machine for the life of the host.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
function ConvertTo-Json {{ throw "record build failed" }}
$First = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "doomed"
if ($First) {{ Write-Output "FIRST-ACQUIRED"; exit 2 }}
Write-Output "FIRST-NULL"
Remove-Item function:ConvertTo-Json
$Second = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "survivor"
if (-not $Second) {{ Write-Output "SECOND-FAILED"; exit 3 }}
Write-Output "SECOND-OK"
if (Exit-MachineLease $Second) {{ Write-Output "SECOND-RELEASED" }}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIRST-NULL" in result.stdout
    assert "SECOND-OK" in result.stdout, (
        "the failed initialization left the machine excluded in the same host"
    )
    assert "SECOND-RELEASED" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_leased_stage_child_keeps_the_machine_after_parent_death(
    anchor: Path, tmp_path: Path
) -> None:
    """The launcher's own child carries the exclusion past the parent's death.

    The holder Enters and launches its stage child through the
    Invoke-LeasedStage call-operator path, then the HOLDER is killed while
    that child runs. The child's inherited handle must keep a rival refused
    for as long as the child can still act, and the machine frees once the
    child dies.
    """

    pid_file = tmp_path / "stage_pid.txt"
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "launcher-parent"
if (-not $Lease) {{ exit 1 }}
Invoke-LeasedStage -Label "sleeping stage" -FilePath "{sys.executable}" `
    -ArgumentList @("{sleeper}", "{pid_file}") -Lease $Lease
"""
    # No pipes: with handle inheritance on, a pipe handed to the holder
    # would flow to the stage child and outlive this call.
    holder = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stage_pid = -1
    try:
        deadline = time.monotonic() + 60

        def holder_alive() -> None:
            assert holder.poll() is None, "the holder died before its stage ran"

        stage_pid = int(_settled_text(pid_file, deadline=deadline, still_running=holder_alive))

        holder.kill()
        holder.wait(timeout=30)

        with pytest.raises(LeaseRefused, match="launcher-parent"):
            _NAMESPACE["acquire"]("rival", repo=anchor)
        time.sleep(1.0)
        with pytest.raises(LeaseRefused):
            _NAMESPACE["acquire"]("rival-again", repo=anchor)
    finally:
        if holder.poll() is None:
            holder.kill()
        if stage_pid != -1:
            subprocess.run(
                ["taskkill", "/PID", str(stage_pid), "/F"],
                capture_output=True,
                check=False,
            )

    deadline = time.monotonic() + 15
    while True:
        try:
            lease = _NAMESPACE["acquire"]("successor", repo=anchor)
            break
        except LeaseRefused:
            assert time.monotonic() < deadline, "the machine stayed held after the stage child died"
            time.sleep(0.2)
    _NAMESPACE["release"](lease)


def test_a_refused_descriptor_close_after_a_failed_acquisition_is_reported(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both failures surface: the acquisition's and the cleanup's.

    Initialization fails after the descriptor owns the handle and the
    descriptor then refuses to close. The caller must learn that the
    machine may still be held by this process, with the original failure
    kept as the cause - a silent cleanup would report a clean refusal over a
    stranded exclusion.
    """

    real_close = os.close
    held: dict[str, int] = {}

    def boom_write(descriptor: int, data: bytes) -> int:
        held["descriptor"] = descriptor
        raise OSError("the record write failed")

    def refusing_close(descriptor: int) -> None:
        raise OSError("the kernel refused the close")

    monkeypatch.setattr(os, "write", boom_write)
    monkeypatch.setattr(os, "close", refusing_close)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="may still hold") as caught:
        _NAMESPACE["acquire"]("doomed", repo=anchor)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, OSError), "the original failure was lost"
    assert "record write failed" in str(caught.value.__cause__)

    # The exclusion IS stranded until the descriptor really closes: a
    # contender must be refused, which is exactly what the report warned.
    with pytest.raises(LeaseRefused):
        _NAMESPACE["acquire"]("contender", repo=anchor)
    real_close(held["descriptor"])
    successor = _NAMESPACE["acquire"]("successor", repo=anchor)
    _NAMESPACE["release"](successor)


def test_a_refused_raw_handle_close_after_a_failed_acquisition_is_reported(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw-handle branch of the cleanup reports a refused CloseHandle.

    Initialization fails before any descriptor exists - the lease handle's
    inherit mark is refused, which is the first step after the kernel open -
    so the raw kernel handle must be closed directly; when that close is
    refused too, the caller learns it with the original refusal kept as the
    cause, and the pins, closed behind it, are not in the strand.
    """

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create_file = kernel32.CreateFileW
    real_mark = kernel32.SetHandleInformation
    real_close_handle = kernel32.CloseHandle
    taken: dict[str, int] = {}

    def create_file_noting_the_lease(name: str, *rest: object) -> object:
        handle = real_create_file(name, *rest)
        if str(name).endswith("machine-exclusive.lease") and "handle" not in taken:
            taken["handle"] = int(handle)
        return handle

    def unmarkable(handle: object, mask: int, flags: int) -> int:
        # The pins are marked inheritable before the lease is opened; only
        # the exclusion handle's mark is refused.
        if int(getattr(handle, "value", handle)) == taken.get("handle"):  # type: ignore[arg-type]
            return 0
        return int(real_mark(handle, mask, flags))

    def refusing_close_handle(handle: object) -> int:
        # Only the exclusion handle is refused; the pins and the held
        # directory handles close as they would in the kernel.
        if int(getattr(handle, "value", handle)) == taken.get("handle"):  # type: ignore[arg-type]
            return 0
        return int(real_close_handle(handle))

    monkeypatch.setattr(kernel32, "CreateFileW", create_file_noting_the_lease)
    monkeypatch.setattr(kernel32, "SetHandleInformation", unmarkable)
    monkeypatch.setattr(kernel32, "CloseHandle", refusing_close_handle)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="may still hold") as caught:
        _NAMESPACE["acquire"]("doomed", repo=anchor)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, LeaseRefused), "the original refusal was lost"
    assert caught.value.kind == "handle"
    assert caught.value.number == taken["handle"], "the message must name the live handle"
    assert caught.value.strands == (("handle", taken["handle"], caught.value.error),), (
        "the pins closed behind the refused handle must not be reported as strands"
    )

    assert real_close_handle(ctypes.c_void_p(taken["handle"])), "the probe handle was already gone"
    successor = _NAMESPACE["acquire"]("successor", repo=anchor)
    _NAMESPACE["release"](successor)


def test_a_status_probe_whose_close_is_refused_never_answers_free(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status takes a write handle to ask; if it cannot let go it is the holder."""

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create = kernel32.CreateFileW
    real_close_handle = kernel32.CloseHandle
    probes: list[int] = []

    def watching_create(path: str, access: int, *rest: object) -> object:
        handle = real_create(path, access, *rest)
        if access == _NAMESPACE["_GENERIC_WRITE"] and path.endswith(_NAMESPACE["LEASE_BASENAME"]):
            probes.append(int(handle))
        return handle

    def refusing_probe_close(handle: object) -> int:
        if int(getattr(handle, "value", handle)) in probes:  # type: ignore[arg-type]
            return 0
        return int(real_close_handle(handle))

    # A record must exist for the probe to open; release tidies it away, so
    # a stale one is left behind deliberately.
    _lease_file(anchor).write_text("stale record", encoding="utf-8")
    monkeypatch.setattr(kernel32, "CreateFileW", watching_create)
    monkeypatch.setattr(kernel32, "CloseHandle", refusing_probe_close)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="may still hold") as caught:
        _NAMESPACE["status"](anchor)
    monkeypatch.undo()
    assert probes, "the status probe never opened a handle"
    assert caught.value.kind == "probe"
    assert caught.value.number == probes[-1], "the message must name the probe handle"
    assert caught.value.__cause__ is None, "a probe strand has no acquisition behind it"
    assert real_close_handle(ctypes.c_void_p(probes[-1])), "the probe handle was already gone"
    assert "free" in _NAMESPACE["status"](anchor)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_exit_reports_a_stream_that_will_not_dispose(anchor: Path) -> None:
    """Exit-MachineLease carries a Dispose failure into its result.

    The lease's stream is replaced by an object whose Dispose throws; the
    handle itself still closes, and the result must still be false -
    a disposal refusal is a close problem, not a footnote.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "dispose-refused"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
$Lease.Stream.Dispose()
$Refusing = New-Object PSObject
$Refusing | Add-Member -MemberType ScriptMethod -Name Dispose -Value {{ throw "dispose refused" }}
$Refusing | Add-Member -MemberType NoteProperty -Name CanWrite -Value $true
$Lease.Stream = $Refusing
$Result = Exit-MachineLease $Lease
Write-Output "RESULT:$Result"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT:False" in result.stdout, (
        "a stream that would not dispose was reported as a clean release"
    )
    assert "did not dispose" in result.stdout
    lease = _NAMESPACE["acquire"]("after-refusal", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_a_shell_acquisition_that_fails_twice_never_claims_the_handle_closed(
    anchor: Path,
) -> None:
    """Enter reports a refused cleanup close instead of announcing a close.

    The record build is made to fail, and the shadowing function - a child
    scope of Enter-MachineLease that can read its $Handle - first marks that
    handle protect-from-close, so Enter's own CloseHandle is refused. (Closing
    it from the shadow instead would let the kernel reuse the number before
    Enter's close, which then succeeds against a stranger's handle.) Enter
    must answer null AND say the handle was NOT closed; the text that claims
    a close must never appear. The host exits afterwards, so the protected
    handle dies with it and the machine frees.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
function ConvertTo-Json {{
    [LeaseNative.Kernel]::SetHandleInformation($Handle, 2, 2) | Out-Null
    throw "record build failed"
}}
$First = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "doomed-twice"
if ($First) {{ Write-Output "FIRST-ACQUIRED"; exit 2 }}
Write-Output "FIRST-NULL"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIRST-NULL" in result.stdout
    assert "NOT closed" in result.stdout, (
        "a refused cleanup close was not reported as a possible stranded hold"
    )
    assert "the handle was closed" not in result.stdout, (
        "Enter claimed a close that CloseHandle refused"
    )
    assert result.stdout.index("could not be initialized") < result.stdout.index(
        "CloseHandle refused"
    ), "the initialization failure must be reported before the cleanup result"
    lease = _NAMESPACE["acquire"]("after-double-failure", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_stranded_probe_reports_its_own_exit_code(
    anchor: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A host that ran the CLI must learn it may now be the holder: stranded
    is neither success nor ordinary contention, and the message tells the
    host to exit."""

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create = kernel32.CreateFileW
    real_close_handle = kernel32.CloseHandle
    probes: list[int] = []

    def watching_create(path: str, access: int, *rest: object) -> object:
        handle = real_create(path, access, *rest)
        if access == _NAMESPACE["_GENERIC_WRITE"] and path.endswith(_NAMESPACE["LEASE_BASENAME"]):
            probes.append(int(handle))
        return handle

    def refusing_probe_close(handle: object) -> int:
        if int(getattr(handle, "value", handle)) in probes:  # type: ignore[arg-type]
            return 0
        return int(real_close_handle(handle))

    _lease_file(anchor).write_text("stale record", encoding="utf-8")
    monkeypatch.setattr(kernel32, "CreateFileW", watching_create)
    monkeypatch.setattr(kernel32, "CloseHandle", refusing_probe_close)
    code = _NAMESPACE["main"](["--repo", str(anchor), "status"])
    monkeypatch.undo()

    assert code == _NAMESPACE["_STRANDED"] == 4
    assert code != _NAMESPACE["_REFUSED"]
    captured = capsys.readouterr()
    assert "exit this process" in captured.err
    assert str(probes[-1]) in captured.err
    assert real_close_handle(ctypes.c_void_p(probes[-1]))


def test_a_stale_error_copy_never_masks_the_kernels_answer(anchor: Path) -> None:
    """The refusal reports the error of the open that just failed, not a
    copy some earlier call left behind."""

    import ctypes

    holder = _NAMESPACE["acquire"]("holder", repo=anchor)
    try:
        ctypes.set_last_error(999)
        with pytest.raises(LeaseRefused) as caught:
            _NAMESPACE["acquire"]("contender", repo=anchor)
    finally:
        _NAMESPACE["release"](holder)
    assert str(caught.value).startswith("contended:"), str(caught.value)
    assert "(error 999)" not in str(caught.value)


def _swap_directory_under_the_name(held: Path, moved: Path) -> None:
    """Rename the held git directory away and put a copy under its old name."""

    import shutil

    held.rename(moved)
    shutil.copytree(moved, held)


def _act_on_common_name(
    module: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    occurrence: int,
    action: Callable[[], None],
) -> list[str]:
    """Run ``action`` right after the held common directory's name is read
    for the ``occurrence``-th time: once is the hold itself, before the
    repository is re-read and the lease opened; twice is the read-back
    after the lease open, before the binding is re-verified."""

    real_final_path = module["_final_path"]
    seen: list[str] = []

    def final_path(handle: int) -> str:
        name = real_final_path(handle)  # type: ignore[operator]
        if name.endswith(".git"):
            seen.append(name)
            if len(seen) == occurrence:
                action()
        return name

    monkeypatch.setitem(module, "_final_path", final_path)
    return seen


def _restore_swapped(git_dir: Path, moved: Path) -> None:
    import shutil

    shutil.rmtree(git_dir)
    moved.rename(git_dir)


def test_the_common_directory_is_held_from_resolution_through_the_open(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git directory replaced under its name after the hold and before the
    file open is refused: the repository, re-resolved through its held
    directory, no longer names the held object, so no lease is opened under
    the impostor at all. With the impostor gone, the repository leases
    normally."""

    module = _NAMESPACE["acquire"].__globals__
    git_dir = anchor / ".git"
    moved = anchor / ".git-moved"
    seen = _act_on_common_name(
        module, monkeypatch, 1, lambda: _swap_directory_under_the_name(git_dir, moved)
    )
    with pytest.raises(LeaseRefused, match="changed while it was being held"):
        _NAMESPACE["acquire"]("split", repo=anchor)
    monkeypatch.undo()
    assert seen, "the seam never ran"
    assert not (git_dir / "machine-exclusive.lease").exists(), (
        "a lease was opened under the impostor"
    )
    _restore_swapped(git_dir, moved)

    lease = _NAMESPACE["acquire"]("after-move", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_directory_replaced_after_the_re_verification_is_refused_before_the_open(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git directory replaced under its name after the repository was
    re-verified and before the chain was pinned is caught by the resolution
    repeated through the pinned chain: refused before any lease is opened,
    with no lease under the impostor. Once the pins hold, no such
    replacement is possible at all. With the impostor gone, the repository
    leases normally."""

    module = _NAMESPACE["acquire"].__globals__
    real_directory_identity = module["_directory_identity"]
    git_dir = anchor / ".git"
    moved = anchor / ".git-moved"
    swapped: list[Path] = []

    def identity_then_swap(path: Path) -> tuple[int, int, int]:
        identity = real_directory_identity(path)  # type: ignore[operator]
        if not swapped:
            swapped.append(path)
            _swap_directory_under_the_name(git_dir, moved)
        return identity

    monkeypatch.setitem(module, "_directory_identity", identity_then_swap)
    with pytest.raises(LeaseRefused, match="changed while it was being pinned"):
        _NAMESPACE["acquire"]("split-late", repo=anchor)
    monkeypatch.undo()
    assert swapped, "the seam never ran"
    assert not (git_dir / "machine-exclusive.lease").exists(), (
        "a lease was opened under the impostor"
    )
    assert not (moved / "machine-exclusive.lease").exists()
    _restore_swapped(git_dir, moved)

    lease = _NAMESPACE["acquire"]("after-late-move", repo=anchor)
    _NAMESPACE["release"](lease)


def _linked_worktree(anchor: Path, tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    """A linked worktree of ``anchor`` and a second repository: the linked
    checkout, the other repository, the pointer file and its original
    bytes."""

    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(anchor), "commit", "-q", "--allow-empty", "-m", "seed"],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        },
    )
    subprocess.run(
        ["git", "-C", str(anchor), "worktree", "add", "-q", str(linked)],
        check=True,
        capture_output=True,
    )
    other = tmp_path / "other"
    subprocess.run(["git", "init", "-q", str(other)], check=True, capture_output=True)
    pointer = linked / ".git"
    return linked, other, pointer, pointer.read_bytes()


def _redirect(other: Path) -> bytes:
    return f"gitdir: {(other / '.git').as_posix()}\n".encode()


def test_a_held_lease_pins_the_checkout_pointer(anchor: Path, tmp_path: Path) -> None:
    """While a lease is held through a linked checkout, its pointer cannot be
    rewritten - by this process or by another one: the pin shares reads
    only, so the kernel refuses the write. The binding stays what it was,
    no second holder can lease through the checkout, and the pointer moves
    again only once the lease is released."""

    import subprocess
    import sys

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    first = _NAMESPACE["acquire"]("first", repo=linked)
    try:
        first.assert_bound()
        with pytest.raises(PermissionError):
            _repoint(pointer, _redirect(other))
        elsewhere = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pathlib, sys; pathlib.Path(sys.argv[1]).open('r+b').close()",
                str(pointer),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert elsewhere.returncode != 0 and "PermissionError" in elsewhere.stderr
        assert pointer.read_bytes() == original_pointer
        first.assert_bound()
        with pytest.raises(LeaseRefused, match="contended"):
            _NAMESPACE["acquire"]("second", repo=linked)
    finally:
        _NAMESPACE["release"](first)
    _repoint(pointer, _redirect(other))
    second = _NAMESPACE["acquire"]("second", repo=linked)
    assert Path(second.path).parent == (other / ".git").resolve()
    _NAMESPACE["release"](second)
    _repoint(pointer, original_pointer)

    lease = _NAMESPACE["acquire"]("after-pin", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_live_stage_child_cannot_be_redirected(anchor: Path, tmp_path: Path) -> None:
    """A stage child runs while its parent holds the lease; the parent's
    pins outlive the child, so the checkout cannot be redirected while the
    child can act, and no second holder can lease through it meanwhile."""

    import subprocess
    import sys

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    lease = _NAMESPACE["acquire"]("stage", repo=linked)
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            close_fds=False,
        )
        try:
            with pytest.raises(PermissionError):
                _repoint(pointer, _redirect(other))
            with pytest.raises(LeaseRefused, match="contended"):
                _NAMESPACE["acquire"]("second", repo=linked)
            assert child.poll() is None, "the stage ended before the attempt"
        finally:
            child.wait(timeout=60)
        with pytest.raises(PermissionError):
            _repoint(pointer, _redirect(other))
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)
    assert pointer.read_bytes() == original_pointer

    after = _NAMESPACE["acquire"]("after-stage", repo=linked)
    _NAMESPACE["release"](after)


def test_a_lost_pin_is_a_refused_barrier(anchor: Path, tmp_path: Path) -> None:
    """The barrier asks each pin to answer for the link it holds. A pin this
    process let go - closed behind the lease's back - is a hold that is not
    intact, and the next barrier refuses even though the pointer has not
    moved."""

    import ctypes

    linked, _other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    lease = _NAMESPACE["acquire"]("pinned", repo=linked)
    try:
        lease.assert_bound()
        assert lease.binding is not None and len(lease.binding.pins) >= 3
        dropped = lease.binding.pins[0]
        assert _NAMESPACE["_kernel32"]().CloseHandle(ctypes.c_void_p(dropped.handle))
        with pytest.raises(LeaseRefused, match="no longer held"):
            lease.assert_bound()
        assert pointer.read_bytes() == original_pointer
    finally:
        with pytest.raises(_NAMESPACE["LeaseStranded"]):
            _NAMESPACE["release"](lease)
    remaining = [pin for pin in lease.binding.pins if pin is not dropped] if lease.binding else []
    assert remaining == [], "the release let no pin go after the refused one"


def _resolve_then(
    module: dict[str, object], monkeypatch: pytest.MonkeyPatch, action: Callable[[], None]
) -> list[Path]:
    """Run ``action`` right after the repository is first resolved, before
    the resolved common directory is opened."""

    real_common_dir = module["_common_dir"]
    seen: list[Path] = []

    def common_dir(repo: Path | None) -> Path:
        common = real_common_dir(repo)  # type: ignore[operator]
        if not seen:
            seen.append(common)
            action()
        return common

    monkeypatch.setitem(module, "_common_dir", common_dir)
    return seen


def test_a_pointer_moved_before_the_open_is_refused(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A linked worktree whose pointer is redirected between the resolution
    and the open of the resolved directory is refused before any lease is
    opened: the held checkout, re-resolved after that open, names another
    object than the one held. This is the last moment a redirect can land
    at all; the pins that follow refuse every later one."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    module = _NAMESPACE["acquire"].__globals__
    seen = _resolve_then(
        module,
        monkeypatch,
        lambda: _repoint(pointer, f"gitdir: {(other / '.git').as_posix()}\n".encode()),
    )
    with pytest.raises(LeaseRefused, match="changed while it was being held"):
        _NAMESPACE["acquire"]("redirected-early", repo=linked)
    monkeypatch.undo()
    assert seen, "the seam never ran"
    assert not (other / ".git" / "machine-exclusive.lease").exists()
    assert not (anchor / ".git" / "machine-exclusive.lease").exists(), (
        "a lease was opened before the binding was verified"
    )
    _repoint(pointer, original_pointer)

    lease = _NAMESPACE["acquire"]("after-early-redirect", repo=linked)
    _NAMESPACE["release"](lease)


def test_a_directory_replaced_before_the_open_is_the_directory_leased(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git directory replaced under its name between the resolution and the
    open is what the repository now names: the lease is taken in it and
    bound to it, and a contender resolving the same repository is refused.
    The moved-away original has no holder and no checkout naming it."""

    module = _NAMESPACE["acquire"].__globals__
    git_dir = anchor / ".git"
    moved = anchor / ".git-moved"
    seen = _resolve_then(
        module, monkeypatch, lambda: _swap_directory_under_the_name(git_dir, moved)
    )
    lease = _NAMESPACE["acquire"]("current", repo=anchor)
    monkeypatch.undo()
    try:
        assert seen, "the seam never ran"
        assert (git_dir / "machine-exclusive.lease").exists()
        assert not (moved / "machine-exclusive.lease").exists()
        lease.assert_bound()
        with pytest.raises(LeaseRefused, match="contended"):
            _NAMESPACE["acquire"]("contender", repo=anchor)
    finally:
        _NAMESPACE["release"](lease)
    _restore_swapped(git_dir, moved)

    lease = _NAMESPACE["acquire"]("after-restore", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_held_lease_pins_its_directory_against_replacement(anchor: Path) -> None:
    """While a lease is held, the directory it lives in - and every directory
    above it - cannot be renamed away: the kernel refuses to move an
    ancestor of an open file. No impostor can be put under a held name, so
    an earlier holder's directory cannot be replaced beneath it; a
    redirected pointer is the only way a checkout changes repository under
    a holder, and that is what the barrier re-verifies."""

    git_dir = anchor / ".git"
    lease = _NAMESPACE["acquire"]("pinned", repo=anchor)
    try:
        with pytest.raises(PermissionError):
            git_dir.rename(anchor / ".git-moved")
        with pytest.raises(PermissionError):
            anchor.rename(anchor.parent / "repo-moved")
        assert git_dir.is_dir() and (git_dir / "machine-exclusive.lease").is_file()
    finally:
        _NAMESPACE["release"](lease)
    git_dir.rename(anchor / ".git-moved")
    (anchor / ".git-moved").rename(git_dir)

    lease = _NAMESPACE["acquire"]("after-pin", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_repository_replaced_under_its_name_during_resolution_is_refused(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository directory renamed away and replaced under its name while
    that name is being resolved is refused before any lease is opened: the
    held repository object no longer carries the name the resolution used,
    so the directory resolved through it is not this repository's."""

    import shutil

    module = _NAMESPACE["acquire"].__globals__
    moved = anchor.parent / "repo-moved"

    def replace_the_repository() -> None:
        anchor.rename(moved)
        shutil.copytree(moved, anchor)

    seen = _resolve_then(module, monkeypatch, replace_the_repository)
    with pytest.raises(LeaseRefused, match="moved while it was being resolved"):
        _NAMESPACE["acquire"]("resolving", repo=anchor)
    monkeypatch.undo()
    assert seen, "the seam never ran"
    assert not (anchor / ".git" / "machine-exclusive.lease").exists(), (
        "a lease was opened under the replacement"
    )
    assert not (moved / ".git" / "machine-exclusive.lease").exists()
    shutil.rmtree(anchor)
    moved.rename(anchor)

    lease = _NAMESPACE["acquire"]("after-replacement", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_hold_pins_its_checkout_until_it_ends(anchor: Path, tmp_path: Path) -> None:
    """A hold holds the checkout binding, not a poll over it: a redirect
    attempted while it waits is refused at the kernel - each time, however
    brief - the hold ends cleanly when its stdin closes, and only then can
    the pointer move."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    holder = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "machine_lock.py"),
            "--repo",
            str(linked),
            "hold",
            "--purpose",
            "pinned-hold",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdin is not None and holder.stdout is not None and holder.stderr is not None
        assert holder.stdout.readline().strip() == "READY"
        for _attempt in range(25):
            with pytest.raises(PermissionError):
                _repoint(pointer, _redirect(other))
        assert pointer.read_bytes() == original_pointer
        holder.stdin.close()
        holder.wait(timeout=30)
        report = holder.stderr.read()
    finally:
        with suppress(Exception):
            holder.kill()
    assert holder.returncode == 0, report
    _repoint(pointer, _redirect(other))
    assert pointer.read_bytes() == _redirect(other)
    _repoint(pointer, original_pointer)

    lease = _NAMESPACE["acquire"]("after-hold", repo=linked)
    _NAMESPACE["release"](lease)


def test_a_refused_close_on_a_refused_acquisition_is_a_strand(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal path closes the lease it opened; when the kernel refuses
    that close, the caller learns it is stranded with the refusal kept as
    the cause, and a contender stays excluded until the handle really
    closes."""

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create_file = kernel32.CreateFileW
    taken: dict[str, int] = {}

    def create_file_protecting_the_lease(name: str, *rest: object) -> object:
        handle = real_create_file(name, *rest)
        if str(name).endswith("machine-exclusive.lease") and "handle" not in taken:
            taken["handle"] = int(handle)
            assert kernel32.SetHandleInformation(ctypes.c_void_p(int(handle)), 2, 2)
        return handle

    module = _NAMESPACE["acquire"].__globals__

    def refuse_binding(binding: object) -> None:
        raise LeaseRefused("the repository moved under the lease: forced")

    monkeypatch.setattr(kernel32, "CreateFileW", create_file_protecting_the_lease)
    monkeypatch.setitem(module, "_assert_binding", refuse_binding)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="may still hold") as caught:
        _NAMESPACE["acquire"]("doomed", repo=anchor)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, LeaseRefused), "the refusal was lost"
    assert "forced" in str(caught.value.__cause__)
    assert caught.value.kind == "handle"
    assert caught.value.number == taken["handle"], "the message must name the live handle"

    with pytest.raises(LeaseRefused, match="contended"):
        _NAMESPACE["acquire"]("contender", repo=anchor)
    assert kernel32.SetHandleInformation(ctypes.c_void_p(taken["handle"]), 2, 0)
    assert kernel32.CloseHandle(ctypes.c_void_p(taken["handle"]))
    successor = _NAMESPACE["acquire"]("successor", repo=anchor)
    _NAMESPACE["release"](successor)


def _repoint(pointer: Path, content: bytes) -> None:
    """Rewrite a linked worktree pointer in place: the file is hidden on
    Windows, and a truncating open of a hidden file is refused."""

    with pointer.open("r+b") as stream:
        stream.write(content)
        stream.truncate()


def test_a_pointer_cannot_move_once_the_chain_is_pinned(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between the lease open and the post-open binding check - the interval
    a redirect once slipped through - the pointer is already pinned: a
    rewrite attempted there is refused by the kernel, the acquisition
    completes bound, and the pointer moves only after the release."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    module = _NAMESPACE["acquire"].__globals__
    real_assert_binding = module["_assert_binding"]
    refused: list[str] = []

    def rewrite_then_verify(binding: object) -> None:
        try:
            _repoint(pointer, _redirect(other))
        except PermissionError as refusal:
            refused.append(str(refusal))
        real_assert_binding(binding)  # type: ignore[operator]

    monkeypatch.setitem(module, "_assert_binding", rewrite_then_verify)
    lease = _NAMESPACE["acquire"]("pinned-open", repo=linked)
    monkeypatch.undo()
    try:
        assert refused, "the rewrite was not attempted at the post-open check"
        assert pointer.read_bytes() == original_pointer
        assert Path(lease.path).parent == (anchor / ".git").resolve()
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)
    _repoint(pointer, _redirect(other))
    assert pointer.read_bytes() == _redirect(other)
    _repoint(pointer, original_pointer)

    lease = _NAMESPACE["acquire"]("after-redirect", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_barrier_observes_the_live_kernel_handle(anchor: Path) -> None:
    """A raw close of the kernel handle leaves the stream object and its
    capability flag in place; the barrier asks the kernel and refuses the
    next stage anyway."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "raw-close"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
Assert-MachineLeaseHeld -Lease $Lease
Write-Output "HELD"
[LeaseNative.Kernel]::CloseHandle($Lease.Handle) | Out-Null
Write-Output "STREAM-CANWRITE:$($Lease.Stream.CanWrite)"
try {{
    Invoke-LeasedStage -Label "after raw close" -FilePath "{sys.executable}" `
        -ArgumentList @("-c", "print('stage ran')") -Lease $Lease
    Write-Output "STAGE-RAN"
}} catch {{
    Write-Output "STAGE-REFUSED: $_"
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "HELD" in result.stdout, result.stdout + result.stderr
    assert "STREAM-CANWRITE:True" in result.stdout, "the stream flag alone cannot see a raw close"
    assert "STAGE-REFUSED" in result.stdout, result.stdout + result.stderr
    assert "kernel handle is no longer held" in result.stdout
    assert "STAGE-RAN" not in result.stdout and "stage ran" not in result.stdout
    lease = _NAMESPACE["acquire"]("after-raw-close", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_holds_the_common_directory_through_the_open(anchor: Path) -> None:
    """A git directory replaced under its name after the hold and before the
    file open is refused by the shell too: the repository re-resolved
    through its held directory names another object, no lease is opened
    under the impostor, and the repository leases normally once the
    impostor is gone."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:Seen = 0
function Get-MachineLeaseFinalPath {{
    param([Parameter(Mandatory)]$Handle)
    $Buffer = New-Object System.Text.StringBuilder 32768
    $Length = [LeaseNative.Kernel]::GetFinalPathNameByHandleW( `
        $Handle, $Buffer, [uint32]$Buffer.Capacity, [uint32]0)
    $Name = $Buffer.ToString()
    if ($Name.EndsWith('.git')) {{
        $script:Seen += 1
        if ($script:Seen -eq 1) {{
            Rename-Item -LiteralPath "{anchor / ".git"}" -NewName ".git-moved"
            Copy-Item -LiteralPath "{anchor / ".git-moved"}" `
                -Destination "{anchor / ".git"}" -Recurse
        }}
    }}
    return $Name
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "split"
if ($Lease) {{ Write-Output "LEASED-UNDER-IMPOSTOR"; exit 1 }}
Write-Output "REFUSED-AS-EXPECTED:$script:Seen"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed while it was being held" in result.stdout, result.stdout
    assert "REFUSED-AS-EXPECTED:1" in result.stdout
    assert not (anchor / ".git" / "machine-exclusive.lease").exists(), (
        "a lease was opened under the impostor"
    )
    _restore_swapped(anchor / ".git", anchor / ".git-moved")
    lease = _NAMESPACE["acquire"]("after-move", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_refuses_a_directory_replaced_before_the_pins(anchor: Path) -> None:
    """A git directory replaced under its name after the shell re-verified
    the repository and before it pinned the chain is caught by the
    resolution repeated through the pinned chain: refused before any lease
    is opened, with no lease under the impostor."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealIdentity = ${{function:Get-MachineLeaseDirectoryIdentity}}
$script:Swapped = $false
function Get-MachineLeaseDirectoryIdentity {{
    param([Parameter(Mandatory)][string]$Path)
    $Identity = & $script:RealIdentity -Path $Path
    if (-not $script:Swapped) {{
        $script:Swapped = $true
        Rename-Item -LiteralPath "{anchor / ".git"}" -NewName ".git-moved"
        Copy-Item -LiteralPath "{anchor / ".git-moved"}" `
            -Destination "{anchor / ".git"}" -Recurse
    }}
    return $Identity
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "split-late"
if ($Lease) {{ Write-Output "LEASED-UNDER-IMPOSTOR"; exit 1 }}
Write-Output "REFUSED-AS-EXPECTED:$script:Swapped"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed while it was being pinned" in result.stdout, result.stdout
    assert "REFUSED-AS-EXPECTED:True" in result.stdout
    assert not (anchor / ".git" / "machine-exclusive.lease").exists(), (
        "a lease was opened under the impostor"
    )
    assert not (anchor / ".git-moved" / "machine-exclusive.lease").exists()
    _restore_swapped(anchor / ".git", anchor / ".git-moved")
    lease = _NAMESPACE["acquire"]("after-late-move", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_holds_the_checkout_binding_through_a_live_stage(
    anchor: Path, tmp_path: Path
) -> None:
    """While a leased stage child runs, the shell's pins refuse every
    rewrite of the checkout pointer from outside the process, the stage
    completes, and the pointer moves only after the lease is released."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "live-stage"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
Assert-MachineLeaseHeld -Lease $Lease
Write-Output "STAGE-STARTING"
Invoke-LeasedStage -Label "a stage that waits" -FilePath "{sys.executable}" `
    -ArgumentList @("-c", "import time; print('stage running', flush=True); time.sleep(3)") `
    -Lease $Lease
Write-Output "STAGE-DONE"
Write-Output "RELEASED:$(Exit-MachineLease $Lease)"
# Stay alive after the release so the pins' fate is observable: a release
# that kept them would still hold the pointer here.
Start-Sleep -Seconds 3
Write-Output "EXITING"
"""
    shell = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    refusals = 0
    try:
        assert shell.stdout is not None and shell.stderr is not None
        while True:
            line = shell.stdout.readline()
            assert line, "the shell ended before its stage ran"
            if line.strip() == "stage running":
                break
        for _attempt in range(10):
            with pytest.raises(PermissionError):
                _repoint(pointer, _redirect(other))
            refusals += 1
        with pytest.raises(LeaseRefused, match="contended"):
            _NAMESPACE["acquire"]("second", repo=linked)
        seen: list[str] = []
        while True:
            line = shell.stdout.readline()
            assert line, "the shell ended before it released"
            seen.append(line.strip())
            if line.strip() == "RELEASED:True":
                break
        assert "STAGE-DONE" in seen, seen
        # The lease is released but the process still lives: the pins went
        # with the lease, so the pointer moves now, not at process exit.
        _repoint(pointer, _redirect(other))
        assert pointer.read_bytes() == _redirect(other)
        _repoint(pointer, original_pointer)
        rest, errors = shell.communicate(timeout=120)
    finally:
        with suppress(Exception):
            shell.kill()
    assert shell.returncode == 0, rest + errors
    assert "EXITING" in rest, rest
    assert refusals == 10
    assert pointer.read_bytes() == original_pointer
    lease = _NAMESPACE["acquire"]("after-shell-stage", repo=linked)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_refuses_a_pointer_moved_before_the_open(anchor: Path, tmp_path: Path) -> None:
    """The shell refuses a linked worktree whose pointer is redirected
    between the resolution and the open of the resolved directory, before
    any lease is opened."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealCommonDir = ${{function:Get-MachineLeaseCommonDir}}
$script:Calls = 0
function Get-MachineLeaseCommonDir {{
    param([Parameter(Mandatory)][string]$RepositoryRoot)
    $Common = & $script:RealCommonDir -RepositoryRoot $RepositoryRoot
    $script:Calls += 1
    if ($script:Calls -eq 1) {{
        $Stream = [IO.File]::Open("{pointer}", [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite)
        $Bytes = [Text.Encoding]::ASCII.GetBytes("gitdir: {(other / ".git").as_posix()}`n")
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.SetLength($Bytes.Length)
        $Stream.Dispose()
    }}
    return $Common
}}
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "redirected-early"
if ($Lease) {{ Write-Output "LEASED-UNDER-REDIRECT"; exit 1 }}
Write-Output "REFUSED-AS-EXPECTED:$script:Calls"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed while it was being held" in result.stdout, result.stdout
    assert "REFUSED-AS-EXPECTED:2" in result.stdout, result.stdout
    assert not (other / ".git" / "machine-exclusive.lease").exists()
    assert not (anchor / ".git" / "machine-exclusive.lease").exists(), (
        "a lease was opened before the binding was verified"
    )
    _repoint(pointer, original_pointer)
    lease = _NAMESPACE["acquire"]("after-shell-early-redirect", repo=linked)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_leases_a_directory_replaced_before_the_open(anchor: Path) -> None:
    """The shell leases the directory the repository names once it is held:
    a git directory replaced under its name between the resolution and the
    open is the one leased and bound, and a contender is refused while it
    is held."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealCommonDir = ${{function:Get-MachineLeaseCommonDir}}
$script:Calls = 0
function Get-MachineLeaseCommonDir {{
    param([Parameter(Mandatory)][string]$RepositoryRoot)
    $Common = & $script:RealCommonDir -RepositoryRoot $RepositoryRoot
    $script:Calls += 1
    if ($script:Calls -eq 1) {{
        Rename-Item -LiteralPath "{anchor / ".git"}" -NewName ".git-moved"
        Copy-Item -LiteralPath "{anchor / ".git-moved"}" `
            -Destination "{anchor / ".git"}" -Recurse
    }}
    return $Common
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "current"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
Assert-MachineLeaseHeld -Lease $Lease
Write-Output "BOUND"
$Contender = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "contender"
if ($Contender) {{ Write-Output "CONTENDER-LEASED"; exit 1 }}
Write-Output "CONTENDER-REFUSED"
Write-Output "RELEASED:$(Exit-MachineLease $Lease)"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "BOUND" in out and "CONTENDER-REFUSED" in out, out
    assert "the machine is held" in out, out
    assert "RELEASED:True" in out, out
    assert not (anchor / ".git-moved" / "machine-exclusive.lease").exists(), (
        "the moved-away original was leased"
    )
    _restore_swapped(anchor / ".git", anchor / ".git-moved")
    lease = _NAMESPACE["acquire"]("after-shell-current", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_refuses_a_repository_replaced_during_resolution(anchor: Path) -> None:
    """The shell refuses a repository directory renamed away and replaced
    under its name while that name is being resolved, before any lease is
    opened."""

    import shutil

    moved = anchor.parent / "repo-moved"
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealCommonDir = ${{function:Get-MachineLeaseCommonDir}}
$script:Calls = 0
function Get-MachineLeaseCommonDir {{
    param([Parameter(Mandatory)][string]$RepositoryRoot)
    $Common = & $script:RealCommonDir -RepositoryRoot $RepositoryRoot
    $script:Calls += 1
    if ($script:Calls -eq 1) {{
        Rename-Item -LiteralPath "{anchor}" -NewName "repo-moved"
        Copy-Item -LiteralPath "{moved}" -Destination "{anchor}" -Recurse
    }}
    return $Common
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "resolving"
if ($Lease) {{ Write-Output "LEASED-UNDER-REPLACEMENT"; exit 1 }}
Write-Output "REFUSED-AS-EXPECTED:$script:Calls"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "moved while it was being resolved" in result.stdout, result.stdout
    assert "REFUSED-AS-EXPECTED:1" in result.stdout, result.stdout
    assert not (anchor / ".git" / "machine-exclusive.lease").exists(), (
        "a lease was opened under the replacement"
    )
    assert not (moved / ".git" / "machine-exclusive.lease").exists()
    shutil.rmtree(anchor)
    moved.rename(anchor)
    lease = _NAMESPACE["acquire"]("after-shell-replacement", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_pins_the_pointer_before_its_post_open_check(
    anchor: Path, tmp_path: Path
) -> None:
    """At the shell's post-open binding check the pointer is already pinned:
    a rewrite attempted there fails at the kernel, the acquisition
    completes bound, and the pointer moves only after the release."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealBinding = ${{function:Test-MachineLeaseBinding}}
$script:Refused = 0
function Test-MachineLeaseBinding {{
    param([Parameter(Mandatory)]$Binding)
    try {{
        $Stream = [IO.File]::Open("{pointer}", [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite)
        $Stream.Dispose()
    }} catch [System.IO.IOException] {{
        $script:Refused += 1
    }}
    return (& $script:RealBinding -Binding $Binding)
}}
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "pinned-open"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
Assert-MachineLeaseHeld -Lease $Lease
Write-Output "REFUSED-WRITES:$script:Refused"
Write-Output "RELEASED:$(Exit-MachineLease $Lease)"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REFUSED-WRITES:2" in result.stdout, result.stdout
    assert "RELEASED:True" in result.stdout
    assert pointer.read_bytes() == original_pointer
    _repoint(pointer, _redirect(other))
    _repoint(pointer, original_pointer)
    lease = _NAMESPACE["acquire"]("after-shell-pin", repo=linked)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_exits_stranded_when_a_refusal_close_is_refused(anchor: Path) -> None:
    """When the shell refuses an acquisition after the open and the kernel
    then refuses to close the lease it opened, the process reports the
    strand and exits 4 instead of returning an ordinary refusal over a
    live hold."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealCommonDir = ${{function:Get-MachineLeaseCommonDir}}
$script:Calls = 0
function Get-MachineLeaseCommonDir {{
    param([Parameter(Mandatory)][string]$RepositoryRoot)
    $script:Calls += 1
    if ($script:Calls -eq 4) {{
        # The post-open re-verification - after the resolution, the
        # re-resolution and the pinned re-resolution: the lease handle is
        # live in the opening function's scope; protect it, then stop
        # resolving.
        [LeaseNative.Kernel]::SetHandleInformation($Handle, 2, 2) | Out-Null
        return $null
    }}
    return (& $script:RealCommonDir -RepositoryRoot $RepositoryRoot)
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "stranded"
Write-Output "RETURNED:$($null -eq $Lease)"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 4, out + result.stderr
    assert "no longer resolves a git repository" in out, out
    assert "stranded holding the machine" in out, out
    assert "RETURNED:" not in out, "a stranded process must not return to its caller"
    lease = _NAMESPACE["acquire"]("after-strand", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_success_message_follows_the_release(anchor: Path) -> None:
    """Complete-MachineLeaseRun prints the success message only after a
    clean release, and withholds it when the release is refused."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "clean-run"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
Write-Output "BODY-DONE"
$Ok = Complete-MachineLeaseRun -Lease $Lease -BodyPassed $true `
    -SuccessMessage "GREEN-BANNER" -Epilogue "EPILOGUE"
Write-Output "FIRST-RESULT:$Ok"
$Second = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "refused-release"
if (-not $Second) {{ Write-Output "ENTER-FAILED"; exit 1 }}
[LeaseNative.Kernel]::SetHandleInformation($Second.Handle, 2, 2) | Out-Null
$Ok2 = Complete-MachineLeaseRun -Lease $Second -BodyPassed $true `
    -SuccessMessage "GREEN-BANNER-2" -Epilogue "EPILOGUE-2"
Write-Output "SECOND-RESULT:$Ok2"
[LeaseNative.Kernel]::SetHandleInformation($Second.Handle, 2, 0) | Out-Null
[LeaseNative.Kernel]::CloseHandle($Second.Handle) | Out-Null
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert "FIRST-RESULT:True" in out, out + result.stderr
    assert out.index("BODY-DONE") < out.index("GREEN-BANNER") < out.index("EPILOGUE")
    assert "SECOND-RESULT:False" in out, out + result.stderr
    assert "GREEN-BANNER-2" not in out and "EPILOGUE-2" not in out, (
        "a refused release must withhold the success message"
    )
    assert "did not release cleanly" in out
    lease = _NAMESPACE["acquire"]("after-completion", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_catch_reports_the_failure_before_its_cleanup(anchor: Path) -> None:
    """The initialization failure is the first line the catch prints; the
    cleanup result follows it, and a close that succeeded says so."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
function ConvertTo-Json {{ throw "record build failed" }}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "ordered"
Write-Output "RESULT:$($null -eq $Lease)"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert "RESULT:True" in out, out + result.stderr
    errors = [line for line in out.splitlines() if line.startswith("ERROR:")]
    assert errors and "could not be initialized" in errors[0], errors
    assert "handle was closed" not in errors[0], (
        "the cleanup result is its own report, after the initialization failure"
    )
    assert any("handle was closed" in line for line in errors[1:]), errors
    assert "NOT closed" not in out
    lease = _NAMESPACE["acquire"]("after-ordered", repo=anchor)
    _NAMESPACE["release"](lease)


def test_a_held_lease_pins_every_link_of_the_chain(anchor: Path, tmp_path: Path) -> None:
    """Under a lease taken through a linked checkout, the pointer, the
    private git directory and its commondir file are all held: rewriting
    the commondir file or renaming the private directory is refused just as
    rewriting the pointer is, and each moves again only after the release."""

    linked, _other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    private = Path(pointer.read_text(encoding="utf-8")[len("gitdir:") :].strip())
    commondir = private / "commondir"
    original_commondir = commondir.read_bytes()
    lease = _NAMESPACE["acquire"]("chained", repo=linked)
    try:
        assert lease.binding is not None
        pinned = {Path(pin.path) for pin in lease.binding.pins}
        assert {pointer, private, private.parent, commondir, anchor / ".git"} <= pinned
        with pytest.raises(PermissionError):
            _repoint(commondir, b"../../elsewhere" + bytes([10]))
        with pytest.raises(PermissionError):
            private.rename(private.with_name("moved-private"))
        assert commondir.read_bytes() == original_commondir
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)
    _repoint(commondir, original_commondir)
    private.rename(private.with_name("moved-private"))
    private.with_name("moved-private").rename(private)
    assert pointer.read_bytes() == original_pointer

    after = _NAMESPACE["acquire"]("after-chain", repo=linked)
    _NAMESPACE["release"](after)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_barrier_refuses_a_lost_pin(anchor: Path, tmp_path: Path) -> None:
    """The shell barrier asks each pin to answer for its link: a pin the
    process let go - closed behind the lease's back - refuses the next
    stage even though the pointer has not moved and the kernel handle is
    still live."""

    linked, _other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "lost-pin"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
Assert-MachineLeaseHeld -Lease $Lease
Write-Output "HELD:$(@($Lease.Binding.Pins).Count)"
[LeaseNative.Kernel]::CloseHandle($Lease.Binding.Pins[0].Handle) | Out-Null
Write-Output "LIVE:$(Get-MachineLeaseIdentity -Handle $Lease.Handle)"
try {{
    Invoke-LeasedStage -Label "after a lost pin" -FilePath "{sys.executable}" `
        -ArgumentList @("-c", "print('stage ran')") -Lease $Lease
    Write-Output "STAGE-RAN"
}} catch {{
    Write-Output "STAGE-REFUSED: $_"
}}
[LeaseNative.Kernel]::CloseHandle($Lease.Handle) | Out-Null
foreach ($Pin in @($Lease.Binding.Pins) | Select-Object -Skip 1) {{
    [LeaseNative.Kernel]::CloseHandle($Pin.Handle) | Out-Null
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    held = [line for line in out.splitlines() if line.startswith("HELD:")]
    assert len(held) == 1 and int(held[0].split(":")[1]) >= 4, out + result.stderr
    live = [line for line in out.splitlines() if line.startswith("LIVE:")]
    assert live and live[0] != "LIVE:", "the kernel handle must still be live"
    assert "STAGE-REFUSED" in out, out + result.stderr
    assert "is no longer held by this process" in out
    assert "STAGE-RAN" not in out and "stage ran" not in out
    assert pointer.read_bytes() == original_pointer
    lease = _NAMESPACE["acquire"]("after-lost-pin", repo=linked)
    _NAMESPACE["release"](lease)


def _junction(link: Path, target: Path) -> None:
    """A directory junction at ``link`` naming ``target``: the reparse
    point Windows follows, and retargets, per component."""

    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def _wait_until_the_pointer_moves(pointer: Path, content: bytes) -> None:
    deadline = time.monotonic() + 15
    while True:
        try:
            _repoint(pointer, content)
            return
        except PermissionError:
            assert time.monotonic() < deadline, "the pointer stayed pinned after every holder died"
            time.sleep(0.2)


def _protect(handle: int) -> None:
    import ctypes

    assert _NAMESPACE["_kernel32"]().SetHandleInformation(ctypes.c_void_p(handle), 2, 2)


def _unprotect_and_close(handle: int) -> None:
    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    assert kernel32.SetHandleInformation(ctypes.c_void_p(handle), 2, 0)
    assert kernel32.CloseHandle(ctypes.c_void_p(handle))


def _protecting_entry_pin(taken: dict[str, int]):  # type: ignore[no-untyped-def]
    """Protect the retained .git entry pin, leaving earlier parent retirement free."""

    real_open_pin = _NAMESPACE["acquire"].__globals__["_open_pin"]

    def open_pin(path: Path, role: str):  # type: ignore[no-untyped-def]
        pin = real_open_pin(path, role)
        if role == "the repository's .git entry" and "pin" not in taken:
            taken["pin"] = pin.handle
            _protect(pin.handle)
        return pin

    return open_pin


def test_a_stage_child_keeps_the_chain_pinned_after_its_parent_dies(
    anchor: Path, tmp_path: Path
) -> None:
    """The holder acquires through a linked checkout, launches a child that
    inherits its handles, and dies without releasing. The pins are
    inheritable exactly as the lease is: while the child can still act the
    pointer cannot be rewritten and no rival can lease through the
    checkout, and the pointer moves again only once the child is gone."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    script = tmp_path / "parent_that_dies.py"
    script.write_text(_PARENT_THAT_DIES, encoding="utf-8")
    pid_file = tmp_path / "child_pid.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(linked),
            str(pid_file),
            str(ROOT / "scripts" / "machine_lock.py"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        with pytest.raises(PermissionError):
            _repoint(pointer, _redirect(other))
        with pytest.raises(LeaseRefused, match="parent-that-dies"):
            _NAMESPACE["acquire"]("rival", repo=linked)
        time.sleep(1.0)
        with pytest.raises(PermissionError):
            _repoint(pointer, _redirect(other))
        assert pointer.read_bytes() == original_pointer
    finally:
        subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], capture_output=True, check=False)

    _wait_until_the_pointer_moves(pointer, _redirect(other))
    moved = _NAMESPACE["acquire"]("after-the-child", repo=linked)
    assert Path(moved.path).parent == (other / ".git").resolve()
    _NAMESPACE["release"](moved)
    _repoint(pointer, original_pointer)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_stage_child_keeps_the_chain_pinned_after_the_launcher_dies(
    anchor: Path, tmp_path: Path
) -> None:
    """The shell launcher enters through a linked checkout and runs its
    stage child through the leased-stage runner; the launcher is killed
    while the child runs. The child's inherited pins keep the pointer
    immovable and a rival refused for as long as it can act, and the
    pointer moves only once it exits."""

    linked, other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    pid_file = tmp_path / "stage_pid.txt"
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "launcher-parent"
if (-not $Lease) {{ exit 1 }}
Invoke-LeasedStage -Label "sleeping stage" -FilePath "{sys.executable}" `
    -ArgumentList @("{sleeper}", "{pid_file}") -Lease $Lease
"""
    holder = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stage_pid = -1
    try:
        deadline = time.monotonic() + 60

        def holder_alive() -> None:
            assert holder.poll() is None, "the holder died before its stage ran"

        stage_pid = int(_settled_text(pid_file, deadline=deadline, still_running=holder_alive))

        holder.kill()
        holder.wait(timeout=30)

        with pytest.raises(PermissionError):
            _repoint(pointer, _redirect(other))
        with pytest.raises(LeaseRefused, match="launcher-parent"):
            _NAMESPACE["acquire"]("rival", repo=linked)
        time.sleep(1.0)
        with pytest.raises(PermissionError):
            _repoint(pointer, _redirect(other))
        assert pointer.read_bytes() == original_pointer
    finally:
        if holder.poll() is None:
            holder.kill()
        if stage_pid != -1:
            subprocess.run(
                ["taskkill", "/PID", str(stage_pid), "/F"], capture_output=True, check=False
            )

    _wait_until_the_pointer_moves(pointer, _redirect(other))
    moved = _NAMESPACE["acquire"]("after-the-stage", repo=linked)
    assert Path(moved.path).parent == (other / ".git").resolve()
    _NAMESPACE["release"](moved)
    _repoint(pointer, original_pointer)


def _settled_text(
    path: Path,
    *,
    deadline: float,
    still_running: Callable[[], None],
) -> str:
    """The file's content once it has been written AND closed.

    Waiting on `path.exists()` waits for the wrong thing. A writer that creates
    the file before it finishes writing holds it in between, and on Windows it
    holds it without share-read - so a reader in that window gets
    `PermissionError: [Errno 13]` on a path that plainly exists. That is a real
    failure this suite hit in a merge-group run rather than a hypothetical: the
    same commit had passed the identical job minutes earlier, and only the
    timing differed.

    So the condition is content, not presence. An unreadable or empty file is
    the writer mid-write and is retried; `still_running` re-checks whatever
    produced it, so a dead producer fails immediately instead of at the
    deadline.
    """

    while True:
        still_running()
        assert time.monotonic() < deadline, f"{path.name} never became readable"
        with suppress(OSError):
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text
        time.sleep(0.2)


def _pointer_through_a_junction(
    anchor: Path, tmp_path: Path
) -> tuple[Path, Path, Path, Path, bytes]:
    """A linked checkout whose pointer names its private directory through
    a junction: the checkout, the junction, the directory the junction
    names, the pointer and its original bytes."""

    linked, _other, pointer, original_pointer = _linked_worktree(anchor, tmp_path)
    worktrees = (anchor / ".git" / "worktrees").resolve()
    # Through a junction a relative commondir collapses lexically to the
    # junction's parent, so the private directory names its common
    # directory absolutely, as git accepts.
    (worktrees / linked.name / "commondir").write_bytes(
        f"{(anchor / '.git').resolve().as_posix()}\n".encode()
    )
    jump = tmp_path / "jump"
    _junction(jump, worktrees)
    _repoint(pointer, f"gitdir: {(jump / linked.name).as_posix()}\n".encode())
    subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
    )
    return linked, jump, worktrees, pointer, original_pointer


def test_a_junction_on_the_way_to_the_private_directory_is_pinned(
    anchor: Path, tmp_path: Path
) -> None:
    """The pointer's text crosses a junction. Retargeting that junction
    would change what the text names while the private directory at its
    end stayed held, so the junction is a link of the chain: it is pinned
    as itself, cannot be removed or retargeted by this process or another
    while the lease lives, and moves again only after the release."""

    linked, jump, worktrees, pointer, original_pointer = _pointer_through_a_junction(
        anchor, tmp_path
    )
    try:
        lease = _NAMESPACE["acquire"]("through-a-junction", repo=linked)
        try:
            roles = {Path(pin.path): pin.role for pin in lease.binding.pins}
            assert (
                roles.get(jump) == "a component on the way to the checkout's private git directory"
            )
            with pytest.raises(PermissionError):
                os.rmdir(jump)
            elsewhere = subprocess.run(
                [sys.executable, "-c", "import os, sys; os.rmdir(sys.argv[1])", str(jump)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert elsewhere.returncode != 0 and "PermissionError" in elsewhere.stderr
            assert jump.is_dir()
            lease.assert_bound()
        finally:
            _NAMESPACE["release"](lease)
        os.rmdir(jump)
        assert not jump.exists()
    finally:
        if not jump.exists():
            _junction(jump, worktrees)
        _repoint(pointer, original_pointer)
        os.rmdir(jump)


def test_a_junction_on_the_way_to_the_common_directory_is_pinned(
    anchor: Path, tmp_path: Path
) -> None:
    """The commondir file names the common directory through a junction.
    The junction is a link of the chain exactly as the file is: pinned as
    itself, immovable while the lease lives, movable after it."""

    linked, _other, _pointer, _original = _linked_worktree(anchor, tmp_path)
    common = (anchor / ".git").resolve()
    private = common / "worktrees" / linked.name
    commondir = private / "commondir"
    original_text = commondir.read_bytes()
    common_link = tmp_path / "common-link"
    _junction(common_link, common)
    commondir.write_bytes(f"{common_link.as_posix()}\n".encode())
    try:
        subprocess.run(
            ["git", "-C", str(linked), "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
        )
        lease = _NAMESPACE["acquire"]("common-through-a-junction", repo=linked)
        try:
            roles = {Path(pin.path): pin.role for pin in lease.binding.pins}
            assert roles.get(common_link) == "the common git directory"
            with pytest.raises(PermissionError):
                os.rmdir(common_link)
            assert common_link.is_dir()
            lease.assert_bound()
        finally:
            _NAMESPACE["release"](lease)
        os.rmdir(common_link)
        assert not common_link.exists()
    finally:
        commondir.write_bytes(original_text)
        if common_link.exists():
            os.rmdir(common_link)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_pins_a_junction_on_the_way_to_the_private_directory(
    anchor: Path, tmp_path: Path
) -> None:
    """The shell entry point pins the junction the pointer crosses: while
    the shell holds the lease the junction cannot be removed from another
    process, and it can once the shell has exited the lease."""

    linked, jump, worktrees, pointer, original_pointer = _pointer_through_a_junction(
        anchor, tmp_path
    )
    entered = tmp_path / "entered.txt"
    release = tmp_path / "release.txt"
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "shell-junction"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
$Roles = @($Lease.Binding.Pins | ForEach-Object {{ "$($_.Role)=$($_.Path)" }})
Set-Content -LiteralPath "{entered}" -Value ($Roles -join "`n") -Encoding utf8
while (-not (Test-Path -LiteralPath "{release}")) {{ Start-Sleep -Milliseconds 200 }}
if (Exit-MachineLease $Lease) {{ Write-Output "RELEASED" }}
"""
    process = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60

        def shell_alive() -> None:
            assert process.poll() is None, process.communicate()[0]

        roles = _settled_text(entered, deadline=deadline, still_running=shell_alive)
        assert f"a component on the way to the checkout's private git directory={jump}" in roles, (
            roles
        )
        with pytest.raises(PermissionError):
            os.rmdir(jump)
        assert jump.is_dir()
        release.write_text("release", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stdout + stderr
        assert "RELEASED" in stdout
        os.rmdir(jump)
        assert not jump.exists()
    finally:
        with suppress(Exception):
            process.kill()
        if not jump.exists():
            _junction(jump, worktrees)
        _repoint(pointer, original_pointer)
        os.rmdir(jump)


def test_a_release_reports_the_descriptor_and_a_pin_that_will_not_close(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The descriptor's close is refused and so is one pin's: the release
    still attempts every pin, and one strand names both refused closes,
    the descriptor first with the kernel's refusal as its cause."""

    def refusing(descriptor: int) -> None:
        raise OSError("the kernel refused the close")

    real_close = os.close
    lease = _NAMESPACE["acquire"]("two-strands", repo=anchor)
    descriptor = lease.descriptor
    pins = lease.binding.pins
    assert len(pins) >= 2
    protected = pins[1].handle
    _protect(protected)
    monkeypatch.setattr(os, "close", refusing)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="every refused close") as caught:
        _NAMESPACE["release"](lease)
    monkeypatch.undo()
    real_close(descriptor)
    strands = caught.value.strands
    assert [kind for kind, _number, _error in strands] == ["descriptor", "pin"]
    assert strands[0][1] == descriptor and strands[1][1] == protected
    assert isinstance(caught.value.__cause__, OSError)
    for pin in pins:
        if pin.handle != protected:
            assert not _NAMESPACE["_kernel32"]().GetFileInformationByHandle(
                __import__("ctypes").c_void_p(pin.handle),
                __import__("ctypes").byref(_NAMESPACE["_ByHandleFileInformation"]()),
            ), "a pin behind the refused closes stayed open"
    _unprotect_and_close(protected)
    successor = _NAMESPACE["acquire"]("after-two-strands", repo=anchor)
    _NAMESPACE["release"](successor)


def _pin_refused_at(taken: dict[str, int], refused_role: str):  # type: ignore[no-untyped-def]
    """Protect the retained entry pin, then refuse the named acquisition boundary."""

    protecting = _protecting_entry_pin(taken)

    def open_pin(path: Path, role: str):  # type: ignore[no-untyped-def]
        if role == refused_role:
            raise LeaseRefused(f"{role} could not be held: forced")
        return protecting(path, role)

    return open_pin


def test_a_partial_pinning_reports_a_pin_that_will_not_close(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The private directory cannot be pinned. Earlier pins are let go,
    and when the kernel refuses the entry pin's close
    the caller learns it as a strand raised from the pinning refusal, not
    as an ordinary refusal over a live pin."""

    linked, _other, _pointer, _original = _linked_worktree(anchor, tmp_path)
    taken: dict[str, int] = {}
    module = _NAMESPACE["acquire"].__globals__
    monkeypatch.setitem(
        module, "_open_pin", _pin_refused_at(taken, "the checkout's private git directory")
    )
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="a refused pinning") as caught:
        _NAMESPACE["acquire"]("doomed-pinning", repo=linked)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, LeaseRefused)
    assert "private git directory could not be held: forced" in str(caught.value.__cause__)
    assert caught.value.strands == (("pin", taken["pin"], caught.value.error),)
    _unprotect_and_close(taken["pin"])
    successor = _NAMESPACE["acquire"]("after-partial-pinning", repo=linked)
    _NAMESPACE["release"](successor)


def test_a_refused_common_directory_pin_reports_a_chain_pin_that_will_not_close(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain is pinned and the common directory's own pin refuses; the
    chain's pins are let go, and the one the kernel will not close is a
    strand raised from the refusal."""

    taken: dict[str, int] = {}
    module = _NAMESPACE["acquire"].__globals__
    monkeypatch.setitem(module, "_open_pin", _pin_refused_at(taken, "the common git directory"))
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="a refused acquisition") as caught:
        _NAMESPACE["acquire"]("doomed-common-pin", repo=anchor)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, LeaseRefused)
    assert "common git directory could not be held: forced" in str(caught.value.__cause__)
    assert caught.value.strands == (("pin", taken["pin"], caught.value.error),)
    _unprotect_and_close(taken["pin"])
    successor = _NAMESPACE["acquire"]("after-common-pin", repo=anchor)
    _NAMESPACE["release"](successor)


def test_a_failed_initialization_reports_a_pin_that_will_not_close(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record cannot be built after the open. The descriptor closes,
    every pin is attempted, and the one the kernel refuses is a strand
    raised from the initialization failure; the machine itself is free."""

    linked, _other, _pointer, _original = _linked_worktree(anchor, tmp_path)
    taken: dict[str, int] = {}
    module = _NAMESPACE["acquire"].__globals__

    def failing_dumps(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("record build failed")

    monkeypatch.setitem(module, "_open_pin", _protecting_entry_pin(taken))
    monkeypatch.setattr(module["json"], "dumps", failing_dumps)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="a failed acquisition") as caught:
        _NAMESPACE["acquire"]("doomed-record", repo=linked)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.strands == (("pin", taken["pin"], caught.value.error),)
    # The lease handle closed: a contender takes the machine at once.
    successor = _NAMESPACE["acquire"]("after-failed-record", repo=linked)
    _NAMESPACE["release"](successor)
    _unprotect_and_close(taken["pin"])


def test_a_refused_binding_reports_the_handle_and_a_pin_that_will_not_close(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binding is refused after the open; the lease handle and one pin
    both refuse to close. One strand names both, the handle first, raised
    from the refusal."""

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create_file = kernel32.CreateFileW
    linked, _other, _pointer, _original = _linked_worktree(anchor, tmp_path)
    taken: dict[str, int] = {}

    def create_file_protecting_the_lease(name: str, *rest: object) -> object:
        handle = real_create_file(name, *rest)
        if str(name).endswith("machine-exclusive.lease") and "handle" not in taken:
            taken["handle"] = int(handle)
            _protect(int(handle))
        return handle

    module = _NAMESPACE["acquire"].__globals__

    def refuse_binding(binding: object) -> None:
        raise LeaseRefused("the repository moved under the lease: forced")

    monkeypatch.setitem(module, "_open_pin", _protecting_entry_pin(taken))
    monkeypatch.setattr(kernel32, "CreateFileW", create_file_protecting_the_lease)
    monkeypatch.setitem(module, "_assert_binding", refuse_binding)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="every refused close") as caught:
        _NAMESPACE["acquire"]("doomed-binding", repo=linked)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, LeaseRefused)
    kinds = [(kind, number) for kind, number, _error in caught.value.strands]
    assert kinds == [("handle", taken["handle"]), ("pin", taken["pin"])]
    with pytest.raises(LeaseRefused, match="contended"):
        _NAMESPACE["acquire"]("contender", repo=linked)
    assert kernel32.SetHandleInformation(ctypes.c_void_p(taken["handle"]), 2, 0)
    assert kernel32.CloseHandle(ctypes.c_void_p(taken["handle"]))
    _unprotect_and_close(taken["pin"])
    successor = _NAMESPACE["acquire"]("after-refused-binding", repo=linked)
    _NAMESPACE["release"](successor)


def test_a_status_probe_reports_a_pin_that_will_not_close_and_frees_the_machine(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status probe's pin refuses to close: the probe's write-access
    handle is closed first all the same, so the machine is free, and the
    strand names the pin."""

    _lease_file(anchor).write_text("stale record", encoding="utf-8")
    linked, _other, _pointer, _original = _linked_worktree(anchor, tmp_path)
    taken: dict[str, int] = {}
    module = _NAMESPACE["acquire"].__globals__
    monkeypatch.setitem(module, "_open_pin", _protecting_entry_pin(taken))
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="a status probe") as caught:
        _NAMESPACE["status"](linked)
    monkeypatch.undo()
    assert caught.value.strands == (("pin", taken["pin"], caught.value.error),)
    # The probe closed: a contender takes the machine at once.
    successor = _NAMESPACE["acquire"]("after-probe-strand", repo=linked)
    _NAMESPACE["release"](successor)
    _unprotect_and_close(taken["pin"])


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_reports_a_pin_that_will_not_close_after_a_failed_initialization(
    anchor: Path,
) -> None:
    """Enter's record build fails after the open with one pin protected
    from close: the lease handle closes, every pin is attempted, the
    refused pin is reported, and Enter never claims that nothing is
    held."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
function ConvertTo-Json {{
    [LeaseNative.Kernel]::SetHandleInformation($Opened.Binding.Pins[0].Handle, 2, 2) | Out-Null
    throw "record build failed"
}}
$First = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "doomed-pin"
if ($First) {{ Write-Output "FIRST-ACQUIRED"; exit 2 }}
Write-Output "FIRST-NULL"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "FIRST-NULL" in out
    assert "refused the pin on" in out, out
    assert "stranded after the failed acquisition" in out, out
    assert "nothing is held" not in out, "Enter claimed a clean cleanup over a refused pin close"
    assert "refused the lease handle" not in out, "the lease handle closed and must not be reported"
    lease = _NAMESPACE["acquire"]("after-shell-pin-strand", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_exits_stranded_when_a_refused_acquisition_cannot_close_a_pin(
    anchor: Path,
) -> None:
    """The post-open binding check refuses with one pin protected from
    close: the lease handle closes, the pin's refusal is reported, and the
    process exits 4 rather than return an ordinary refusal over a live
    pin."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealCommonDir = ${{function:Get-MachineLeaseCommonDir}}
$script:Calls = 0
function Get-MachineLeaseCommonDir {{
    param([Parameter(Mandatory)][string]$RepositoryRoot)
    $script:Calls += 1
    if ($script:Calls -eq 4) {{
        [LeaseNative.Kernel]::SetHandleInformation($Pins[0].Handle, 2, 2) | Out-Null
        return $null
    }}
    return (& $script:RealCommonDir -RepositoryRoot $RepositoryRoot)
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "stranded-pin"
Write-Output "RETURNED:$($null -eq $Lease)"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 4, out + result.stderr
    assert "no longer resolves a git repository" in out, out
    assert "refused the pin on" in out, out
    assert "refused the lease handle" not in out, "the lease handle closed and must not be reported"
    assert "RETURNED:" not in out, "a stranded process must not return to its caller"
    lease = _NAMESPACE["acquire"]("after-shell-refusal-strand", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_exits_stranded_when_a_partial_pinning_cannot_close_a_pin(
    anchor: Path,
) -> None:
    """The second link cannot be pinned and the first pin, protected from
    close, refuses: the refusal is reported and the process exits 4."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealOpenPin = ${{function:Open-MachineLeasePin}}
$script:Opened = 0
function Open-MachineLeasePin {{
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Role)
    $script:Opened += 1
    if ($script:Opened -eq 2) {{ throw "$Role could not be held: forced" }}
    $Pin = & $script:RealOpenPin -Path $Path -Role $Role
    [LeaseNative.Kernel]::SetHandleInformation($Pin.Handle, 2, 2) | Out-Null
    return $Pin
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "stranded-pinning"
Write-Output "RETURNED:$($null -eq $Lease)"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 4, out + result.stderr
    assert "could not be pinned" in out, out
    assert "forced" in out, out
    assert "refused the pin on" in out, out
    assert "RETURNED:" not in out, "a stranded process must not return to its caller"
    lease = _NAMESPACE["acquire"]("after-shell-pinning-strand", repo=anchor)
    _NAMESPACE["release"](lease)


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
def test_the_shell_exit_reports_a_pin_that_will_not_close(anchor: Path) -> None:
    """Exit closes the lease handle and then every pin; a pin the kernel
    refuses to close is reported and carried into a false result."""

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "exit-pin"
if (-not $Lease) {{ Write-Output "ENTER-FAILED"; exit 1 }}
[LeaseNative.Kernel]::SetHandleInformation($Lease.Binding.Pins[1].Handle, 2, 2) | Out-Null
$Result = Exit-MachineLease $Lease
Write-Output "RESULT:$Result"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "RESULT:False" in out, "a pin that would not close was reported as a clean release"
    assert "refused the pin on" in out, out
    assert "refused the lease handle" not in out, "the lease handle closed and must not be reported"
    lease = _NAMESPACE["acquire"]("after-shell-exit-pin", repo=anchor)
    _NAMESPACE["release"](lease)


def _pin_marks_refused(monkeypatch: pytest.MonkeyPatch, *, protect: bool) -> dict[str, int]:
    """Refuse the inherit mark of the first chain pin - the .git entry,
    opened read-share - and, when asked, protect that pin from close so
    its cleanup refuses too."""

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create = kernel32.CreateFileW
    real_mark = kernel32.SetHandleInformation
    taken: dict[str, int] = {}

    def create_noting_the_entry_pin(name: str, *rest: object) -> object:
        handle = real_create(name, *rest)
        if str(name).endswith(".git") and rest[1] == 1 and "pin" not in taken:
            taken["pin"] = int(handle)
        return handle

    def refusing_mark(handle: object, mask: int, flags: int) -> int:
        if int(getattr(handle, "value", handle)) == taken.get("pin"):  # type: ignore[arg-type]
            if protect:
                assert real_mark(handle, 2, 2)
            return 0
        return int(real_mark(handle, mask, flags))

    monkeypatch.setattr(kernel32, "CreateFileW", create_noting_the_entry_pin)
    monkeypatch.setattr(kernel32, "SetHandleInformation", refusing_mark)
    return taken


def test_a_pin_whose_inherit_mark_is_refused_is_let_go(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin that cannot be made to share a child's lifetime holds nothing:
    the acquisition is refused naming the link, the pin is closed, and the
    machine is free."""

    import ctypes

    taken = _pin_marks_refused(monkeypatch, protect=False)
    with pytest.raises(LeaseRefused, match="for a child's lifetime"):
        _NAMESPACE["acquire"]("unmarked-pin", repo=anchor)
    monkeypatch.undo()
    assert "pin" in taken
    assert not _NAMESPACE["_kernel32"]().GetFileInformationByHandle(
        ctypes.c_void_p(taken["pin"]), ctypes.byref(_NAMESPACE["_ByHandleFileInformation"]())
    ), "the unmarked pin stayed open"
    successor = _NAMESPACE["acquire"]("after-unmarked-pin", repo=anchor)
    _NAMESPACE["release"](successor)


def test_a_pin_whose_inherit_mark_is_refused_and_will_not_close_is_a_strand(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unmarked pin's close is refused too: the caller learns it as a
    strand naming the pin, raised from the refusal."""

    taken = _pin_marks_refused(monkeypatch, protect=True)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="a refused pinning") as caught:
        _NAMESPACE["acquire"]("unmarked-stranded-pin", repo=anchor)
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, LeaseRefused)
    assert "for a child's lifetime" in str(caught.value.__cause__)
    assert caught.value.strands == (("pin", taken["pin"], caught.value.error),)
    _unprotect_and_close(taken["pin"])
    successor = _NAMESPACE["acquire"]("after-unmarked-strand", repo=anchor)
    _NAMESPACE["release"](successor)


def _powershell_hosts() -> list[str]:
    """Every PowerShell host on this machine, PowerShell 7 first.

    Whether binding $null to a [string] parameter yields the empty string
    or removes the variable is host-dependent, so the contract has to hold
    in each host the machine has rather than in one of them.
    """

    hosts = []
    for name in ("pwsh", "powershell"):
        if shutil.which(name):
            hosts.append(name)
    return hosts


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
@pytest.mark.parametrize("host", _powershell_hosts())
def test_the_scrub_leaves_no_redirection_variable_behind(anchor: Path, host: str) -> None:
    """The resolution runs with git's redirection environment ABSENT.

    Absent, not empty: PowerShell 7 binds a null string as the empty string,
    and git reads an empty GIT_DIR as a repository path and fails. The
    control sets two variables to literal paths, clears a third, and
    requires the common directory to resolve, the lease to be acquired, the
    two set variables restored to exactly their previous values, and the
    third still absent afterwards.
    """

    set_dir = "D:/bogus-git-dir"
    set_tree = "D:/bogus-work-tree"
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$env:GIT_DIR = "{set_dir}"
$env:GIT_WORK_TREE = "{set_tree}"
Remove-Item -LiteralPath "Env:GIT_INDEX_FILE" -ErrorAction SilentlyContinue
$Common = Get-MachineLeaseCommonDir -RepositoryRoot "{anchor}"
Write-Output "COMMON:$([bool]$Common)"
Write-Output "DIR:$env:GIT_DIR"
Write-Output "TREE:$env:GIT_WORK_TREE"
Write-Output "INDEX_ABSENT:$(-not (Test-Path -LiteralPath 'Env:GIT_INDEX_FILE'))"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "scrubbed"
Write-Output "ACQUIRED:$([bool]$Lease)"
if ($Lease) {{ Exit-MachineLease $Lease | Out-Null }}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "COMMON:True" in out, f"{host} did not resolve with redirection set: {out}"
    assert "ACQUIRED:True" in out, f"{host} refused the lease with redirection set: {out}"
    assert f"DIR:{set_dir}" in out, out
    assert f"TREE:{set_tree}" in out, out
    assert "INDEX_ABSENT:True" in out, "a variable that was absent came back set: " + out


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
@pytest.mark.parametrize("host", _powershell_hosts())
def test_a_redirection_variable_that_will_not_clear_refuses_the_resolution(
    anchor: Path, host: str
) -> None:
    """A removal that does not take effect must refuse, not proceed.

    If the variable survives and still names another valid repository, git
    resolves THAT repository and the gate takes its lease. The control
    shadows the removal with a no-op, points GIT_DIR at a real second
    repository, and requires the resolution to refuse and to say why -
    never to return the other repository's directory.
    """

    other = anchor.parent / "other-repo"
    subprocess.run(["git", "init", "-q", str(other)], check=True, capture_output=True)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
function Remove-Item {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) }}
$env:GIT_DIR = "{(other / ".git").as_posix()}"
$Common = Get-MachineLeaseCommonDir -RepositoryRoot "{anchor}"
Write-Output "COMMON:[$Common]"
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = result.stdout + result.stderr
    assert "COMMON:[]" in result.stdout, (
        "the resolution proceeded with git redirection still in force and returned "
        f"a directory: {out}"
    )
    assert "could not be removed from the environment" in out, "the refusal was silent: " + out
    assert str(other.name) not in result.stdout.split("COMMON:")[-1], (
        "the other repository was accepted: " + out
    )


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
@pytest.mark.parametrize("host", _powershell_hosts())
def test_a_restoration_that_does_not_take_effect_refuses_the_resolution(
    anchor: Path, host: str
) -> None:
    """Restoring the environment is part of the answer, not a courtesy.

    A resolution that succeeds but leaves this process redirected hands the
    caller a correct directory and a poisoned environment for every later
    git command. The control lets the scrub and the resolution run, has the
    git invocation introduce a variable that was absent, and then refuses
    the removal that would take it away again; the resolution must return
    nothing rather than a usable repository.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealRemove = ${{function:Remove-Item}}
$script:Removals = 0
function Remove-Item {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $script:Removals += 1
    # The scrub's own removals are allowed; the restoration's are not.
    if ($script:Removals -le 8) {{ & $script:RealRemove @Rest }}
}}
$script:RealGit = (Get-Command git).Source
function git {{
    # A variable that was absent before the resolution appears during it.
    $env:GIT_INDEX_FILE = "D:/introduced-index"
    & $script:RealGit @args
}}
$Common = Get-MachineLeaseCommonDir -RepositoryRoot "{anchor}"
Write-Output "COMMON_RETURNED:$([bool]$Common)"
Write-Output "INDEX_REMAINS:$(Test-Path -LiteralPath 'Env:GIT_INDEX_FILE')"
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = result.stdout + result.stderr
    assert "COMMON_RETURNED:False" in result.stdout, (
        "a resolution whose restoration did not take effect returned a usable repository: " + out
    )
    assert "could not be removed again" in out, "the restoration failure was silent: " + out


@pytest.mark.skipif(os.name != "nt", reason="the gate's shell runs on Windows")
@pytest.mark.parametrize("host", _powershell_hosts())
def test_a_present_variable_is_restored_to_its_exact_value(anchor: Path, host: str) -> None:
    """The other restoration branch: a variable that had a value gets that
    exact value back, and the resolution is only returned when it did."""

    value = "D:/exact-value-git-dir"
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$env:GIT_DIR = "{value}"
$Common = Get-MachineLeaseCommonDir -RepositoryRoot "{anchor}"
Write-Output "COMMON_RETURNED:$([bool]$Common)"
Write-Output "EXACT:$($env:GIT_DIR -ceq "{value}")"
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = result.stdout + result.stderr
    assert "COMMON_RETURNED:True" in result.stdout, out
    assert "EXACT:True" in result.stdout, "the value came back changed: " + out


@pytest.mark.parametrize("host", _powershell_hosts())
def test_pytest_scratch_is_held_outside_the_repository(
    anchor: Path, tmp_path: Path, host: str
) -> None:
    target = tmp_path / "available-scratch"
    target.mkdir()
    assert _write_example_config(target, "before").returncode == 0
    fallback = tmp_path / "fallback-scratch"
    fallback.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
. "{ROOT / "scripts" / "held-pytest-scratch.ps1"}"
$env:RUNNER_TEMP = "{target}"
$env:TEMP = "{fallback}"
$env:TMP = "{fallback}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "scratch-test"
if (-not $Lease) {{ exit 2 }}
try {{
    $Scratch = New-HeldPytestScratch -RepositoryRoot "{anchor}" -Lease $Lease
    & git config --file "{target / "example.config"}" example.value during
    Write-Output "CONFIG-EXIT:$LASTEXITCODE"
    New-Item -ItemType Directory -Path $Scratch -ErrorAction Stop | Out-Null
    Set-Content -LiteralPath (Join-Path $Scratch "write.txt") -Value "ok"
    Write-Output "SCRATCH:$Scratch"
    Write-Output "INSIDE:$(Test-PytestScratchContainedBy -Path $Scratch -Root "{anchor}")"
    Write-Output "RUNNER:$(Test-PytestScratchContainedBy -Path $Scratch -Root "{target}")"
    Write-Output "EXISTS:$(Test-Path -LiteralPath $Scratch)"
    Write-Output "USABLE:$(Test-Path -LiteralPath (Join-Path $Scratch "write.txt"))"
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 3 }}
}}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert _write_example_config(target, "after").returncode == 0
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "INSIDE:False" in result.stdout, out
    assert "RUNNER:True" in result.stdout, out
    assert "EXISTS:True" in result.stdout, out
    assert "USABLE:True" in result.stdout, out
    assert "CONFIG-EXIT:0" in result.stdout, out


@pytest.mark.parametrize("host", _powershell_hosts())
def test_pytest_scratch_refuses_a_root_that_reaches_the_repository(
    anchor: Path, tmp_path: Path, host: str
) -> None:
    target = anchor / "scratch-target"
    target.mkdir()
    link = tmp_path / "scratch-link"
    _junction(link, target)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
. "{ROOT / "scripts" / "held-pytest-scratch.ps1"}"
$env:RUNNER_TEMP = ""
$env:TEMP = "{link}"
$env:TMP = "{link}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "scratch-refusal"
if (-not $Lease) {{ exit 2 }}
try {{
    try {{
        New-HeldPytestScratch -RepositoryRoot "{anchor}" -Lease $Lease | Out-Null
        Write-Output "ACCEPTED"
    }} catch {{
        Write-Output "REFUSED:$($_.Exception.Message)"
    }}
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 3 }}
}}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "REFUSED:The pytest scratch root resolves inside the repository." in out
    assert "ACCEPTED" not in out


@pytest.mark.parametrize("host", _powershell_hosts())
def test_pytest_scratch_holds_every_link_to_its_external_root(
    anchor: Path, tmp_path: Path, host: str
) -> None:
    target = tmp_path / "external-scratch"
    target.mkdir()
    link = tmp_path / "external-link"
    _junction(link, target)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
. "{ROOT / "scripts" / "held-pytest-scratch.ps1"}"
$env:RUNNER_TEMP = ""
$env:TEMP = "{link}"
$env:TMP = "{link}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "scratch-link"
if (-not $Lease) {{ exit 2 }}
try {{
    $Scratch = New-HeldPytestScratch -RepositoryRoot "{anchor}" -Lease $Lease
    try {{
        Remove-Item -LiteralPath "{link}" -Force -ErrorAction Stop
        Write-Output "MOVED-WHILE-HELD"
    }} catch {{
        Write-Output "PINNED"
    }}
    try {{
        Rename-Item -LiteralPath "{target}" -NewName "external-moved" -ErrorAction Stop
        Write-Output "ROOT-MOVED-WHILE-HELD"
    }} catch {{
        Write-Output "ROOT-PINNED"
    }}
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 3 }}
}}
Write-Output "RELEASED"
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "PINNED" in result.stdout, out
    assert "MOVED-WHILE-HELD" not in result.stdout, out
    assert "ROOT-PINNED" in result.stdout, out
    assert "ROOT-MOVED-WHILE-HELD" not in result.stdout, out
    assert "RELEASED" in result.stdout, out
    os.rmdir(link)
    assert not link.exists()


def test_gate_uses_the_held_external_pytest_scratch() -> None:
    source = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")
    helper_at = source.index("held-pytest-scratch.ps1")
    select_at = source.index("New-HeldPytestScratch")
    api_at = source.index('Invoke-Checked "API tests"')
    assert helper_at < select_at < api_at
    assert 'Join-Path $RepositoryRoot "temp"' not in source


def _second_pin_mark_refused(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Refuse the SECOND chain pin's inherit mark and protect both pins.

    The first pin is protected from close as it is opened, so the chain
    opener's own cleanup is refused; the second is protected when its mark is
    refused, so the pin opener's cleanup is refused as well. Protection is
    HANDLE_FLAG_PROTECT_FROM_CLOSE, a different bit from the inherit flag the
    opener sets under mask 0x1, so marking a protected handle inheritable
    still works.
    """

    import ctypes

    kernel32 = _NAMESPACE["_kernel32"]()
    real_create = kernel32.CreateFileW
    real_mark = kernel32.SetHandleInformation
    order: list[int] = []
    taken: dict[str, int] = {}

    def create_noting_pins(name: str, *rest: object) -> object:
        handle = real_create(name, *rest)
        # Pins are the only read-share opens: _SHARE_READ is 1.
        number = 0 if handle is None else int(handle)
        if rest[1] == 1 and number and number != _NAMESPACE["_INVALID_HANDLE"]:
            order.append(number)
            if len(order) == 1:
                taken["outer"] = number
                assert real_mark(ctypes.c_void_p(number), 2, 2)
            elif len(order) == 2:
                taken["inner"] = number
        return handle

    def refusing_mark(handle: object, mask: int, flags: int) -> int:
        number = int(getattr(handle, "value", handle))  # type: ignore[arg-type]
        if number == taken.get("inner") and mask == 1:
            assert real_mark(ctypes.c_void_p(number), 2, 2)
            return 0
        return int(real_mark(handle, mask, flags))

    monkeypatch.setattr(kernel32, "CreateFileW", create_noting_pins)
    monkeypatch.setattr(kernel32, "SetHandleInformation", refusing_mark)
    return taken


def test_a_nested_pin_refusal_joins_the_outer_report(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One report names both refused closes, not one plus a hidden cause.

    The pin opener lets go of the pin it has just taken and its close is
    refused; the chain opener then lets go of the earlier pin and that close
    is refused too. Before this, the pin opener's report reached the chain
    opener as an exception and became `__cause__` of the outer one - so the
    outer report named the earlier pin only, and the new pin survived
    somewhere no command line prints. The design promises ONE report naming
    every acquired object whose close was refused.
    """

    taken = _second_pin_mark_refused(monkeypatch)
    with pytest.raises(_NAMESPACE["LeaseStranded"], match="every refused close") as caught:
        _NAMESPACE["acquire"]("nested-pin-strands", repo=anchor)
    monkeypatch.undo()
    assert "inner" in taken and "outer" in taken, "the two pins were never opened"
    numbers = [number for _kind, number, _error in caught.value.strands]
    # The nested cleanup ran first, so it is reported first.
    assert numbers == [taken["inner"], taken["outer"]], caught.value.strands
    assert all(kind == "pin" for kind, _number, _error in caught.value.strands)
    # The failure that aborted the acquisition is still the cause, not the
    # nested report: a report is not a reason.
    assert isinstance(caught.value.__cause__, LeaseRefused)
    assert "for a child's lifetime" in str(caught.value.__cause__)
    _unprotect_and_close(taken["inner"])
    _unprotect_and_close(taken["outer"])
    successor = _NAMESPACE["acquire"]("after-nested-pin-strands", repo=anchor)
    _NAMESPACE["release"](successor)


def test_the_shell_pin_opener_hands_its_outcome_back_instead_of_exiting(
    anchor: Path,
) -> None:
    """The shell's inner helper must not exit before the outer closer runs.

    PowerShell cannot be made to refuse a real `SetHandleInformation`: the
    P/Invoke lives on a compiled type, so unlike the Python control above this
    one reads the source rather than driving the kernel. It is here because
    the behaviour it guards is invisible in any run that succeeds - an `exit`
    inside `Open-MachineLeasePin` leaves the caller's earlier pins unattempted
    and unreported, and nothing else in this file would notice it coming back.
    """

    source = (ROOT / "scripts" / "machine-lease.ps1").read_text(encoding="utf-8")
    opener = source[source.index("function Open-MachineLeasePin") :]
    opener = opener[: opener.index("function Close-MachineLeaseAcquired")]
    assert "exit 4" not in opener, (
        "the pin opener exits before its caller can close the earlier pins"
    )
    assert "$script:MachineLeaseStranded = $true" in opener, (
        "the pin opener must hand its refused close back to the caller"
    )
    assert "$script:MachineLeaseStranded = $false" in source, (
        "a stale flag from an earlier acquisition would exit a healthy one"
    )
    assert "if (-not $Closed -or $script:MachineLeaseStranded) {" in source, (
        "the chain opener must close its own pins and then consult the flag"
    )


def test_a_signal_file_is_read_once_it_is_written_not_once_it_appears() -> None:
    """A file that exists but is still held must be waited for, not read.

    This is the failure that removed a candidate from the merge queue. The
    shell writes its signal with `Set-Content`, which creates the file and then
    writes and closes it; a reader in between gets errno 13 on Windows. Polling
    `exists()` returns as soon as the path appears, which is before the content
    is there.

    The fake reproduces exactly that window - two refusals, then an empty read,
    then the content - without depending on the timing that made it rare.
    """

    attempts: list[str] = []

    class HeldFile:
        name = "held.txt"

        def read_text(self, encoding: str = "utf-8") -> str:
            attempts.append(encoding)
            if len(attempts) <= 2:
                raise PermissionError(13, "the writer still holds it")
            if len(attempts) == 3:
                return ""
            return "settled content"

    text = _settled_text(
        HeldFile(),  # type: ignore[arg-type]
        deadline=time.monotonic() + 30,
        still_running=lambda: None,
    )
    assert text == "settled content"
    assert len(attempts) == 4, attempts


@pytest.mark.skipif(os.name != "nt", reason="the lease helper is PowerShell")
def test_a_culturally_equal_restored_value_still_refuses_the_resolution(
    anchor: Path,
) -> None:
    """Case is not the only way two different strings compare as one.

    PowerShell's comparison operators are cultural, `-cne` included: it is
    case-sensitive but still asks the culture, and the culture says "Strasse"
    and the eszett spelling are the same string. They are not the same bytes,
    so a value restored as one when it was saved as the other is not the
    environment this process was handed - and a case-only control cannot see
    that, because the case is identical.

    Same seam as the case control, for the same reason: the real setter
    round-trips exactly, so only a shadowed read can produce a value that
    differs from what was saved.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$env:GIT_DIR = "C:\Repository\Strasse\.git"
$global:LeaseReads = 0
function Read-MachineLeaseEnvironment {{
    param([Parameter(Mandatory = $true)][string] $Name)
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ($Name -eq "GIT_DIR" -and $Value) {{
        $global:LeaseReads++
        if ($global:LeaseReads -gt 1) {{ return $Value.Replace("Strasse", "Straße") }}
    }}
    return $Value
}}
try {{
    $Common = Get-MachineLeaseCommonDir -RepositoryRoot "{anchor}"
    if ($null -ne $Common) {{ Write-Output "RESOLVED-ANYWAY"; exit 2 }}
    Write-Output "REFUSED"
}} finally {{
    Remove-Item -LiteralPath "Env:GIT_DIR" -ErrorAction SilentlyContinue
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESOLVED-ANYWAY" not in result.stdout, (
        "a value the culture calls equal but the bytes do not was accepted as restored"
    )
    assert "REFUSED" in result.stdout
    assert "could not be restored to the value it had" in result.stdout, (
        "the refusal did not name what went wrong"
    )


@pytest.mark.skipif(os.name != "nt", reason="the lease helper is PowerShell")
def test_a_restored_value_that_differs_only_in_case_refuses_the_resolution(
    anchor: Path,
) -> None:
    """The guard is about the environment being what it was, not about paths.

    PowerShell's `-ne` is case-insensitive, so a value coming back with
    changed case was accepted as restored and the resolution was handed to the
    caller. Every scrubbed name is path-valued and a path differing only in
    case names the same object on Windows, so no caller was pointed at the
    wrong repository by it - but the guard did not hold the property it exists
    for.

    The branch is unreachable without the seam: the real setter round-trips a
    value exactly, so nothing a test can do to the environment produces a
    case-altered read. `Read-MachineLeaseEnvironment` is shadowed here to
    return the value uppercased on the SECOND read of one variable - the save
    reads it first, the restoration check reads it second - which is the one
    thing the production path cannot produce on its own.
    """

    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$env:GIT_DIR = "C:\\NotThisRepository\\.git"
$global:LeaseReads = 0
function Read-MachineLeaseEnvironment {{
    param([Parameter(Mandatory = $true)][string] $Name)
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ($Name -eq "GIT_DIR" -and $Value) {{
        $global:LeaseReads++
        if ($global:LeaseReads -gt 1) {{ return $Value.ToUpperInvariant() }}
    }}
    return $Value
}}
try {{
    $Common = Get-MachineLeaseCommonDir -RepositoryRoot "{anchor}"
    if ($null -ne $Common) {{ Write-Output "RESOLVED-ANYWAY"; exit 2 }}
    Write-Output "REFUSED"
}} finally {{
    Remove-Item -LiteralPath "Env:GIT_DIR" -ErrorAction SilentlyContinue
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESOLVED-ANYWAY" not in result.stdout, (
        "a value that came back with changed case was accepted as restored"
    )
    assert "REFUSED" in result.stdout
    assert "could not be restored to the value it had" in result.stdout, (
        "the refusal did not name what went wrong"
    )


def test_the_python_lease_never_mutates_the_process_environment() -> None:
    """The other half of the same question, and its answer is different.

    The row asks whether `machine_lock.py` shares the inexact comparison. It
    cannot: it never restores anything. It builds a scrubbed COPY of the
    environment and hands that to the subprocess, so the process it runs in is
    never redirected and there is nothing to compare on the way back.

    Checked here rather than asserted in prose, so that a later change which
    starts mutating the real environment has to notice this.
    """

    source = (ROOT / "scripts" / "machine_lock.py").read_text(encoding="utf-8")

    assert "_scrubbed_git_env" in source
    assert "environment = dict(os.environ)" in source, (
        "the scrub no longer works on a copy of the environment"
    )
    for mutation in ("os.environ[", "os.environ.pop(", "os.putenv", "setdefault("):
        assert mutation not in source, (
            f"machine_lock.py now writes the process environment through {mutation!r}, "
            "so it needs the exact restoration comparison the shell helper has"
        )


@pytest.mark.parametrize(
    ("operation", "protected_open"),
    [("identity", 1), ("acquire", 1), ("acquire", 2), ("binding", 1), ("binding", 2)],
)
def test_temporary_directory_close_refusal_is_reported(
    anchor: Path, monkeypatch: pytest.MonkeyPatch, operation: str, protected_open: int
) -> None:
    """A temporary probe must not silently survive a successful operation."""
    module = _NAMESPACE["acquire"].__globals__
    original_open = module["_open_directory"]
    original_close = module["_close"]
    held = _NAMESPACE["acquire"]("binding-probe", repo=anchor) if operation == "binding" else None
    returned = None
    opened: list[int] = []
    protected: list[int] = []
    attempts: list[int] = []

    def open_probe(path: Path) -> int:
        handle = original_open(path)
        opened.append(handle)
        if len(opened) == protected_open:
            _protect(handle)
            protected.append(handle)
        return handle

    def close_probe(handle: int) -> bool:
        if handle in protected:
            attempts.append(handle)
        return bool(original_close(handle))

    failure: BaseException | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setitem(module, "_open_directory", open_probe)
            patch.setitem(module, "_close", close_probe)
            try:
                if operation == "identity":
                    module["_directory_identity"](anchor)
                elif operation == "binding":
                    held.assert_bound()
                else:
                    returned = _NAMESPACE["acquire"]("probe-refusal", repo=anchor)
            except _NAMESPACE["LeaseStranded"] as refusal:
                failure = refusal
        if returned is not None:
            _NAMESPACE["release"](returned)
        if held is not None:
            _NAMESPACE["release"](held)
            held = None
        assert len(protected) == 1
        assert isinstance(failure, _NAMESPACE["LeaseStranded"]), "temporary close refusal was lost"
        assert [(kind, number) for kind, number, _error in failure.strands] == [
            ("directory-probe", protected[0])
        ]
        assert attempts == protected, "a refused handle must be attempted exactly once"
        successor = _NAMESPACE["acquire"]("after-temporary-probe-refusal", repo=anchor)
        _NAMESPACE["release"](successor)
    finally:
        if held is not None:
            _NAMESPACE["release"](held)
        for handle in protected:
            _unprotect_and_close(handle)


def test_directory_probe_close_refusal_keeps_the_identity_error(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _NAMESPACE["acquire"].__globals__
    original_open = module["_open_directory"]
    protected: list[int] = []
    primary = LeaseRefused("constructed identity refusal")

    def open_probe(path: Path) -> int:
        handle = original_open(path)
        _protect(handle)
        protected.append(handle)
        return handle

    def refuse_identity(handle: int) -> tuple[int, int, int]:
        raise primary

    failure: BaseException | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setitem(module, "_open_directory", open_probe)
            patch.setitem(module, "_identity", refuse_identity)
            try:
                module["_directory_identity"](anchor)
            except BaseException as refusal:
                failure = refusal
        assert isinstance(failure, _NAMESPACE["LeaseStranded"]), "cleanup refusal was discarded"
        assert failure.__cause__ is primary
        assert [(kind, number) for kind, number, _error in failure.strands] == [
            ("directory-probe", protected[0])
        ]
    finally:
        for handle in protected:
            _unprotect_and_close(handle)


@pytest.mark.parametrize("protected_opens", [(1,), (2,), (3,), (1, 2, 3)])
@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_acquisition_reports_every_temporary_probe_close_refusal(
    anchor: Path, host: str, protected_opens: tuple[int, ...]
) -> None:
    """Real protected handles cannot be returned as an ordinary acquisition."""
    selected = ",".join(str(number) for number in protected_opens)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealDirectory = (Get-Command Open-MachineLeaseDirectory).ScriptBlock
$script:ProbeCount = 0
$script:ProtectedProbes = @()
function Open-MachineLeaseDirectory {{
    param([string]$Path)
    $Handle = & $script:RealDirectory -Path $Path
    $script:ProbeCount++
    if (@({selected}) -contains $script:ProbeCount) {{
        if (-not [LeaseNative.Kernel]::SetHandleInformation($Handle, 2, 2)) {{
            throw "could not protect constructed probe"
        }}
        $script:ProtectedProbes += $Handle
        Write-Host "PROTECTED:$Handle"
    }}
    return $Handle
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "temporary-probe"
Write-Output "RETURNED"
if ($Lease) {{ Exit-MachineLease $Lease | Out-Null }}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout + result.stderr
    protected = [
        line.removeprefix("PROTECTED:")
        for line in out.splitlines()
        if line.startswith("PROTECTED:")
    ]
    assert protected, out
    assert result.returncode == 4, out
    assert "RETURNED" not in out, "the helper returned despite a refused temporary close"
    for number in protected:
        assert f"refused the directory probe {number} " in out, out
    successor = _NAMESPACE["acquire"]("after-shell-temporary-probes", repo=anchor)
    _NAMESPACE["release"](successor)


def test_acquisition_reports_temporary_probes_and_pins_in_one_failure(
    anchor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _NAMESPACE["acquire"].__globals__
    original_open = module["_open_directory"]
    original_pin = module["_open_pin"]
    original_common = module["_common_dir"]
    original_close = module["_close"]
    protected: list[tuple[str, int]] = []
    opens = 0
    resolutions = 0
    closes: list[int] = []
    primary = LeaseRefused("constructed final resolution refusal")

    def open_probe(path: Path) -> int:
        nonlocal opens
        handle = original_open(path)
        opens += 1
        if opens <= 2:
            _protect(handle)
            protected.append(("directory-probe", handle))
        return handle

    def open_pin(path: Path, role: str) -> object:
        pin = original_pin(path, role)
        if role == "the repository's .git entry" and not any(
            kind == "pin" for kind, _number in protected
        ):
            _protect(pin.handle)
            protected.append(("pin", pin.handle))
        return pin

    def resolve_common(repo: Path | None) -> Path:
        nonlocal resolutions
        resolutions += 1
        if resolutions == 3:
            raise primary
        return original_common(repo)

    def close_handle(handle: int) -> bool:
        if any(number == handle for _kind, number in protected):
            closes.append(handle)
        return bool(original_close(handle))

    failure: BaseException | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setitem(module, "_open_directory", open_probe)
            patch.setitem(module, "_open_pin", open_pin)
            patch.setitem(module, "_common_dir", resolve_common)
            patch.setitem(module, "_close", close_handle)
            try:
                _NAMESPACE["acquire"]("several-temporary-refusals", repo=anchor)
            except BaseException as refusal:
                failure = refusal
        assert len(protected) == 3
        assert isinstance(failure, _NAMESPACE["LeaseStranded"])
        assert failure.__cause__ is primary
        assert sorted((kind, number) for kind, number, _error in failure.strands) == sorted(
            protected
        ), "the report omitted a temporary probe or pin"
        assert sorted(closes) == sorted(number for _kind, number in protected)
    finally:
        for _kind, handle in protected:
            _unprotect_and_close(handle)
    successor = _NAMESPACE["acquire"]("after-several-temporary-refusals", repo=anchor)
    _NAMESPACE["release"](successor)


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_directory_probe_preserves_the_primary_failure(anchor: Path, host: str) -> None:
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealDirectory = (Get-Command Open-MachineLeaseDirectory).ScriptBlock
function Open-MachineLeaseDirectory {{
    param([string]$Path)
    $Handle = & $script:RealDirectory -Path $Path
    if (-not [LeaseNative.Kernel]::SetHandleInformation($Handle, 2, 2)) {{
        throw "could not protect constructed probe"
    }}
    Write-Host "PROTECTED:$Handle"
    return $Handle
}}
function Get-MachineLeaseIdentity {{
    param($Handle)
    throw [System.InvalidOperationException]::new("constructed identity refusal")
}}
try {{
    Get-MachineLeaseDirectoryIdentity -Path "{anchor}" | Out-Null
    Write-Output "RETURNED"
}} catch {{
    Write-Output "CAUGHT:$($_.Exception.ToString())"
}}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "RETURNED" not in out
    assert "CAUGHT:" in out and "constructed identity refusal" in out
    assert "refused the directory probe " in out, "cleanup lost its own failure"


@pytest.mark.parametrize("boundary", ["enumeration", "kernel_open"])
def test_a_new_junction_cannot_leave_the_lease_mapping_unheld(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """The same final identity must not hide a newly removable path mapping."""
    linked, _other, pointer, _original = _linked_worktree(anchor, tmp_path)
    private = Path(pointer.read_text(encoding="utf-8").removeprefix("gitdir:").strip())
    chosen = private.parent if boundary == "enumeration" else private
    moved = chosen.with_name(chosen.name + "-held")
    module = _NAMESPACE["acquire"].__globals__
    real_chain = module["_resolution_chain"]
    kernel = module["_kernel32"]()
    real_create = kernel.CreateFileW
    attempted = False
    switched = False
    protected = False
    lease = None

    def switch_to_junction() -> None:
        nonlocal attempted, switched, protected
        attempted = True
        identity = module["_directory_identity"](chosen)
        try:
            chosen.rename(moved)
        except PermissionError:
            protected = True
            return
        try:
            _junction(chosen, moved)
        except BaseException:
            if chosen.exists():
                os.rmdir(chosen)
            moved.rename(chosen)
            raise
        switched = True
        assert module["_directory_identity"](chosen) == identity

    def after_enumeration(path: Path) -> list[tuple[Path, str]]:
        chain = real_chain(path)
        if not attempted:
            switch_to_junction()
        return chain

    def before_kernel_open(name: str, *rest: object) -> object:
        if not attempted and Path(name) == chosen and rest[1] == module["_SHARE_READ"]:
            switch_to_junction()
        return real_create(name, *rest)

    try:
        with monkeypatch.context() as patch:
            if boundary == "enumeration":
                patch.setitem(module, "_resolution_chain", after_enumeration)
            else:
                patch.setattr(kernel, "CreateFileW", before_kernel_open)
            try:
                lease = _NAMESPACE["acquire"]("new-junction-mapping", repo=linked)
            except LeaseRefused:
                protected = True
        assert attempted, "the constructed rename boundary was not reached"
        if lease is not None and switched:
            lease.assert_bound()
            assert chosen.is_junction()
            try:
                os.rmdir(chosen)
            except PermissionError:
                protected = True
        assert protected, "acquisition accepted a junction that remained removable"
    finally:
        if lease is not None:
            _NAMESPACE["release"](lease)
        if switched:
            if chosen.is_junction():
                os.rmdir(chosen)
            assert not chosen.exists()
            moved.rename(chosen)


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_pinning_protects_a_junction_added_after_enumeration(
    anchor: Path, tmp_path: Path, host: str
) -> None:
    linked, _other, pointer, _original = _linked_worktree(anchor, tmp_path)
    private = Path(pointer.read_text(encoding="utf-8").removeprefix("gitdir:").strip())
    chosen = private.parent
    moved = chosen.with_name(chosen.name + "-held")
    switch = (
        "import os,sys,_winapi;"
        "exec('try:\\n os.rename(sys.argv[1],sys.argv[2])\\n"
        "except PermissionError:\\n sys.exit(73)');"
        "_winapi.CreateJunction(sys.argv[2],sys.argv[1])"
    )
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:RealChain = (Get-Command Get-MachineLeaseResolutionChain).ScriptBlock
$script:Attempted = $false
function Get-MachineLeaseResolutionChain {{
    param([string]$Anchor, $Pins)
    if ($PSBoundParameters.ContainsKey("Pins")) {{
        $Links = @(& $script:RealChain -Anchor $Anchor -Pins $Pins)
    }} else {{
        $Links = @(& $script:RealChain -Anchor $Anchor)
    }}
    if (-not $script:Attempted) {{
        $script:Attempted = $true
        $Before = Get-MachineLeaseDirectoryIdentity -Path "{chosen}"
        & "{sys.executable}" -c "{switch}" "{chosen}" "{moved}"
        if ($LASTEXITCODE -eq 73) {{ Write-Host "SWITCH-PROTECTED"; return $Links }}
        if ($LASTEXITCODE -ne 0) {{ throw "constructed junction switch failed" }}
        $After = Get-MachineLeaseDirectoryIdentity -Path "{chosen}"
        if ($Before -ne $After) {{ throw "constructed directory identity changed" }}
        Write-Host "SWITCHED"
    }}
    return $Links
}}
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "new-junction-mapping"
if (-not $Lease) {{ Write-Output "REFUSED"; exit 0 }}
try {{
    Assert-MachineLeaseHeld -Lease $Lease
    try {{
        [IO.Directory]::Delete("{chosen}")
        Write-Output "MAPPING-REMOVED"
    }} catch [IO.IOException] {{
        Write-Output "MAPPING-PROTECTED"
    }} catch [UnauthorizedAccessException] {{
        Write-Output "MAPPING-PROTECTED"
    }}
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 4 }}
}}
"""
    try:
        result = subprocess.run(
            [host, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        out = result.stdout + result.stderr
        assert result.returncode == 0, out
        assert "SWITCHED" in out or "SWITCH-PROTECTED" in out, out
        assert "REFUSED" in out or "MAPPING-PROTECTED" in out, out
        assert "MAPPING-REMOVED" not in out, "the accepted mapping remained removable"
    finally:
        if chosen.is_junction():
            os.rmdir(chosen)
        if moved.exists():
            assert not chosen.exists()
            moved.rename(chosen)


def _write_example_config(directory: Path, value: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "config", "--file", str(directory / "example.config"), "example.value", value],
        text=True,
        capture_output=True,
        check=False,
    )


def test_holding_ancestors_allows_atomic_child_replacement(anchor: Path) -> None:
    assert _write_example_config(anchor, "before").returncode == 0
    lease = _NAMESPACE["acquire"]("atomic-child-replacement", repo=anchor)
    try:
        lease.assert_bound()
        during = _write_example_config(anchor, "during")
        with pytest.raises(PermissionError):
            anchor.rename(anchor.with_name("moved-repository"))
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)
    assert _write_example_config(anchor, "after").returncode == 0
    assert during.returncode == 0, during.stderr


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_holding_ancestors_allows_atomic_child_replacement(anchor: Path, host: str) -> None:
    assert _write_example_config(anchor, "before").returncode == 0
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "atomic-child-replacement"
if (-not $Lease) {{ throw "lease was not acquired" }}
try {{
    & git config --file "{anchor / "example.config"}" example.value during
    Write-Output "CONFIG-EXIT:$LASTEXITCODE"
    try {{
        [IO.Directory]::Move("{anchor}", "{anchor.with_name("moved-repository")}")
        throw "the held repository could be moved"
    }} catch [IO.IOException] {{
        Write-Output "NAME-HELD"
    }}
    Assert-MachineLeaseHeld -Lease $Lease
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 4 }}
}}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False
    )
    assert _write_example_config(anchor, "after").returncode == 0
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NAME-HELD" in result.stdout
    assert "CONFIG-EXIT:0" in result.stdout, result.stdout + result.stderr


def test_an_ordinary_component_is_held_before_its_attributes_are_read(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked, _other, pointer, _original = _linked_worktree(anchor, tmp_path)
    private = Path(pointer.read_text(encoding="utf-8").removeprefix("gitdir:").strip())
    chosen = private.parent
    moved = chosen.with_name(chosen.name + "-during-read")
    module = _NAMESPACE["acquire"].__globals__
    original = module["_attributes"]
    attempts: list[bool] = []
    open_pin = module["_open_pin"]
    held: set[Path] = set()
    inspected: set[Path] = set()

    def pin(path: Path, role: str) -> object:
        value = open_pin(path, role)
        held.add(path)
        return value

    def read_attributes(path: Path) -> int:
        assert path in held, "a component was inspected before its own pin opened"
        inspected.add(path)
        if path == chosen and not attempts:
            try:
                chosen.rename(moved)
            except PermissionError:
                attempts.append(False)
            else:
                attempts.append(True)
                moved.rename(chosen)
        return original(path)

    monkeypatch.setitem(module, "_open_pin", pin)
    monkeypatch.setitem(module, "_attributes", read_attributes)
    lease = _NAMESPACE["acquire"]("hold-before-reading", repo=linked)
    try:
        lease.assert_bound()
        assert {linked, pointer, private, private / "commondir", anchor / ".git"} <= inspected
        assert attempts == [False], "the component could move at its attributes-read boundary"
    finally:
        _NAMESPACE["release"](lease)


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_holds_an_ordinary_component_before_reading_it(
    anchor: Path, tmp_path: Path, host: str
) -> None:
    linked, _other, pointer, _original = _linked_worktree(anchor, tmp_path)
    private = Path(pointer.read_text(encoding="utf-8").removeprefix("gitdir:").strip())
    chosen = private.parent
    moved = chosen.with_name(chosen.name + "-during-read")
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:Attempted = $false
function Get-Item {{
    [CmdletBinding()]
    param([string]$LiteralPath, [switch]$Force)
    if ($LiteralPath -eq "{chosen}" -and -not $script:Attempted) {{
        $script:Attempted = $true
        $Moved = $false
        try {{
            [IO.Directory]::Move("{chosen}", "{moved}")
            $Moved = $true
        }} catch [IO.IOException] {{
            Write-Host "PROBE-HELD"
        }}
        if ($Moved) {{
            [IO.Directory]::Move("{moved}", "{chosen}")
            Write-Host "PROBE-UNHELD"
        }}
    }}
    Microsoft.PowerShell.Management\\Get-Item -LiteralPath $LiteralPath -Force:$Force
}}
$Lease = Enter-MachineLease -RepositoryRoot "{linked}" -Purpose "hold-before-reading"
if (-not $Lease) {{ throw "lease was not acquired" }}
try {{
    Assert-MachineLeaseHeld -Lease $Lease
    if (-not $script:Attempted) {{ throw "the attributes read never ran" }}
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 4 }}
}}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROBE-HELD" in result.stdout and "PROBE-UNHELD" not in result.stdout


def _set_junction_in_place(directory: Path, target: Path) -> tuple[bool, bool, int]:
    import ctypes
    import struct

    kernel = _NAMESPACE["_kernel32"]()
    handle = kernel.CreateFileW(str(directory), 0x40000080, 7, None, 3, 0x02200000, None)
    if handle is None or handle == _NAMESPACE["_INVALID_HANDLE"]:
        return False, False, ctypes.get_last_error()
    substitute = ("\\??\\" + str(target)).encode("utf-16-le")
    display = str(target).encode("utf-16-le")
    names = substitute + b"\0\0" + display + b"\0\0"
    payload = (
        struct.pack(
            "<IHHHHHH",
            0xA0000003,
            8 + len(names),
            0,
            0,
            len(substitute),
            len(substitute) + 2,
            len(display),
        )
        + names
    )
    buffer = ctypes.create_string_buffer(payload)
    returned = ctypes.c_uint32()
    kernel.DeviceIoControl.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel.DeviceIoControl.restype = ctypes.c_int
    try:
        success = bool(
            kernel.DeviceIoControl(
                ctypes.c_void_p(handle),
                0x900A4,
                buffer,
                len(payload),
                None,
                0,
                ctypes.byref(returned),
                None,
            )
        )
        error = ctypes.get_last_error()
        return True, success, 0 if success else error
    finally:
        assert kernel.CloseHandle(ctypes.c_void_p(handle))


def test_a_writable_parent_cannot_be_retargeted_with_its_child_held(
    anchor: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-directory"
    target = tmp_path / "junction-target"
    empty.mkdir()
    target.mkdir()
    # Positive control: use the exact same write open and FSCTL on an empty directory.
    try:
        assert _set_junction_in_place(empty, target) == (True, True, 0)
        assert empty.is_junction()
    finally:
        os.rmdir(empty)
    lease = _NAMESPACE["acquire"]("nonempty-parent", repo=anchor)
    try:
        opened, changed, error = _set_junction_in_place(anchor, target)
        assert opened, f"ordinary parent write access was refused: {error}"
        assert not changed and error == 145, "the pinned child did not keep its parent ordinary"
        with pytest.raises(PermissionError):
            (anchor / ".git").rename(anchor / "moved-git-directory")
        assert not anchor.is_junction()
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)


@pytest.mark.parametrize("location", ["repository", "common"])
def test_a_refused_parent_pin_retirement_closes_the_replacement_once(
    anchor: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    import ctypes

    chosen = anchor if location == "repository" else anchor / ".git"
    module = _NAMESPACE["acquire"].__globals__
    original_pin = module["_open_pin"]
    original_close = module["_close"]
    kernel = module["_kernel32"]()
    original_create = kernel.CreateFileW
    taken: list[int] = []
    opened: list[int] = []
    closes: list[int] = []
    lease = None
    failure = None

    def create(name: str, *arguments: object) -> object:
        handle = original_create(name, *arguments)
        if Path(name) == chosen and int(arguments[0]) & 0x80000000:
            assert handle is not None and handle != module["_INVALID_HANDLE"]
            opened.append(int(handle))
        return handle

    def pin(path: Path, role: str) -> object:
        value = original_pin(path, role)
        if path == chosen and not taken:
            taken.append(value.handle)
            _protect(value.handle)
        return value

    def close(handle: int) -> bool:
        if handle in taken:
            closes.append(handle)
        return bool(original_close(handle))

    try:
        with monkeypatch.context() as patch:
            patch.setattr(kernel, "CreateFileW", create)
            patch.setitem(module, "_open_pin", pin)
            patch.setitem(module, "_close", close)
            try:
                lease = module["acquire"]("refused-parent-retirement", repo=anchor)
            except BaseException as exc:
                failure = exc
        assert isinstance(failure, module["LeaseStranded"])
        assert failure.strands == (("pin", taken[0], failure.error),)
        assert "strict parent hold could not be retired" in str(failure.__cause__)
        assert closes == taken, "cleanup attempted the refused old handle twice"
        assert len(opened) == (2 if location == "repository" else 3)
        assert opened[0] == taken[0]
        for handle in opened[1:]:
            assert not kernel.GetFileInformationByHandle(
                ctypes.c_void_p(handle), ctypes.byref(module["_ByHandleFileInformation"]())
            ), "a replacement hold leaked when old-pin retirement failed"
    finally:
        if taken:
            assert kernel.SetHandleInformation(ctypes.c_void_p(taken[0]), 2, 0)
            if lease is None:
                assert kernel.CloseHandle(ctypes.c_void_p(taken[0]))
        if lease is not None:
            module["release"](lease)
    successor = module["acquire"]("after-parent-retirement-refusal", repo=anchor)
    module["release"](successor)


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_reports_a_refused_parent_pin_retirement(anchor: Path, host: str) -> None:
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$script:OriginalPin = (Get-Command Open-MachineLeasePin).ScriptBlock
$script:ProtectedParent = $false
function Open-MachineLeasePin {{
    param([string]$Path, [string]$Role)
    $Pin = & $script:OriginalPin -Path $Path -Role $Role
    if ($Path -eq "{anchor}" -and -not $script:ProtectedParent) {{
        if (-not [LeaseNative.Kernel]::SetHandleInformation($Pin.Handle, 2, 2)) {{
            throw "the constructed parent could not be protected"
        }}
        $script:ProtectedParent = $true
        Write-Host "PARENT-PROTECTED"
    }}
    return $Pin
}}
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "refused-parent-retirement"
if ($Lease) {{
    Write-Host "RETURNED"
    if (-not (Exit-MachineLease $Lease)) {{ exit 4 }}
}}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert result.returncode == 4, output
    assert "PARENT-PROTECTED" in output and "RETURNED" not in output
    assert "after parent sharing adjustment" in output
    assert output.count("CloseHandle refused the pin") == 1, output
    successor = _NAMESPACE["acquire"]("after-shell-retirement-refusal", repo=anchor)
    _NAMESPACE["release"](successor)


def test_a_held_reparse_parent_refuses_write_access(anchor: Path, tmp_path: Path) -> None:
    import ctypes

    linked, jump, worktrees, pointer, original_pointer = _pointer_through_a_junction(
        anchor, tmp_path
    )
    kernel = _NAMESPACE["_kernel32"]()

    def writable() -> bool:
        handle = kernel.CreateFileW(str(jump), 0x40000080, 7, None, 3, 0x02200000, None)
        if handle is None or handle == _NAMESPACE["_INVALID_HANDLE"]:
            assert ctypes.get_last_error() == 32
            return False
        assert kernel.CloseHandle(ctypes.c_void_p(handle))
        return True

    try:
        assert writable(), "the constructed junction cannot exercise a write open"
        lease = _NAMESPACE["acquire"]("strict-reparse-parent", repo=linked)
        try:
            assert jump.is_junction()
            assert any(Path(pin.path).parent == jump for pin in lease.binding.pins)
            assert not writable(), "the held reparse parent admitted a write handle"
            lease.assert_bound()
        finally:
            _NAMESPACE["release"](lease)
        assert writable(), "release did not restore the junction's write access"
    finally:
        _repoint(pointer, original_pointer)
        os.rmdir(jump)


def test_parent_sharing_refuses_a_different_replacement_identity(
    anchor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other-directory"
    other.mkdir()
    module = _NAMESPACE["acquire"].__globals__
    assert module["_directory_identity"](anchor) != module["_directory_identity"](other)
    open_shared = module["_open_shared_pin"]
    substituted: list[Path] = []

    def different_directory(path: Path, role: str, share: int) -> object:
        if path == anchor and share & 2:
            substituted.append(path)
            return open_shared(other, role, share)
        return open_shared(path, role, share)

    with monkeypatch.context() as patch:
        patch.setitem(module, "_open_shared_pin", different_directory)
        with pytest.raises(LeaseRefused, match="parent changed while its sharing was adjusted"):
            lease = module["acquire"]("different-parent-identity", repo=anchor)
            module["release"](lease)
    assert substituted == [anchor], "the replacement boundary was not exercised once"
    successor = module["acquire"]("after-parent-identity-refusal", repo=anchor)
    module["release"](successor)


def test_git_configuration_can_be_written_while_its_directory_is_held(anchor: Path) -> None:
    lease = _NAMESPACE["acquire"]("configuration-write", repo=anchor)
    try:
        written = subprocess.run(
            ["git", "-C", str(anchor), "config", "--local", "lease.example", "recorded"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert written.returncode == 0, written.stdout + written.stderr
        actual = subprocess.run(
            ["git", "-C", str(anchor), "config", "--local", "--get", "lease.example"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert actual.returncode == 0 and actual.stdout.strip() == "recorded"
        with pytest.raises(PermissionError):
            (anchor / ".git").rename(anchor / "moved-git-directory")
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_lease_allows_git_configuration_without_allowing_a_swap(
    anchor: Path, host: str
) -> None:
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "configuration-write"
if (-not $Lease) {{ throw "lease was not acquired" }}
try {{
    & git -C "{anchor}" config --local lease.example recorded
    if ($LASTEXITCODE -ne 0) {{ throw "git configuration write failed" }}
    $Actual = & git -C "{anchor}" config --local --get lease.example
    if ($LASTEXITCODE -ne 0 -or $Actual -ne "recorded") {{
        throw "git did not store the requested configuration"
    }}
    $Moved = $false
    try {{
        [IO.Directory]::Move("{anchor / ".git"}", "{anchor / "moved-git-directory"}")
        $Moved = $true
    }} catch [IO.IOException] {{}}
    if ($Moved) {{
        [IO.Directory]::Move("{anchor / "moved-git-directory"}", "{anchor / ".git"}")
        throw "the held directory could be moved"
    }}
    Assert-MachineLeaseHeld -Lease $Lease
}} finally {{
    if (-not (Exit-MachineLease $Lease)) {{ exit 4 }}
}}
Write-Output "CONFIGURATION-STORED-AND-DIRECTORY-HELD"
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONFIGURATION-STORED-AND-DIRECTORY-HELD" in result.stdout


def test_the_writable_common_directory_keeps_its_lease_entry_held(
    anchor: Path, tmp_path: Path
) -> None:
    target = tmp_path / "junction-target"
    target.mkdir()
    lease = _NAMESPACE["acquire"]("ordinary-common-directory", repo=anchor)
    try:
        opened, changed, error = _set_junction_in_place(anchor / ".git", target)
        assert opened, f"ordinary common-directory writes were refused: {error}"
        assert not changed and error == 145
        with pytest.raises(PermissionError):
            _lease_file(anchor).rename(anchor / ".git" / "moved-lease")
        assert not (anchor / ".git").is_junction()
        lease.assert_bound()
    finally:
        _NAMESPACE["release"](lease)


def _lease_file_link(anchor: Path, tmp_path: Path) -> Path:
    target = tmp_path / "lease-target"
    target.write_bytes(b"untouched target")
    try:
        _lease_file(anchor).symlink_to(target)
    except OSError as error:
        if error.winerror == 1314:
            pytest.skip("creating file links requires the Windows developer setting")
        raise
    return target


def test_a_linked_lease_entry_is_refused_before_its_target_is_written(
    anchor: Path, tmp_path: Path
) -> None:
    target = _lease_file_link(anchor, tmp_path)
    lease = None
    try:
        with pytest.raises(LeaseRefused, match="not an ordinary file"):
            lease = _NAMESPACE["acquire"]("linked-lease", repo=anchor)
    finally:
        if lease is not None:
            _NAMESPACE["release"](lease)
    assert target.read_bytes() == b"untouched target"


@pytest.mark.parametrize("host", _powershell_hosts())
def test_shell_refuses_a_linked_lease_entry_without_writing_its_target(
    anchor: Path, tmp_path: Path, host: str
) -> None:
    target = _lease_file_link(anchor, tmp_path)
    script = f"""
$ErrorActionPreference = "Stop"
. "{ROOT / "scripts" / "machine-lease.ps1"}"
$Lease = Enter-MachineLease -RepositoryRoot "{anchor}" -Purpose "linked-lease"
if ($Lease) {{
    [void](Exit-MachineLease $Lease)
    exit 7
}}
Write-Output "LINKED-LEASE-REFUSED"
"""
    result = subprocess.run(
        [host, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not an ordinary file" in result.stdout
    assert "LINKED-LEASE-REFUSED" in result.stdout
    assert target.read_bytes() == b"untouched target"
