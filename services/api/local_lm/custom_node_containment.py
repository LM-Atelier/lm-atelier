"""Report whether a custom-node worker is confined.

A trusted package hash names the code. It does not stop that code from
reading the database, a backup, or a home directory, or from writing
outside its scratch. This probe runs a canary that tries those two
escapes and two allowed actions: reading its input and writing its scratch.
The host in which the escapes succeed is unavailable.

No containment backend is installed. A constructed record, a process-tree
limit, and a rewritten environment are not a launch witness. The public
result stays unavailable and does not authorize execution or an Offline badge.
"""

from __future__ import annotations

import enum
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_INPUT = b"neutral-input"
_MARKER = b"neutral-canary"
_OUTSIDE_NAME = "canary-outside-write"
_PROFILE_VERSION = 1
_BACKEND = "none"
_BACKEND_VERSION = "0"
_RESULT_FIELDS = (
    "environment_limited",
    "read_input",
    "read_outside",
    "wrote_outside",
    "wrote_scratch",
)
_RUNTIME_ENV_NAMES = (
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
)
_CHILD_ENV_NAMES = frozenset(_RUNTIME_ENV_NAMES) | {
    # Linux CPython writes LC_CTYPE when it coerces a C locale. The canary
    # sets that name itself, so the write is not a parent variable leaking in.
    "LC_CTYPE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONIOENCODING",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
}
_EXIT_MISSING_FIXTURE = 3
_EXIT_SCRATCH_FAILED = 4


class ContainmentLevel(enum.StrEnum):
    UNAVAILABLE = "unavailable"
    PROCESS_TREE_ONLY = "process_tree_only"
    FILESYSTEM_RESTRICTED = "filesystem_restricted"
    FILESYSTEM_NO_EGRESS = "filesystem_no_egress"
    VERIFIED = "verified"


@dataclass(frozen=True)
class PlatformContainmentCapability:
    """The installed proof, or the explicit absence of one.

    `profile_sha256` is null when no profile is bound. File and connect
    denial are provable only for an installed backend, which this host
    does not have.
    """

    version: int
    platform: str
    backend: str
    backend_version: str
    profile_sha256: str | None
    file_denial_provable: bool
    connect_denial_provable: bool


@dataclass(frozen=True)
class ContainmentProbe:
    """One canary attempt and the authority that attempt supports.

    Boolean fields are null when that action was not observed. Null is not
    a denial. `authorizes_execution` and `offline_badge` come only from a
    bound backend witness. This probe does not set them.
    """

    level: ContainmentLevel
    platform: str
    profile_version: int
    backend: str
    backend_version: str
    profile_sha256: str | None
    tested_at: str | None
    read_input_allowed: bool | None
    write_scratch_allowed: bool | None
    read_outside_denied: bool | None
    write_outside_denied: bool | None
    connect_denied: bool | None
    environment_limited: bool | None
    diagnostic: str
    authorizes_execution: bool
    offline_badge: bool


def host_containment_capability() -> PlatformContainmentCapability:
    """Return this process's capability. No backend is installed."""

    return PlatformContainmentCapability(
        version=_PROFILE_VERSION,
        platform=sys.platform,
        backend=_BACKEND,
        backend_version=_BACKEND_VERSION,
        profile_sha256=None,
        file_denial_provable=False,
        connect_denial_provable=False,
    )


def classify_observations(
    *,
    read_outside_denied: bool,
    write_outside_denied: bool,
    connect_denied: bool,
    capabilities_held: bool,
    process_tree_limited: bool = False,
) -> ContainmentLevel:
    """Map observations to a hypothetical level.

    This is not a launch witness. It never returns `verified`. A process-tree
    limit without file denial is `process_tree_only`, which is not confinement.
    """

    if not capabilities_held:
        return ContainmentLevel.UNAVAILABLE
    file_denied = read_outside_denied and write_outside_denied
    if not file_denied:
        if process_tree_limited:
            return ContainmentLevel.PROCESS_TREE_ONLY
        return ContainmentLevel.UNAVAILABLE
    if not connect_denied:
        return ContainmentLevel.FILESYSTEM_RESTRICTED
    return ContainmentLevel.FILESYSTEM_NO_EGRESS


def execution_authorized(
    probe: ContainmentProbe,
    *,
    fresh_after: datetime | None,
) -> bool:
    """Return whether `probe` may start a worker or grant an Offline badge.

    Missing, unparseable, stale, and unbound records are refused. A complete
    fresh record is refused too: this module has no backend witness.
    """

    if not _complete_verified_record(probe):
        return False
    if _proof_is_stale(probe.tested_at, fresh_after):
        return False
    return False


def offline_badge_allowed(
    probe: ContainmentProbe,
    *,
    fresh_after: datetime | None,
) -> bool:
    """Return whether `probe` may upgrade an Offline badge. It may not."""

    return execution_authorized(probe, fresh_after=fresh_after)


def child_environment(package_root: str) -> dict[str, str]:
    """Return the environment a canary child is allowed to see.

    The source pin is `package_root` alone. Parent variables other than the
    Windows runtime names are not copied.
    """

    env = {name: os.environ[name] for name in _RUNTIME_ENV_NAMES if name in os.environ}
    env["LC_CTYPE"] = "C.UTF-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = package_root
    return env


def probe_worker_host(root: Path) -> ContainmentProbe:
    """Run the canary under `root` and report the authority this host earned.

    The input, the scratch, and the outside marker live only under `root`.
    The child is ordinary, so an unconfined host lets it read and write.
    That result is unavailable.
    """

    outside = root / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    marker = outside / "marker.bin"
    marker.write_bytes(_MARKER)
    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "input.bin").write_bytes(_INPUT)
    return _run_canary(scratch, marker)


def main(argv: list[str]) -> int:
    """Entry point for the canary child. It prints booleans, never file bytes."""

    if len(argv) != 3 or argv[0] != "--canary":
        return 2
    scratch = Path(argv[1])
    secret = Path(argv[2])
    try:
        observed_input = (scratch / "input.bin").read_bytes()
    except FileNotFoundError:
        return _EXIT_MISSING_FIXTURE
    except OSError:
        return 2
    if observed_input != _INPUT:
        return _EXIT_MISSING_FIXTURE
    try:
        (scratch / "canary-scratch-result").write_bytes(b"ok")
    except OSError:
        return _EXIT_SCRATCH_FAILED
    try:
        secret.read_bytes()
    except PermissionError:
        read_outside = False
    except OSError:
        return 2
    else:
        read_outside = True
    target = secret.with_name(_OUTSIDE_NAME)
    try:
        target.write_bytes(b"wrote")
    except PermissionError:
        wrote_outside = False
    except OSError:
        return 2
    else:
        wrote_outside = True
    sys.stdout.write(
        json.dumps(
            {
                "environment_limited": _environment_is_limited(set(os.environ)),
                "read_input": True,
                "read_outside": read_outside,
                "wrote_outside": wrote_outside,
                "wrote_scratch": True,
            }
        )
    )
    return 0


def _run_canary(scratch: Path, marker: Path) -> ContainmentProbe:
    package_root = str(Path(__file__).resolve().parent.parent)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_lm.custom_node_containment",
                "--canary",
                str(scratch),
                str(marker),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=child_environment(package_root),
        )
    except subprocess.TimeoutExpired:
        return _unobserved("canary-timeout")
    except OSError:
        return _unobserved("canary-launch-failed")
    if completed.returncode == _EXIT_MISSING_FIXTURE:
        return _unobserved("canary-missing-fixture")
    if completed.returncode == _EXIT_SCRATCH_FAILED:
        return _unobserved("canary-scratch-failed")
    if completed.returncode != 0:
        return _unobserved("canary-no-result")
    observed = _parse_canary_output(completed.stdout)
    if observed is None:
        return _unobserved("canary-malformed-output")
    if not observed["environment_limited"]:
        return _unobserved("canary-environment")
    return _observed_probe(observed)


def _parse_canary_output(text: str) -> dict[str, bool] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != set(_RESULT_FIELDS):
        return None
    parsed: dict[str, bool] = {}
    for name in _RESULT_FIELDS:
        value = payload[name]
        if not isinstance(value, bool):
            return None
        parsed[name] = value
    return parsed


def _observed_probe(observed: dict[str, bool]) -> ContainmentProbe:
    capability = host_containment_capability()
    return ContainmentProbe(
        level=ContainmentLevel.UNAVAILABLE,
        platform=capability.platform,
        profile_version=capability.version,
        backend=capability.backend,
        backend_version=capability.backend_version,
        profile_sha256=None,
        tested_at=datetime.now(UTC).isoformat(),
        read_input_allowed=observed["read_input"],
        write_scratch_allowed=observed["wrote_scratch"],
        read_outside_denied=not observed["read_outside"],
        write_outside_denied=not observed["wrote_outside"],
        connect_denied=None,
        environment_limited=True,
        diagnostic="ok",
        authorizes_execution=False,
        offline_badge=False,
    )


def _unobserved(diagnostic: str) -> ContainmentProbe:
    capability = host_containment_capability()
    return ContainmentProbe(
        level=ContainmentLevel.UNAVAILABLE,
        platform=capability.platform,
        profile_version=capability.version,
        backend=capability.backend,
        backend_version=capability.backend_version,
        profile_sha256=None,
        tested_at=None,
        read_input_allowed=None,
        write_scratch_allowed=None,
        read_outside_denied=None,
        write_outside_denied=None,
        connect_denied=None,
        environment_limited=None,
        diagnostic=diagnostic,
        authorizes_execution=False,
        offline_badge=False,
    )


def _environment_is_limited(names: set[str]) -> bool:
    allowed = {name.casefold() for name in _CHILD_ENV_NAMES}
    return {name.casefold() for name in names}.issubset(allowed)


def _complete_verified_record(probe: ContainmentProbe) -> bool:
    if probe.level is not ContainmentLevel.VERIFIED:
        return False
    if probe.backend != _BACKEND or probe.backend_version != _BACKEND_VERSION:
        return False
    digest = probe.profile_sha256
    if digest is None or len(digest) != 64:
        return False
    if any(character not in "0123456789abcdef" for character in digest):
        return False
    return _parse_tested_at(probe.tested_at) is not None


def _proof_is_stale(tested_at: str | None, fresh_after: datetime | None) -> bool:
    tested = _parse_tested_at(tested_at)
    if tested is None or fresh_after is None:
        return True
    if fresh_after.tzinfo is None:
        return True
    return tested <= fresh_after


def _parse_tested_at(tested_at: str | None) -> datetime | None:
    if tested_at is None:
        return None
    try:
        parsed = datetime.fromisoformat(tested_at)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
