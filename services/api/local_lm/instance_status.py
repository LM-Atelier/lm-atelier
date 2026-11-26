"""The rules behind one status read a script can make of a running instance.

A script driving LM Atelier needs to know, before it acts, whether this is the
instance it expects, whether it has finished starting, whether it is locked,
whether its workers are up or down on purpose, whether there is room to work,
and whether a chat's workflows would run. Each rule here decides one of those
answers from state the application already keeps. None of them asks an engine
anything, so reading the status never starts, wakes or waits on a worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args

from .artifacts import COPY_HEADROOM_BYTES
from .domain import Operation
from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    available_bytes,
    open_entry,
)
from .schemas import (
    InstanceBuildKind,
    InstanceStartupOut,
    InstanceStartupState,
    OperationReadinessOut,
    RecipeReadinessReason,
    StartupTaskName,
    StartupTaskOut,
    StartupTaskState,
    StorageStandingOut,
    StorageState,
    WorkerCondition,
    WorkerEnd,
    WorkerRestoreState,
    WorkerStatus,
    WorkerStopReason,
    WorkflowReadinessReason,
    WorkflowTrust,
    WorkflowVariantReadiness,
)

if TYPE_CHECKING:
    from .worker_startup import RestoredWorkers

RELEASE_MANIFEST_FILE: Final = "release-manifest.json"
#: Far above any manifest a release writes. A larger file is not read at all.
MAX_RELEASE_MANIFEST_BYTES: Final = 1024 * 1024
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ReleaseBuild:
    kind: InstanceBuildKind
    commit: str | None = None
    dirty: bool | None = None


def read_release_build(root: Path | None) -> ReleaseBuild:
    """Whether this is a release, and the commit its bundled manifest says it was built from.

    `root` is the folder a frozen release unpacks into, or None when running
    from source, which has no commit to report. The manifest is opened through
    the held bundle folder, so a filesystem link on the way to it or in its
    place is refused rather than followed, and only a bounded amount is read.
    A manifest that is missing, refused, oversized or not shaped as a release
    writes it leaves a release with no commit: a value that cannot be checked
    is not reported.
    """

    if root is None:
        return ReleaseBuild(kind="source")
    document = _read_manifest(root)
    source = document.get("source") if isinstance(document, dict) else None
    if not isinstance(source, dict):
        return ReleaseBuild(kind="release")
    commit = source.get("commit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        return ReleaseBuild(kind="release")
    dirty = source.get("dirty")
    return ReleaseBuild(
        kind="release", commit=commit, dirty=dirty if isinstance(dirty, bool) else None
    )


def _read_manifest(root: Path) -> object:
    try:
        with AnchoredDirectory(root) as bundle:
            descriptor = open_entry(bundle, RELEASE_MANIFEST_FILE)
            if descriptor is None:
                return None
            try:
                raw = _read_bounded(descriptor, MAX_RELEASE_MANIFEST_BYTES + 1)
            finally:
                os.close(descriptor)
    except (AnchoredDirectoryError, OSError):
        return None
    if len(raw) > MAX_RELEASE_MANIFEST_BYTES:
        return None
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        # RecursionError is what deeply nested JSON raises instead of a ValueError.
        return None


def _read_bounded(descriptor: int, limit: int) -> bytes:
    """Read up to `limit` bytes, going on through short reads until the end of the file.

    One read may return fewer bytes than asked for before the end, so a single
    read could miss the byte past the bound that marks a file as too large.
    """

    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class StartupProgress:
    """The startup work that goes on after the port opens, as the lifespan holds it.

    Nothing records progress as it happens: each read evaluates the tasks as
    they stand, so the answer cannot fall behind them.
    """

    worker_restore: asyncio.Task[RestoredWorkers]
    #: None when no managed runtime needed verifying.
    runtime_verification: asyncio.Task[None] | None
    reference_index: asyncio.Task[None]
    #: Set once the reference index is current. The retention sweep waits for it.
    index_ready: asyncio.Event
    #: Its result says whether the sweep finished; False when it gave up.
    retention_sweep: asyncio.Task[bool]


def startup_standing(progress: StartupProgress | None) -> InstanceStartupOut:
    """Each startup task's state, and whether startup is complete, stalled or still running.

    None is a lifespan that has not reached this work yet, so all of it is
    still to run. Startup has stalled when nothing is running any more and some
    of the work gave up: the reference index upkeep that ended without the
    index becoming current leaves the retention sweep waiting for good, and a
    sweep that failed or kept meeting contention gives up until the next start.
    """

    if progress is None:
        names = cast("tuple[StartupTaskName, ...]", get_args(StartupTaskName))
        return InstanceStartupOut(
            state="running", tasks=[StartupTaskOut(task=name, state="running") for name in names]
        )
    index: StartupTaskState
    if progress.index_ready.is_set():
        index = "done"
    elif progress.reference_index.done():
        index = "gave_up"
    else:
        index = "running"
    sweep: StartupTaskState
    if progress.retention_sweep.done():
        sweep = _ended(progress.retention_sweep)
        if sweep == "done" and progress.retention_sweep.result() is False:
            sweep = "gave_up"
    else:
        sweep = "running" if progress.index_ready.is_set() else "waiting"
    tasks = [
        StartupTaskOut(task="worker_restore", state=_task_state(progress.worker_restore)),
        StartupTaskOut(
            task="runtime_verification", state=_task_state(progress.runtime_verification)
        ),
        StartupTaskOut(task="reference_index", state=index),
        StartupTaskOut(task="retention_sweep", state=sweep),
    ]
    states = {task.state for task in tasks}
    overall: InstanceStartupState
    if "running" in states:
        overall = "running"
    elif "gave_up" in states:
        overall = "stalled"
    elif states == {"done"}:
        overall = "complete"
    else:
        # Only a sweep waiting on an index still being kept, which is running.
        overall = "running"
    return InstanceStartupOut(state=overall, tasks=tasks)


def _task_state(task: asyncio.Task[Any] | None) -> StartupTaskState:
    if task is None:
        return "done"
    if not task.done():
        return "running"
    return _ended(task)


def _ended(task: asyncio.Task[Any]) -> StartupTaskState:
    # One that raised, or was cancelled, ended without doing all of its work.
    if task.cancelled() or task.exception() is not None:
        return "gave_up"
    return "done"


def restore_state(
    progress: StartupProgress | None, worker: Literal["chat", "media"]
) -> WorkerRestoreState:
    """How the restore at startup ended for one worker, or `pending` until it has."""

    if progress is None or not progress.worker_restore.done():
        return "pending"
    task = progress.worker_restore
    if task.cancelled() or task.exception() is not None:
        # The restore ended without saying how it went for either worker.
        return "failed"
    restored = task.result()
    return restored.chat if worker == "chat" else restored.media


def worker_condition(
    status: WorkerStatus, *, last_end: WorkerEnd | None, restore: WorkerRestoreState
) -> WorkerCondition:
    """Whether a worker is in use, on its way, down on purpose, or down by failure.

    A named failure is read first, because a stop that could not finish still
    reads `stopping` and a process left holding the port reads `stopped`.
    Otherwise a worker starting, ready or stopping is as it says. A worker that
    is down is down by failure when it last ended by exiting with nothing
    stopping it, or by a stop for failing its health checks, and down on
    purpose when it was last stopped for any other reason. The last stop reason
    alone cannot tell these apart: it is kept through a later crash. A worker
    that has not ended since the service started is failing only when its
    restore failed.
    """

    if status.state == "exited" or status.failure_detail is not None:
        return "failing"
    match status.state:
        case "starting" | "ready" | "stopping" as moving:
            return moving
    if last_end == "exit":
        return "failing"
    if last_end == "stop":
        return "failing" if status.last_stop_reason is WorkerStopReason.UNHEALTHY else "stopped"
    return "failing" if restore == "failed" else "not_started"


def storage_state(free_bytes: int | None, *, floor_bytes: int, warning_bytes: int) -> StorageState:
    if free_bytes is None:
        return "unmeasured"
    if free_bytes < floor_bytes:
        return "below_floor"
    if free_bytes < warning_bytes:
        return "low"
    return "ok"


def storage_standing(artifact_root: Path, *, warning_bytes: int) -> StorageStandingOut:
    """Free space on the artifact store's volume, against its floor and the storage warning.

    Measured through the held store folder, as the private copy that enforces
    the floor measures it, so both describe the same volume.
    """

    free: int | None
    try:
        with AnchoredDirectory(artifact_root) as root:
            free = available_bytes(root)
    except (AnchoredDirectoryError, OSError):
        free = None
    return StorageStandingOut(
        state=storage_state(free, floor_bytes=COPY_HEADROOM_BYTES, warning_bytes=warning_bytes),
        free_bytes=free,
        floor_bytes=COPY_HEADROOM_BYTES,
        warning_bytes=warning_bytes,
    )


#: The vocabulary is a union of Literals, so its members are read from each part.
_KNOWN_REASONS: Final[frozenset[str]] = frozenset(
    member for part in get_args(WorkflowReadinessReason) for member in get_args(part)
)
_REVIEW_REQUIRED: Final[frozenset[str]] = frozenset({"revision_untrusted", "project_pin_untrusted"})
#: Reasons something can be installed, activated or chosen to put right: among
#: them every refusal of a recipe, which editing it or choosing another clears.
_SETUP_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "activation_not_ready",
        "current_revision_invalid",
        "current_revision_missing",
        "default_missing",
        "model_unavailable",
        "no_ready_workflow",
        "node_packages_missing",
        "operation_unavailable",
        "profile_binding_missing",
        "project_pin_model_mismatch",
        *get_args(RecipeReadinessReason),
    }
)


def refused_operation(
    operation: Operation,
    reason: str,
    *,
    trust: WorkflowTrust | None = None,
    workflow_revision_id: str | None = None,
) -> OperationReadinessOut:
    """An operation a plain turn would refuse, and why.

    A reason outside the vocabulary is reported as `operation_unavailable`, the
    resolver's own reason for an operation no variant can run, so the answer
    stays within the closed set. Unless the caller names the trust of the
    revision it resolved, trust is known only when the refusal is about trust.
    """

    known = (
        cast(WorkflowReadinessReason, reason)
        if reason in _KNOWN_REASONS
        else "operation_unavailable"
    )
    readiness: WorkflowVariantReadiness = (
        "review_required"
        if known in _REVIEW_REQUIRED
        else "setup_required"
        if known in _SETUP_REQUIRED
        else "unavailable"
    )
    if trust is None:
        trust = "untrusted" if readiness == "review_required" else "not_applicable"
    return OperationReadinessOut(
        operation=operation,
        readiness=readiness,
        trust=trust,
        reason=known,
        workflow_revision_id=workflow_revision_id,
    )
