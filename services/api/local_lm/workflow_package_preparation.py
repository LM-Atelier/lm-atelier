"""Prepare a Registry package end to end as one durable job.

The composition is resolver -> closure driver -> atomic preparation, each a
frozen contract with typed refusals; nothing here re-implements any of their
semantics. Success is always committed inactive and untrusted - reviewing,
trusting, and activating what the recorded identities describe are separate
explicit steps.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from .comfy_registry import ComfyRegistryClient
from .comfy_registry_closure_driver import (
    ComfyRegistryWheelClosureDriverError,
    ComfyRegistryWheelMetadataClient,
    drive_comfy_registry_wheel_closure,
)
from .comfy_registry_downloads import ComfyRegistryArchiveDownloader
from .comfy_registry_lifecycle import (
    ComfyRegistryLifecycleError,
    ComfyRegistryPreparation,
    prepare_comfy_registry_install,
)
from .comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from .comfy_registry_wheel_projects import ComfyRegistryWheelProjectClient
from .comfy_workflow_packages import WorkflowPackageRequirement
from .config import Settings

# One target interpreter probe: (marker_environment, supported_tags) for the
# managed ComfyUI python. Owned as its own contract because target-binding
# correctness depends on it; until a real probe is wired the preparation
# refuses rather than guessing the target.
InterpreterProbe = Callable[[Path], Awaitable[tuple[Mapping[str, str], Sequence[str]]]]

PreparationPhase = Callable[[str, int | None, int | None], None]


class WorkflowPackagePreparationError(ValueError):
    """A typed refusal from any stage, code preserved from its source."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparationContext:
    """Every machine-specific input, resolved once and passed whole."""

    python_executable: Path
    custom_node_root: Path
    state_root: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> PreparationContext:
        if not settings.comfy_executable or not settings.comfy_directory:
            raise WorkflowPackagePreparationError(
                "managed_runtime_unavailable",
                "The managed ComfyUI runtime is not configured on this machine.",
            )
        return cls(
            python_executable=Path(settings.comfy_executable),
            custom_node_root=Path(settings.comfy_directory) / "custom_nodes",
            state_root=Path(settings.data_dir) / "registry",
        )


async def prepare_workflow_package(
    session: Session,
    *,
    package_id: str,
    version: str | None,
    context: PreparationContext,
    media_worker_stopped: bool,
    interpreter_probe: InterpreterProbe,
    registry_client: ComfyRegistryClient,
    project_client: ComfyRegistryWheelProjectClient,
    metadata_client: ComfyRegistryWheelMetadataClient,
    archive_downloader: ComfyRegistryArchiveDownloader,
    wheel_downloader: ComfyRegistryWheelDownloader,
    phase: PreparationPhase | None = None,
) -> ComfyRegistryPreparation:
    """Resolve, close, and prepare one package; refuse with the source's code.

    The caller holds the media scheduler lease and reports the worker state
    truthfully; the lifecycle re-checks it before any mutation.
    """

    def _phase(name: str, done: int | None = None, total: int | None = None) -> None:
        if phase is not None:
            phase(name, done, total)

    _phase("Resolving the package")
    requirement = WorkflowPackageRequirement(
        package_id=package_id,
        versions=(version,) if version else (),
        node_types=(),
        locally_resolved=False,
    )
    try:
        registry = await registry_client.resolve([requirement])
    except ValueError as exc:
        raise WorkflowPackagePreparationError(
            getattr(exc, "code", "registry_resolution_failed"), str(exc)
        ) from exc
    resolution = registry.packages[0]
    if resolution.error_code:
        raise WorkflowPackagePreparationError(
            resolution.error_code,
            f"The Registry could not resolve {package_id}.",
        )

    _phase("Probing the target runtime")
    try:
        marker_environment, supported_tags = await interpreter_probe(context.python_executable)
    except WorkflowPackagePreparationError:
        raise
    except Exception as exc:
        raise WorkflowPackagePreparationError(
            "interpreter_probe_failed",
            "The managed runtime's package target could not be determined.",
        ) from exc

    async def _closure_progress(name: str, round_number: int, items: tuple[str, ...]) -> None:
        _phase(f"Dependencies: {name.replace('_', ' ')} (round {round_number})", len(items), None)

    try:
        closure_result = await drive_comfy_registry_wheel_closure(
            resolution,
            project_fetcher=project_client.fetch,
            metadata_fetcher=metadata_client.fetch,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
            progress=_closure_progress,
        )
    except ComfyRegistryWheelClosureDriverError as exc:
        raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc

    async def _archive_progress(downloaded: int, total: int | None) -> None:
        _phase("Downloading the node archive", downloaded, total)

    async def _wheel_progress(filename: str, downloaded: int, total: int | None) -> None:
        _phase(f"Downloading {filename}", downloaded, total)

    try:
        return await prepare_comfy_registry_install(
            session,
            resolution=resolution,
            closure=closure_result.closure,
            archive_downloader=archive_downloader,
            wheel_downloader=wheel_downloader,
            python_executable=context.python_executable,
            custom_node_root=context.custom_node_root,
            state_root=context.state_root,
            media_worker_stopped=media_worker_stopped,
            archive_progress=_archive_progress,
            wheel_progress=_wheel_progress,
        )
    except ComfyRegistryLifecycleError as exc:
        raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc


async def refuse_interpreter_probe(
    python_executable: Path,
) -> tuple[Mapping[str, str], Sequence[str]]:
    """Fail closed until a verified probe for the managed python exists.

    Guessing the host interpreter's markers or tags for the managed one would
    bind wheel selection to the wrong target - the exact inference the chain
    exists to prevent.
    """

    raise WorkflowPackagePreparationError(
        "interpreter_probe_unavailable",
        "Determining the managed runtime's package target is not supported yet.",
    )
