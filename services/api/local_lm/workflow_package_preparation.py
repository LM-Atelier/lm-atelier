"""Prepare a Registry package end to end as one durable job.

The composition is resolver -> closure driver -> atomic preparation, each a
frozen contract with typed refusals; nothing here re-implements any of their
semantics. Success is always committed inactive and untrusted - reviewing,
trusting, and activating what the recorded identities describe are separate
explicit steps.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from sqlalchemy.orm import Session

from .comfy_package_requirements import (
    StagedRequirementsError,
    read_staged_requirements,
    select_requirements_manifest,
    staged_requirements_manifests,
)
from .comfy_registry import ComfyRegistryClient
from .comfy_registry_closure_driver import (
    ComfyRegistryWheelClosureDriverError,
    ComfyRegistryWheelMetadataClient,
    drive_comfy_registry_wheel_closure,
)
from .comfy_registry_downloads import ComfyRegistryArchiveDownloader
from .comfy_registry_interpreter import ComfyRegistryInterpreterError
from .comfy_registry_lifecycle import (
    ComfyRegistryLifecycleError,
    ComfyRegistryPreparation,
    ComfyRegistryStagedArchive,
    discard_comfy_registry_staged_archive,
    prepare_comfy_registry_install,
    renew_comfy_registry_install_environment,
    stage_comfy_registry_install_archive,
)
from .comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from .comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from .comfy_registry_wheel_projects import ComfyRegistryWheelProjectClient
from .comfy_workflow_packages import WorkflowPackageRequirement
from .config import Settings
from .models import ComfyRegistryInstall
from .package_sources import partition_unpinned_sources
from .source_omission_proof import PendingOmission, pending_omission_requirement

# One target interpreter probe: markers, wheel tags, and installed distributions for the
# managed ComfyUI python. Owned as its own contract because target-binding
# correctness depends on it; until a real probe is wired the preparation
# refuses rather than guessing the target.
InterpreterProbe = Callable[
    [Path],
    Awaitable[
        tuple[Mapping[str, str], Sequence[str]]
        | tuple[
            Mapping[str, str],
            Sequence[str],
            Sequence[ComfyRegistryRuntimeDistribution],
        ]
    ],
]

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
            state_root=settings.registry_dir,
        )


@dataclass(frozen=True)
class _RenewedInstall:
    """What a renewal needs to know about the install it is refreshing."""

    installed_path: str
    recorded_omissions: frozenset[str]
    # Staging is what learns these, and a renewal deliberately does not stage.
    # They name the archive already on disk - the one being reused - so the
    # renewal carries them forward rather than resolving to nothing and
    # tripping the identity check against itself.
    registry_record_id: str | None
    download_url: str | None


def _renewed_install(session: Session, install_id: str) -> _RenewedInstall | None:
    install = session.get(ComfyRegistryInstall, install_id)
    if install is None or not install.installed_path:
        return None
    recorded = pending_omission_requirement(install_id, install.review_json)
    return _RenewedInstall(
        installed_path=install.installed_path,
        recorded_omissions=frozenset(recorded.omitted_declarations if recorded else ()),
        registry_record_id=install.registry_record_id,
        download_url=install.download_url,
    )


async def prepare_workflow_package(
    session_factory: Callable[[], Session],
    *,
    package_id: str,
    version: str | None,
    node_types: tuple[str, ...],
    context: PreparationContext,
    media_worker_stopped: bool,
    interpreter_probe: InterpreterProbe,
    registry_client: ComfyRegistryClient,
    project_client: ComfyRegistryWheelProjectClient,
    metadata_client: ComfyRegistryWheelMetadataClient,
    archive_downloader: ComfyRegistryArchiveDownloader,
    wheel_downloader: ComfyRegistryWheelDownloader,
    phase: PreparationPhase | None = None,
    renew_install_id: str | None = None,
    authorized_workflow: tuple[str, tuple[str, ...]] | None = None,
) -> ComfyRegistryPreparation:
    """Resolve, close, and prepare one package; refuse with the source's code.

    The caller holds the media scheduler lease and reports the worker state
    truthfully; the lifecycle re-checks it before any mutation. Resolution,
    probing, and closure run without any database session; one opens only
    around the atomic prepare step, freshly, so it enters holding no lock.
    """

    def _phase(name: str, done: int | None = None, total: int | None = None) -> None:
        if phase is not None:
            phase(name, done, total)

    _phase("Resolving the package")
    requirement = WorkflowPackageRequirement(
        package_id=package_id,
        versions=(version,) if version else (),
        # The exact node identities the analyzed graph needs from this package.
        # Sending none produced a prepared package that claimed to provide
        # nothing, which persistence refused and no activation could use.
        node_types=node_types,
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
    renewed_install: _RenewedInstall | None = None
    if renew_install_id is not None:
        with session_factory() as session:
            renewed_install = _renewed_install(session, renew_install_id)
    if (
        renew_install_id is not None
        and resolution.install_kind == "git_commit"
        and renewed_install is None
    ):
        # Fail closed if the install cannot be read: a commit-pinned package
        # states its dependencies inside its own tree, and renewal deliberately
        # does not fetch that tree again, so without the staged copy there is
        # nothing to read them from.
        raise WorkflowPackagePreparationError(
            "registry_renewal_source_unsupported",
            "The staged package this renewal would reuse could not be read",
        )

    _phase("Probing the target runtime")
    try:
        probe_result = await interpreter_probe(context.python_executable)
        if len(probe_result) == 2:
            marker_environment, supported_tags = probe_result
            runtime_distributions: Sequence[ComfyRegistryRuntimeDistribution] = ()
        else:
            marker_environment, supported_tags, runtime_distributions = probe_result
    except WorkflowPackagePreparationError:
        raise
    except ComfyRegistryInterpreterError as exc:
        # The probe's refusals are typed; converting them to one generic code
        # would erase exactly the distinction the caller needs.
        raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc
    except Exception as exc:
        raise WorkflowPackagePreparationError(
            "interpreter_probe_failed",
            "The managed runtime's package target could not be determined.",
        ) from exc

    async def _archive_progress(downloaded: int, total: int | None) -> None:
        _phase("Downloading the node archive", downloaded, total)

    staged_archive: ComfyRegistryStagedArchive | None = None
    effective_resolution = resolution
    # Only a staged commit-pinned package states dependencies inside its own
    # tree, so only that path can set anything aside; every other resolution
    # reaches the recording step with nothing omitted.
    omitted: tuple[str, ...] = ()
    if resolution.install_kind == "git_commit" and renewed_install is not None:
        # Renewal reuses the node code it already reviewed and never fetches it
        # again, so the declarations are read from the copy already on disk.
        # Refusing instead - which is what this did - left a commit-pinned
        # package unable to follow its runtime anywhere, and told the reader to
        # remove it, which nothing offers a way to do.
        _phase("Reading package dependencies")
        staged = context.custom_node_root / renewed_install.installed_path
        manifest = select_requirements_manifest(staged_requirements_manifests(staged))
        effective_resolution = replace(
            resolution,
            pip_dependencies=(
                read_staged_requirements(staged, manifest) if manifest is not None else ()
            ),
            registry_record_id=renewed_install.registry_record_id,
            download_url=renewed_install.download_url,
        )
    elif resolution.install_kind == "git_commit":
        try:
            staged_archive = await stage_comfy_registry_install_archive(
                resolution=resolution,
                archive_downloader=archive_downloader,
                custom_node_root=context.custom_node_root,
                media_worker_stopped=media_worker_stopped,
                archive_progress=_archive_progress,
            )
            _phase("Reading package dependencies")
            manifest = select_requirements_manifest(staged_archive.report.dependency_manifests)
            dependencies = (
                read_staged_requirements(staged_archive.destination, manifest)
                if manifest is not None
                else ()
            )
            effective_resolution = replace(
                resolution,
                pip_dependencies=dependencies,
            )
        except ComfyRegistryLifecycleError as exc:
            raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc
        except StagedRequirementsError as exc:
            if staged_archive is not None:
                await discard_comfy_registry_staged_archive(
                    resolution=resolution,
                    staged_archive=staged_archive,
                    custom_node_root=context.custom_node_root,
                )
            raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc

    # Partitioned here, after both sources of declarations have been read: a
    # commit-pinned package states them inside its staged tree and a Registry
    # archive states them in its record. Doing it per-branch covered one live
    # case and missed the other.
    #
    # In preparation rather than in the planner, which stays uniformly hostile
    # to URLs, and only under an authorized workflow - without one there is
    # nothing an omission could later be proven against, so the ordinary
    # refusal stands.
    installable, omitted = partition_unpinned_sources(
        effective_resolution.pip_dependencies,
        authorized=authorized_workflow is not None,
    )
    effective_resolution = replace(effective_resolution, pip_dependencies=installable)
    pending_omission = (
        PendingOmission(
            omitted_declarations=omitted,
            workflow_revision_id=authorized_workflow[0],
            required_node_types=authorized_workflow[1],
        )
        if omitted and authorized_workflow is not None
        else None
    )

    async def _closure_progress(name: str, round_number: int, items: tuple[str, ...]) -> None:
        _phase(f"Dependencies: {name.replace('_', ' ')} (round {round_number})", len(items), None)

    try:
        closure_result = await drive_comfy_registry_wheel_closure(
            effective_resolution,
            project_fetcher=project_client.fetch,
            metadata_fetcher=metadata_client.fetch,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
            runtime_distributions=runtime_distributions,
            progress=_closure_progress,
        )
    except asyncio.CancelledError:
        if staged_archive is not None:
            await discard_comfy_registry_staged_archive(
                resolution=resolution,
                staged_archive=staged_archive,
                custom_node_root=context.custom_node_root,
            )
        raise
    except ComfyRegistryWheelClosureDriverError as exc:
        if staged_archive is not None:
            await discard_comfy_registry_staged_archive(
                resolution=resolution,
                staged_archive=staged_archive,
                custom_node_root=context.custom_node_root,
            )
        raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc

    async def _wheel_progress(filename: str, downloaded: int, total: int | None) -> None:
        _phase(f"Downloading {filename}", downloaded, total)

    try:
        with session_factory() as session:
            # Audited await: the session is opened fresh with no prior writes,
            # so it holds no SQLite lock while the lifecycle downloads and
            # assembles; the concurrency regression beside this proves another
            # writer makes progress mid-preparation.
            # Renewal refreshes an existing install's dependencies and is not
            # the act that can set a declaration aside, so an omission reaching
            # it is a contradiction rather than something to record quietly.
            if renew_install_id is not None:
                # A renewal may re-derive the omissions this install already
                # carries - it is reading the same declarations from the same
                # reviewed tree, so finding them again is agreement, not news.
                # What it may not do is set aside something new, because an
                # omission is only trustworthy when it was proven against the
                # workflow that authorized it, and a renewal authorizes nothing.
                recorded = renewed_install.recorded_omissions if renewed_install else frozenset()
                introduced = sorted(set(omitted) - recorded)
                if introduced:
                    raise WorkflowPackagePreparationError(
                        "omission_not_renewable",
                        "A dependency renewal cannot set aside something new: "
                        + ", ".join(introduced),
                    )
            if renew_install_id is None:
                preparation = await prepare_comfy_registry_install(
                    session,
                    resolution=effective_resolution,
                    closure=closure_result.closure,
                    archive_downloader=archive_downloader,
                    wheel_downloader=wheel_downloader,
                    python_executable=context.python_executable,
                    custom_node_root=context.custom_node_root,
                    state_root=context.state_root,
                    media_worker_stopped=media_worker_stopped,
                    archive_progress=_archive_progress,
                    wheel_progress=_wheel_progress,
                    staged_archive=staged_archive,
                    pending_omission=pending_omission,
                )
            else:
                preparation = await renew_comfy_registry_install_environment(
                    session,
                    install_id=renew_install_id,
                    resolution=effective_resolution,
                    closure=closure_result.closure,
                    wheel_downloader=wheel_downloader,
                    python_executable=context.python_executable,
                    custom_node_root=context.custom_node_root,
                    state_root=context.state_root,
                    media_worker_stopped=media_worker_stopped,
                    wheel_progress=_wheel_progress,
                )
            staged_archive = None
            return preparation
    except ComfyRegistryLifecycleError as exc:
        if staged_archive is not None:
            await discard_comfy_registry_staged_archive(
                resolution=resolution,
                staged_archive=staged_archive,
                custom_node_root=context.custom_node_root,
            )
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
