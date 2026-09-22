"""Bind inert extension inspection and wheel resolution to later preparation."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_package_requirements import read_staged_requirements, select_requirements_manifest
from .comfy_registry import ComfyNodeResolution, ComfyRegistryClient
from .comfy_registry_archives import ComfyRegistryArchiveReport
from .comfy_registry_closure_driver import (
    ComfyRegistryWheelMetadataClient,
    drive_comfy_registry_wheel_closure,
)
from .comfy_registry_downloads import DownloadProgress
from .comfy_registry_lifecycle import RegistryArchiveDownloader
from .comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies
from .comfy_registry_mixed_wheel_closure import (
    ComfyRegistryMixedWheelClosure,
    validate_comfy_registry_mixed_wheel_closure,
)
from .comfy_registry_reviewed_closure import resolve_comfy_registry_reviewed_closure
from .comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactManifest,
    comfy_registry_wheel_target_sha256,
)
from .comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosure,
    validate_comfy_registry_wheel_closure,
)
from .comfy_registry_wheel_projects import ComfyRegistryWheelProjectClient
from .comfy_workflow_packages import WorkflowPackageRequirement
from .schemas import ApiModel
from .workflow_trust import canonical_graph

if TYPE_CHECKING:
    from .workflow_package_preparation import InterpreterProbe


class WorkflowPackageExecutionPlanError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__("The extension installation plan is unavailable or has changed.")
        self.code = code


class _ExecutionInputs(ApiModel):
    version: Literal[1, 2] = 1
    resolution: ComfyNodeResolution
    archive: ComfyRegistryArchiveReport
    archive_bytes: int = Field(gt=0)
    closure: ComfyRegistryWheelClosure | ComfyRegistryMixedWheelClosure


class WorkflowPackageExecutionPlan(_ExecutionInputs):
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def wheel_bytes(self) -> int:
        retained = (
            sum(item.size_bytes for item in self.closure.manifest.reviewed_local)
            if isinstance(self.closure, ComfyRegistryMixedWheelClosure)
            else 0
        )
        return sum(item.size_bytes for item in self._remote_manifest.artifacts) + retained

    @property
    def _remote_manifest(self) -> ComfyRegistryWheelArtifactManifest:
        return (
            self.closure.manifest.remote
            if isinstance(self.closure, ComfyRegistryMixedWheelClosure)
            else self.closure.manifest
        )

    @property
    def download_bytes(self) -> int:
        return self.archive_bytes + sum(item.size_bytes for item in self._remote_manifest.artifacts)

    def verify(self) -> None:
        mixed = isinstance(self.closure, ComfyRegistryMixedWheelClosure)
        if type(self.version) is not int or self.version != (2 if mixed else 1):
            raise WorkflowPackageExecutionPlanError("workflow-package-execution-plan-changed")
        expected = hashlib.sha256(
            canonical_graph(self.model_dump(mode="json", exclude={"plan_sha256"})).encode("utf-8")
        ).hexdigest()
        if expected != self.plan_sha256 or not self.closure.complete:
            raise WorkflowPackageExecutionPlanError("workflow-package-execution-plan-changed")
        _validate_closure(self.closure)

    def verify_resolution(self, resolution: ComfyNodeResolution) -> None:
        self.verify()
        if resolution != self.resolution:
            raise WorkflowPackageExecutionPlanError("workflow-package-resolution-changed")

    def verify_target(self, environment: Mapping[str, str], tags: Sequence[str]) -> None:
        self.verify()
        if (
            comfy_registry_wheel_target_sha256(environment, tags)
            != self._remote_manifest.target_sha256
        ):
            raise WorkflowPackageExecutionPlanError("workflow-package-runtime-changed")

    def verify_requested(self, package_id: str, version: str | None, nodes: Sequence[str]) -> None:
        self.verify()
        if (
            package_id != self.resolution.package_id
            or version != self.resolution.declared_version
            or sorted(nodes) != sorted(self.resolution.node_types)
        ):
            raise WorkflowPackageExecutionPlanError("workflow-package-resolution-changed")

    def verify_closure(
        self, closure: ComfyRegistryWheelClosure | ComfyRegistryMixedWheelClosure
    ) -> None:
        self.verify()
        try:
            _validate_closure(closure)
        except ValueError as exc:
            raise WorkflowPackageExecutionPlanError(
                "workflow-package-dependencies-changed"
            ) from exc
        if closure != self.closure:
            raise WorkflowPackageExecutionPlanError("workflow-package-dependencies-changed")


def _validate_closure(closure: ComfyRegistryWheelClosure | ComfyRegistryMixedWheelClosure) -> None:
    if isinstance(closure, ComfyRegistryMixedWheelClosure):
        validate_comfy_registry_mixed_wheel_closure(closure)
    else:
        validate_comfy_registry_wheel_closure(closure)


class PlannedArchiveDownloader:
    """Check the approved archive before preparation can assemble or persist it."""

    def __init__(
        self, downloader: RegistryArchiveDownloader, plan: WorkflowPackageExecutionPlan
    ) -> None:
        self.downloader = downloader
        self.plan = plan

    async def download_and_stage(
        self,
        resolution: ComfyNodeResolution,
        destination: Path,
        *,
        progress: DownloadProgress | None = None,
    ) -> ComfyRegistryArchiveReport:
        self.plan.verify()
        downloaded = 0

        async def measure(done: int, total: int | None) -> None:
            nonlocal downloaded
            downloaded = done
            if progress is not None:
                await progress(done, total)

        report = await self.downloader.download_and_stage(resolution, destination, progress=measure)
        if report != self.plan.archive or downloaded != self.plan.archive_bytes:
            raise WorkflowPackageExecutionPlanError("workflow-package-archive-changed")
        return report


async def plan_workflow_package_execution(
    *,
    requirement: WorkflowPackageRequirement,
    python_executable: Path,
    interpreter_probe: InterpreterProbe,
    registry_client: ComfyRegistryClient,
    project_client: ComfyRegistryWheelProjectClient,
    metadata_client: ComfyRegistryWheelMetadataClient,
    archive_downloader: RegistryArchiveDownloader,
    source_session_factory: Callable[[], Session] | None = None,
    source_store: ArtifactStore | None = None,
) -> WorkflowPackageExecutionPlan:
    """Inspect in temporary storage without installing code or writing application rows."""
    resolved = await registry_client.resolve([replace(requirement, locally_resolved=False)])
    if len(resolved.packages) != 1:
        raise WorkflowPackageExecutionPlanError("workflow-package-resolution-unavailable")
    resolution = resolved.packages[0]
    if (
        resolution.error_code
        or resolution.install_kind not in {"git_commit", "registry_archive"}
        or resolution.package_id != requirement.package_id
        or len(requirement.versions) != 1
        or resolution.declared_version != requirement.versions[0]
        or sorted(resolution.node_types) != sorted(requirement.node_types)
    ):
        raise WorkflowPackageExecutionPlanError("workflow-package-resolution-unavailable")
    probed = await interpreter_probe(python_executable)
    environment, tags = probed[:2]
    runtime: Sequence[ComfyRegistryRuntimeDistribution] = probed[2] if len(probed) == 3 else ()
    archive_bytes = 0

    async def measure(done: int, total: int | None) -> None:
        nonlocal archive_bytes
        archive_bytes = done

    with tempfile.TemporaryDirectory(prefix="workflow-extension-plan-") as directory:
        destination = Path(directory) / "package"
        archive = await archive_downloader.download_and_stage(
            resolution, destination, progress=measure
        )
        effective = resolution
        if resolution.install_kind == "git_commit":
            manifest = select_requirements_manifest(archive.dependency_manifests)
            effective = replace(
                resolution,
                pip_dependencies=(
                    read_staged_requirements(destination, manifest) if manifest is not None else ()
                ),
            )
        closure: ComfyRegistryWheelClosure | ComfyRegistryMixedWheelClosure
        if (
            source_session_factory is not None
            and source_store is not None
            and plan_comfy_registry_mixed_dependencies(effective.pip_dependencies).sources
        ):
            closure = await resolve_comfy_registry_reviewed_closure(
                effective.pip_dependencies,
                session_factory=source_session_factory,
                store=source_store,
                project_fetcher=project_client.fetch,
                metadata_fetcher=metadata_client.fetch,
                marker_environment=environment,
                supported_tags=tags,
                runtime_distributions=runtime,
            )
        else:
            closed = await drive_comfy_registry_wheel_closure(
                effective,
                project_fetcher=project_client.fetch,
                metadata_fetcher=metadata_client.fetch,
                marker_environment=environment,
                supported_tags=tags,
                runtime_distributions=runtime,
            )
            closure = closed.closure
    inputs = _ExecutionInputs(
        version=2 if isinstance(closure, ComfyRegistryMixedWheelClosure) else 1,
        resolution=resolution,
        archive=archive,
        archive_bytes=archive_bytes,
        closure=closure,
    )
    digest = hashlib.sha256(
        canonical_graph(inputs.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    plan = WorkflowPackageExecutionPlan(**inputs.model_dump(), plan_sha256=digest)
    plan.verify()
    return plan
