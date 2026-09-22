"""Resolve declared dependencies using reviewed local roots and remote wheel metadata."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence

from packaging.markers import Marker
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_registry_closure_driver import (
    MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES,
    ComfyRegistryWheelClosureDriverError,
    RegistryClosureProgress,
    RegistryMetadataFetcher,
    RegistryProjectFetcher,
    _active_project_names,
    _driver_error,
    _fetch_projects,
    _publish_progress,
)
from .comfy_registry_dependencies import ComfyRegistryDependencyError
from .comfy_registry_mixed_closure_driver import drive_comfy_registry_mixed_wheel_closure
from .comfy_registry_mixed_dependencies_v1 import (
    ComfyRegistryMixedDependencyPlan,
    ComfyRegistrySourceDeclaration,
    plan_comfy_registry_mixed_dependencies,
)
from .comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    canonical_comfy_registry_runtime_distributions,
    comfy_registry_runtime_distribution_map,
)
from .comfy_registry_source_artifacts import (
    ComfyRegistrySourceArtifactError,
    verified_reviewed_source_wheel,
)
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactError,
    comfy_registry_wheel_target_sha256,
    resolve_comfy_registry_wheel_artifacts,
)
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryReviewedWheelInput,
    ComfyRegistryWheelInputError,
    build_comfy_registry_wheel_input_manifest,
    wheel_input_from_verified_source,
)
from .comfy_registry_wheel_metadata import MAX_WHEEL_CORE_METADATA_BYTES


async def resolve_comfy_registry_reviewed_closure(
    declarations: Sequence[str],
    *,
    session_factory: Callable[[], Session],
    store: ArtifactStore,
    project_fetcher: RegistryProjectFetcher,
    metadata_fetcher: RegistryMetadataFetcher,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    runtime_distributions: Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution] = (),
    progress: RegistryClosureProgress | None = None,
) -> ComfyRegistryMixedWheelClosure:
    """Resolve exact reviewed roots; installation must renew the retained review identities."""
    environment = dict(marker_environment)
    tags = tuple(supported_tags)
    try:
        plan = plan_comfy_registry_mixed_dependencies(declarations)
        comfy_registry_wheel_target_sha256(environment, tags)
        runtime = canonical_comfy_registry_runtime_distributions(runtime_distributions)
        runtime_map = comfy_registry_runtime_distribution_map(runtime)
        sources = _active_sources(plan, environment, runtime_map)
        projects = _active_project_names(plan.remote, environment, runtime_map)
    except (
        ComfyRegistryDependencyError,
        ComfyRegistryWheelArtifactError,
        ComfyRegistryRuntimeError,
    ) as exc:
        raise _driver_error(exc) from exc
    context = ComfyRegistryReviewedInputContext(session_factory, store, environment, tags)
    try:
        local, metadata = await _read_work(lambda: _read_roots(sources, context))
        documents: Mapping[str, object] = {}
        if projects:
            await _publish_progress(progress, "fetching_projects", 0, projects)
            documents = await _fetch_projects(project_fetcher, projects)
            await _publish_progress(progress, "selecting_wheels", 0, projects)
        remote = resolve_comfy_registry_wheel_artifacts(
            plan.remote,
            documents,
            marker_environment=environment,
            supported_tags=tags,
            runtime_distributions=runtime,
        )
        manifest = build_comfy_registry_wheel_input_manifest(plan.declaration_sha256, remote, local)
        closure = await drive_comfy_registry_mixed_wheel_closure(
            manifest,
            metadata,
            project_fetcher=project_fetcher,
            metadata_fetcher=metadata_fetcher,
            marker_environment=environment,
            supported_tags=tags,
            runtime_distributions=runtime,
            progress=progress,
        )
        if local:
            await _read_work(lambda: context.validate(closure.manifest))
        return closure
    except (
        ComfyRegistrySourceArtifactError,
        ComfyRegistryWheelInputError,
        ComfyRegistryWheelArtifactError,
    ) as exc:
        raise _driver_error(exc) from exc


def _active_sources(
    plan: ComfyRegistryMixedDependencyPlan,
    marker_environment: Mapping[str, str],
    runtime: Mapping[str, str],
) -> tuple[ComfyRegistrySourceDeclaration, ...]:
    environment = dict(marker_environment)
    environment["extra"] = ""
    names = {
        item.name
        for item in plan.remote.dependencies
        if item.marker is None or Marker(item.marker).evaluate(environment=environment)
    }
    active: list[ComfyRegistrySourceDeclaration] = []
    for item in plan.sources:
        if item.marker and not Marker(item.marker).evaluate(environment=environment):
            continue
        if item.name in names:
            raise ComfyRegistryWheelClosureDriverError(
                "overlapping_dependency_markers",
                f"Multiple Registry requirements target active package {item.name}",
            )
        if item.name in runtime:
            raise ComfyRegistryWheelClosureDriverError(
                "managed_runtime_source_conflict",
                f"Reviewed source {item.name} cannot replace a managed runtime distribution",
                requirement=item.declaration,
            )
        names.add(item.name)
        active.append(item)
    return tuple(active)


def _read_roots(
    sources: tuple[ComfyRegistrySourceDeclaration, ...],
    context: ComfyRegistryReviewedInputContext,
) -> tuple[tuple[ComfyRegistryReviewedWheelInput, ...], dict[str, bytes]]:
    if not sources:
        return (), {}
    local: list[ComfyRegistryReviewedWheelInput] = []
    metadata: dict[str, bytes] = {}
    total = 0
    with context.session_factory() as session:
        for source in sources:
            wheel = verified_reviewed_source_wheel(
                session, context.store, declaration=source.declaration
            )
            item = wheel_input_from_verified_source(
                wheel,
                marker_environment=context.marker_environment,
                supported_tags=context.supported_tags,
            )
            size = len(wheel.core_metadata)
            if size > MAX_WHEEL_CORE_METADATA_BYTES:
                raise ComfyRegistryWheelClosureDriverError(
                    "core_metadata_too_large", "Wheel core metadata exceeds the size limit"
                )
            total += size
            if total > MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES:
                raise ComfyRegistryWheelClosureDriverError(
                    "metadata_total_too_large",
                    "Wheel core metadata exceeds the aggregate size limit",
                )
            local.append(item)
            metadata[item.filename] = wheel.core_metadata
    return tuple(local), metadata


async def _read_work[T](work: Callable[[], T]) -> T:
    """Finish a blocking review read before cancellation releases its caller."""
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise
