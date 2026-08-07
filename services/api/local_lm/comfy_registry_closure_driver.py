from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.version import Version

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_dependencies import (
    ComfyRegistryDependencyError,
    ComfyRegistryDependencyPlan,
    plan_comfy_registry_dependencies,
)
from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    canonical_comfy_registry_runtime_distributions,
    comfy_registry_runtime_distribution_map,
)
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactError,
    ComfyRegistryWheelArtifactManifest,
    build_comfy_registry_wheel_artifact_manifest,
    comfy_registry_wheel_target_sha256,
    resolve_comfy_registry_wheel_artifacts,
    validate_comfy_registry_wheel_artifact_manifest,
)
from .comfy_registry_wheel_closure import (
    MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS,
    ComfyRegistryWheelClosure,
    ComfyRegistryWheelClosureError,
    advance_comfy_registry_wheel_closure,
    plan_comfy_registry_wheel_closure,
    validate_comfy_registry_wheel_closure,
)
from .comfy_registry_wheel_metadata import MAX_WHEEL_CORE_METADATA_BYTES
from .comfy_registry_wheel_selection import (
    ComfyRegistryWheelSelectionError,
    select_comfy_registry_wheel_versions,
)
from .network import shared_tls_context

logger = logging.getLogger(__name__)

MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES = 64 * 1024 * 1024
WHEEL_METADATA_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")

ClosurePhase = Literal[
    "fetching_projects",
    "selecting_wheels",
    "fetching_metadata",
    "validating_closure",
]
RegistryProjectFetcher = Callable[
    [Sequence[str]],
    Awaitable[Mapping[str, object]],
]
RegistryMetadataFetcher = Callable[
    [ComfyRegistryWheelArtifactManifest],
    Awaitable[Mapping[str, bytes]],
]
RegistryClosureProgress = Callable[
    [ClosurePhase, int, tuple[str, ...]],
    Awaitable[None],
]


class ComfyRegistryWheelClosureDriverError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retry_after_seconds: int | None = None,
        requirement: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_seconds = retry_after_seconds
        # The declared requirement a refusal is about, when it is about one.
        self.requirement = requirement


@dataclass(frozen=True)
class ComfyRegistryWheelClosureResult:
    package_id: str
    package_version: str
    closure: ComfyRegistryWheelClosure


class ComfyRegistryWheelMetadataClient:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "accept": "application/octet-stream, text/plain;q=0.9",
                "accept-encoding": "identity",
                "user-agent": "local-lm/0.1",
            },
            timeout=httpx.Timeout(30, read=120),
            follow_redirects=False,
            verify=shared_tls_context(),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch(
        self,
        manifest: ComfyRegistryWheelArtifactManifest,
    ) -> dict[str, bytes]:
        artifacts = _manifest_artifacts(manifest)
        documents: dict[str, bytes] = {}
        total_bytes = 0
        for artifact in artifacts:
            expected_sha256 = artifact.metadata_sha256
            if expected_sha256 is None:
                raise ComfyRegistryWheelClosureDriverError(
                    "metadata_unavailable",
                    f"Wheel {artifact.filename} lacks hash-bound core metadata",
                )
            try:
                content = await self._fetch_one(
                    f"{artifact.url}.metadata",
                    artifact.filename,
                    expected_sha256,
                )
            except ComfyRegistryWheelClosureDriverError:
                raise
            except httpx.HTTPError as exc:
                raise ComfyRegistryWheelClosureDriverError(
                    "metadata_network_error",
                    f"Wheel core metadata for {artifact.filename} could not be retrieved",
                ) from exc
            total_bytes += len(content)
            if total_bytes > MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES:
                raise ComfyRegistryWheelClosureDriverError(
                    "metadata_total_too_large",
                    "Wheel core metadata exceeds the aggregate size limit",
                )
            documents[artifact.filename] = content
        return documents

    async def _fetch_one(
        self,
        url: str,
        filename: str,
        expected_sha256: str,
    ) -> bytes:
        async with self._client.stream("GET", url) as response:
            if response.status_code == 429:
                raise ComfyRegistryWheelClosureDriverError(
                    "metadata_rate_limited",
                    "Wheel core metadata is temporarily rate limited",
                    retry_after_seconds=_retry_after(response),
                )
            if response.status_code != 200:
                raise ComfyRegistryWheelClosureDriverError(
                    "metadata_http_error",
                    f"Wheel core metadata returned HTTP {response.status_code}",
                )
            encoding = response.headers.get("content-encoding", "identity").strip().lower()
            if encoding not in {"", "identity"}:
                raise ComfyRegistryWheelClosureDriverError(
                    "encoded_core_metadata",
                    "Wheel core metadata used unsupported content encoding",
                )
            expected_size = _content_length(response)
            content = bytearray()
            async for chunk in response.aiter_bytes(WHEEL_METADATA_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > MAX_WHEEL_CORE_METADATA_BYTES:
                    raise ComfyRegistryWheelClosureDriverError(
                        "core_metadata_too_large",
                        f"Wheel core metadata for {filename} exceeds the size limit",
                    )
            if expected_size is not None and len(content) != expected_size:
                raise ComfyRegistryWheelClosureDriverError(
                    "core_metadata_size_mismatch",
                    f"Wheel core metadata size for {filename} does not match Content-Length",
                )
        resolved = bytes(content)
        if not hmac.compare_digest(hashlib.sha256(resolved).hexdigest(), expected_sha256):
            raise ComfyRegistryWheelClosureDriverError(
                "core_metadata_hash_mismatch",
                f"Wheel core metadata hash for {filename} does not match",
            )
        return resolved


async def drive_comfy_registry_wheel_closure(
    resolution: ComfyNodeResolution,
    *,
    project_fetcher: RegistryProjectFetcher,
    metadata_fetcher: RegistryMetadataFetcher,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    runtime_distributions: (Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution]) = (),
    progress: RegistryClosureProgress | None = None,
) -> ComfyRegistryWheelClosureResult:
    """Resolve one Registry package to a complete, target-bound wheel closure."""
    package_id, package_version, dependency_plan = _resolution_plan(resolution)
    try:
        runtime = canonical_comfy_registry_runtime_distributions(runtime_distributions)
        runtime_map = comfy_registry_runtime_distribution_map(runtime)
    except ComfyRegistryRuntimeError as exc:
        raise ComfyRegistryWheelClosureDriverError(exc.code, str(exc)) from exc
    try:
        target_sha256 = comfy_registry_wheel_target_sha256(
            marker_environment,
            supported_tags,
        )
    except ComfyRegistryWheelArtifactError as exc:
        raise _driver_error(exc) from exc
    project_names = _active_project_names(
        dependency_plan,
        marker_environment,
        runtime_map,
    )
    if project_names:
        await _publish_progress(progress, "fetching_projects", 0, project_names)
        project_documents = await _fetch_projects(project_fetcher, project_names)
        await _publish_progress(progress, "selecting_wheels", 0, project_names)
        try:
            manifest = resolve_comfy_registry_wheel_artifacts(
                dependency_plan,
                project_documents,
                marker_environment=marker_environment,
                supported_tags=supported_tags,
                runtime_distributions=runtime,
            )
        except ComfyRegistryWheelArtifactError as exc:
            raise _driver_error(exc) from exc
    else:
        try:
            manifest = build_comfy_registry_wheel_artifact_manifest(
                dependency_plan.declaration_sha256,
                target_sha256,
                (),
            )
        except ComfyRegistryWheelArtifactError as exc:
            raise _driver_error(exc) from exc

    metadata_documents = await _fetch_metadata(
        metadata_fetcher,
        manifest,
        round_number=0,
        progress=progress,
    )
    await _publish_progress(progress, "validating_closure", 0, project_names)
    try:
        closure = plan_comfy_registry_wheel_closure(
            manifest,
            metadata_documents,
            marker_environment=marker_environment,
            runtime_distributions=runtime,
        )
    except ComfyRegistryWheelClosureError as exc:
        raise _driver_error(exc) from exc

    for _ in range(MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS):
        if closure.complete:
            return _result(package_id, package_version, dependency_plan, closure)
        closure = await _advance_round(
            closure,
            metadata_documents,
            project_fetcher=project_fetcher,
            metadata_fetcher=metadata_fetcher,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
            progress=progress,
        )
    if closure.complete:
        return _result(package_id, package_version, dependency_plan, closure)
    raise ComfyRegistryWheelClosureDriverError(
        "closure_round_limit",
        "Wheel dependency closure exceeds the round limit",
    )


async def _advance_round(
    closure: ComfyRegistryWheelClosure,
    metadata_documents: dict[str, bytes],
    *,
    project_fetcher: RegistryProjectFetcher,
    metadata_fetcher: RegistryMetadataFetcher,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    progress: RegistryClosureProgress | None,
) -> ComfyRegistryWheelClosure:
    round_number = closure.round_number + 1
    projects = closure.pending_projects
    await _publish_progress(progress, "fetching_projects", round_number, projects)
    project_documents = await _fetch_projects(project_fetcher, projects)
    await _publish_progress(progress, "selecting_wheels", round_number, projects)
    try:
        selection = select_comfy_registry_wheel_versions(
            closure.metadata_plan,
            project_documents,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
        )
        selected_manifest = build_comfy_registry_wheel_artifact_manifest(
            selection.selection_sha256,
            selection.target_sha256,
            selection.artifacts,
        )
    except (ComfyRegistryWheelSelectionError, ComfyRegistryWheelArtifactError) as exc:
        raise _driver_error(exc) from exc
    selected_metadata = await _fetch_metadata(
        metadata_fetcher,
        selected_manifest,
        round_number=round_number,
        progress=progress,
    )
    overlap = set(metadata_documents) & set(selected_metadata)
    if overlap:
        raise ComfyRegistryWheelClosureDriverError(
            "closure_no_progress",
            "Wheel metadata repeats an artifact already present in the closure",
        )
    metadata_documents.update(selected_metadata)
    await _publish_progress(progress, "validating_closure", round_number, projects)
    prior_artifact_count = len(closure.manifest.artifacts)
    try:
        advanced = advance_comfy_registry_wheel_closure(
            closure,
            selection,
            metadata_documents,
            marker_environment=marker_environment,
        )
    except ComfyRegistryWheelClosureError as exc:
        raise _driver_error(exc) from exc
    if (
        advanced.round_number != round_number
        or len(advanced.manifest.artifacts) <= prior_artifact_count
    ):
        raise ComfyRegistryWheelClosureDriverError(
            "closure_no_progress",
            "Wheel dependency closure made no progress",
        )
    return advanced


def _resolution_plan(
    resolution: ComfyNodeResolution,
) -> tuple[str, str, ComfyRegistryDependencyPlan]:
    if (
        not isinstance(resolution, ComfyNodeResolution)
        or not resolution.resolved
        or not isinstance(resolution.package_id, str)
        or _PACKAGE_ID.fullmatch(resolution.package_id) is None
        or not isinstance(resolution.declared_version, str)
        or not (
            resolution.install_kind == "git_commit"
            and _COMMIT.fullmatch(resolution.declared_version)
            or resolution.install_kind == "registry_archive"
            and _SEMANTIC_VERSION.fullmatch(resolution.declared_version)
            and isinstance(resolution.registry_record_id, str)
            and bool(resolution.registry_record_id)
            and len(resolution.registry_record_id) <= 1_000
            and not _has_control(resolution.registry_record_id)
        )
    ):
        raise ComfyRegistryWheelClosureDriverError(
            "invalid_resolution",
            "Registry package resolution is incomplete",
        )
    try:
        plan = plan_comfy_registry_dependencies(resolution.pip_dependencies)
    except ComfyRegistryDependencyError as exc:
        raise _driver_error(exc) from exc
    return resolution.package_id, resolution.declared_version, plan


def _active_project_names(
    plan: ComfyRegistryDependencyPlan,
    marker_environment: Mapping[str, str],
    runtime_distributions: Mapping[str, str],
) -> tuple[str, ...]:
    active: list[str] = []
    seen: set[str] = set()
    environment = dict(marker_environment)
    environment["extra"] = ""
    for dependency in plan.dependencies:
        if dependency.marker and not Marker(dependency.marker).evaluate(environment=environment):
            continue
        if dependency.name in seen:
            raise ComfyRegistryWheelClosureDriverError(
                "overlapping_dependency_markers",
                f"Multiple Registry requirements target active package {dependency.name}",
            )
        seen.add(dependency.name)
        runtime_version = runtime_distributions.get(dependency.name)
        if runtime_version is not None:
            if Requirement(dependency.requirement).specifier.contains(
                Version(runtime_version),
                prereleases=True,
            ):
                continue
            raise ComfyRegistryWheelClosureDriverError(
                "managed_runtime_dependency_conflict",
                f"Managed runtime {dependency.name} {runtime_version} does not satisfy "
                f"{dependency.requirement}",
            )
        active.append(dependency.name)
    return tuple(sorted(active))


async def _fetch_projects(
    fetcher: RegistryProjectFetcher,
    names: tuple[str, ...],
) -> Mapping[str, object]:
    try:
        return await fetcher(names)
    except asyncio.CancelledError:
        raise
    except ComfyRegistryWheelClosureDriverError:
        raise
    except Exception as exc:
        raise _driver_error(exc, fallback_code="project_fetch_failed") from exc


async def _fetch_metadata(
    fetcher: RegistryMetadataFetcher,
    manifest: ComfyRegistryWheelArtifactManifest,
    *,
    round_number: int,
    progress: RegistryClosureProgress | None,
) -> dict[str, bytes]:
    artifacts = _manifest_artifacts(manifest)
    missing = tuple(artifact.filename for artifact in artifacts if artifact.metadata_sha256 is None)
    if missing:
        raise ComfyRegistryWheelClosureDriverError(
            "metadata_unavailable",
            f"Wheel {missing[0]} lacks hash-bound core metadata",
        )
    filenames = tuple(artifact.filename for artifact in artifacts)
    await _publish_progress(progress, "fetching_metadata", round_number, filenames)
    if not artifacts:
        return {}
    try:
        documents = await fetcher(manifest)
    except asyncio.CancelledError:
        raise
    except ComfyRegistryWheelClosureDriverError:
        raise
    except Exception as exc:
        raise _driver_error(exc, fallback_code="metadata_fetch_failed") from exc
    if not isinstance(documents, Mapping):
        raise ComfyRegistryWheelClosureDriverError(
            "invalid_core_metadata",
            "Wheel core metadata must be an object",
        )
    return dict(documents)


def _manifest_artifacts(
    manifest: ComfyRegistryWheelArtifactManifest,
) -> tuple[ComfyRegistryWheelArtifact, ...]:
    try:
        return validate_comfy_registry_wheel_artifact_manifest(manifest)
    except ComfyRegistryWheelArtifactError as exc:
        raise _driver_error(exc) from exc


def _result(
    package_id: str,
    package_version: str,
    dependency_plan: ComfyRegistryDependencyPlan,
    closure: ComfyRegistryWheelClosure,
) -> ComfyRegistryWheelClosureResult:
    try:
        validate_comfy_registry_wheel_closure(closure)
    except ComfyRegistryWheelClosureError as exc:
        raise _driver_error(exc) from exc
    if (
        not closure.complete
        or closure.manifest.declaration_sha256 != dependency_plan.declaration_sha256
    ):
        raise ComfyRegistryWheelClosureDriverError(
            "dependency_closure_mismatch",
            "Wheel dependency closure does not belong to this Registry package",
        )
    return ComfyRegistryWheelClosureResult(package_id, package_version, closure)


async def _publish_progress(
    callback: RegistryClosureProgress | None,
    phase: ClosurePhase,
    round_number: int,
    items: tuple[str, ...],
) -> None:
    if callback is None:
        return
    try:
        await callback(phase, round_number, items)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Could not publish Registry closure progress", exc_info=True)


def _content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        size = int(raw)
    except ValueError as exc:
        raise ComfyRegistryWheelClosureDriverError(
            "invalid_metadata_content_length",
            "Wheel core metadata returned an invalid Content-Length",
        ) from exc
    if size < 0:
        raise ComfyRegistryWheelClosureDriverError(
            "invalid_metadata_content_length",
            "Wheel core metadata returned an invalid Content-Length",
        )
    if size > MAX_WHEEL_CORE_METADATA_BYTES:
        raise ComfyRegistryWheelClosureDriverError(
            "core_metadata_too_large",
            "Wheel core metadata exceeds the size limit",
        )
    return size


def _retry_after(response: httpx.Response) -> int | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = int(raw)
    except ValueError:
        return None
    return seconds if 0 <= seconds <= 24 * 60 * 60 else None


def _has_control(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)


def _driver_error(
    exc: Exception,
    *,
    fallback_code: str = "closure_resolution_failed",
) -> ComfyRegistryWheelClosureDriverError:
    code = getattr(exc, "code", fallback_code)
    if not isinstance(code, str) or not code:
        code = fallback_code
    retry_after = getattr(exc, "retry_after_seconds", None)
    if not isinstance(retry_after, int) or isinstance(retry_after, bool):
        retry_after = None
    requirement = getattr(exc, "requirement", None)
    if not isinstance(requirement, str) or not requirement:
        requirement = None
    return ComfyRegistryWheelClosureDriverError(
        code,
        str(exc) or "Registry wheel closure resolution failed",
        retry_after_seconds=retry_after,
        requirement=requirement,
    )
