from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence

import httpx
import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_closure_driver as driver_module
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_closure_driver import (
    ComfyRegistryWheelClosureDriverError,
    ComfyRegistryWheelMetadataClient,
    drive_comfy_registry_wheel_closure,
)
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactManifest,
    resolve_comfy_registry_wheel_artifacts,
)

_TAG = "py3-none-any"
_SHA256 = "a" * 64


def _environment(**updates: str) -> dict[str, str]:
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    environment.update(updates)
    return environment


def _core_metadata(name: str, version: str, requirements: Sequence[str]) -> bytes:
    headers = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
    ]
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    return ("\n".join(headers) + "\n\n").encode()


def _wheel_file(
    name: str,
    version: str,
    metadata: bytes,
    *,
    metadata_available: bool = True,
) -> dict[str, object]:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    return {
        "filename": filename,
        "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
        "hashes": {"sha256": _SHA256},
        "requires-python": ">=3.12",
        "yanked": False,
        "size": 100,
        "core-metadata": (
            {"sha256": hashlib.sha256(metadata).hexdigest()} if metadata_available else False
        ),
    }


def _project(
    name: str,
    entries: Sequence[tuple[str, Sequence[str]]],
    *,
    metadata_available: bool = True,
) -> tuple[dict[str, object], dict[str, bytes]]:
    files: list[object] = []
    metadata_documents: dict[str, bytes] = {}
    for version, requirements in entries:
        metadata = _core_metadata(name, version, requirements)
        record = _wheel_file(
            name,
            version,
            metadata,
            metadata_available=metadata_available,
        )
        files.append(record)
        metadata_documents[str(record["filename"])] = metadata
    return (
        {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": files,
        },
        metadata_documents,
    )


def _resolution(*dependencies: str) -> ComfyNodeResolution:
    return ComfyNodeResolution(
        package_id="example-node",
        declared_version="1.2.3",
        node_types=("ExampleNode",),
        install_kind="registry_archive",
        registry_record_id="registry-record",
        download_url="https://api.comfy.org/nodes/example-node/versions/1.2.3/download",
        pip_dependencies=dependencies,
    )


class _Sources:
    def __init__(
        self,
        projects: Mapping[str, object],
        metadata: Mapping[str, bytes],
    ) -> None:
        self.projects = dict(projects)
        self.metadata = dict(metadata)
        self.project_calls: list[tuple[str, ...]] = []
        self.metadata_calls: list[tuple[str, ...]] = []

    async def fetch_projects(self, names: Sequence[str]) -> Mapping[str, object]:
        resolved = tuple(names)
        self.project_calls.append(resolved)
        return {name: self.projects[name] for name in resolved}

    async def fetch_metadata(
        self,
        manifest: ComfyRegistryWheelArtifactManifest,
    ) -> Mapping[str, bytes]:
        filenames = tuple(artifact.filename for artifact in manifest.artifacts)
        self.metadata_calls.append(filenames)
        return {filename: self.metadata[filename] for filename in filenames}


@pytest.mark.asyncio
async def test_drives_direct_and_transitive_rounds_to_one_verified_closure() -> None:
    alpha, alpha_metadata = _project("alpha", [("1.0", ["beta>=1"])])
    beta, beta_metadata = _project("beta", [("1.0", []), ("2.0", ["gamma>=1"])])
    gamma, gamma_metadata = _project("gamma", [("3.0", [])])
    sources = _Sources(
        {"alpha": alpha, "beta": beta, "gamma": gamma},
        {**alpha_metadata, **beta_metadata, **gamma_metadata},
    )
    progress: list[tuple[str, int, tuple[str, ...]]] = []

    async def record_progress(
        phase: driver_module.ClosurePhase,
        round_number: int,
        items: tuple[str, ...],
    ) -> None:
        progress.append((phase, round_number, items))

    resolution = _resolution("alpha==1.0")
    result = await drive_comfy_registry_wheel_closure(
        resolution,
        project_fetcher=sources.fetch_projects,
        metadata_fetcher=sources.fetch_metadata,
        marker_environment=_environment(),
        supported_tags=(_TAG,),
        progress=record_progress,
    )

    assert result.package_id == "example-node"
    assert result.package_version == "1.2.3"
    assert result.closure.complete is True
    assert result.closure.round_number == 2
    assert [artifact.name for artifact in result.closure.manifest.artifacts] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert (
        result.closure.manifest.declaration_sha256
        == plan_comfy_registry_dependencies(resolution.pip_dependencies).declaration_sha256
    )
    assert sources.project_calls == [("alpha",), ("beta",), ("gamma",)]
    assert len(sources.metadata_calls) == 3
    assert progress[0] == ("fetching_projects", 0, ("alpha",))
    assert progress[-1] == ("validating_closure", 2, ("gamma",))


@pytest.mark.asyncio
async def test_dependency_free_resolution_uses_no_network() -> None:
    async def unexpected_projects(names: Sequence[str]) -> Mapping[str, object]:
        raise AssertionError(names)

    async def unexpected_metadata(
        manifest: ComfyRegistryWheelArtifactManifest,
    ) -> Mapping[str, bytes]:
        raise AssertionError(manifest)

    result = await drive_comfy_registry_wheel_closure(
        _resolution(),
        project_fetcher=unexpected_projects,
        metadata_fetcher=unexpected_metadata,
        marker_environment=_environment(),
        supported_tags=(_TAG,),
    )

    assert result.closure.complete is True
    assert result.closure.manifest.artifacts == ()


@pytest.mark.asyncio
async def test_inactive_direct_marker_is_not_fetched() -> None:
    active, metadata = _project("active", [("2.0", [])])
    sources = _Sources({"active": active}, metadata)

    result = await drive_comfy_registry_wheel_closure(
        _resolution(
            'inactive==1.0; sys_platform == "win32"',
            'active==2.0; sys_platform == "linux"',
        ),
        project_fetcher=sources.fetch_projects,
        metadata_fetcher=sources.fetch_metadata,
        marker_environment=_environment(sys_platform="linux"),
        supported_tags=(_TAG,),
    )

    assert [artifact.name for artifact in result.closure.manifest.artifacts] == ["active"]
    assert sources.project_calls == [("active",)]


@pytest.mark.asyncio
async def test_unpinned_direct_dependency_refuses_before_network() -> None:
    called = False

    async def projects(names: Sequence[str]) -> Mapping[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
        await drive_comfy_registry_wheel_closure(
            _resolution("alpha>=1"),
            project_fetcher=projects,
            metadata_fetcher=lambda manifest: asyncio.sleep(0, result={}),
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )

    assert raised.value.code == "version_resolution_required"
    assert called is False


@pytest.mark.asyncio
async def test_invalid_registry_identity_refuses_before_network() -> None:
    called = False

    async def projects(names: Sequence[str]) -> Mapping[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
        await drive_comfy_registry_wheel_closure(
            ComfyNodeResolution(
                package_id="../outside",
                declared_version="1.2.3",
                node_types=("ExampleNode",),
                install_kind="registry_archive",
                registry_record_id="registry-record",
                pip_dependencies=("alpha==1.0",),
            ),
            project_fetcher=projects,
            metadata_fetcher=lambda manifest: asyncio.sleep(0, result={}),
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )

    assert raised.value.code == "invalid_resolution"
    assert called is False


@pytest.mark.asyncio
async def test_missing_project_mapping_preserves_typed_refusal() -> None:
    async def no_projects(names: Sequence[str]) -> Mapping[str, object]:
        return {}

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
        await drive_comfy_registry_wheel_closure(
            _resolution("alpha==1.0"),
            project_fetcher=no_projects,
            metadata_fetcher=lambda manifest: asyncio.sleep(0, result={}),
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )

    assert raised.value.code == "missing_project_metadata"


@pytest.mark.asyncio
async def test_wheel_without_hash_bound_metadata_refuses_before_metadata_fetch() -> None:
    alpha, metadata = _project(
        "alpha",
        [("1.0", [])],
        metadata_available=False,
    )
    sources = _Sources({"alpha": alpha}, metadata)

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
        await drive_comfy_registry_wheel_closure(
            _resolution("alpha==1.0"),
            project_fetcher=sources.fetch_projects,
            metadata_fetcher=sources.fetch_metadata,
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )

    assert raised.value.code == "metadata_unavailable"
    assert sources.metadata_calls == []


@pytest.mark.asyncio
async def test_metadata_hash_mismatch_preserves_typed_refusal() -> None:
    alpha, metadata = _project("alpha", [("1.0", [])])
    filename = next(iter(metadata))
    sources = _Sources({"alpha": alpha}, {filename: b"wrong"})

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
        await drive_comfy_registry_wheel_closure(
            _resolution("alpha==1.0"),
            project_fetcher=sources.fetch_projects,
            metadata_fetcher=sources.fetch_metadata,
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )

    assert raised.value.code == "core_metadata_hash_mismatch"


@pytest.mark.asyncio
async def test_project_fetch_cancellation_propagates() -> None:
    async def cancelled(names: Sequence[str]) -> Mapping[str, object]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await drive_comfy_registry_wheel_closure(
            _resolution("alpha==1.0"),
            project_fetcher=cancelled,
            metadata_fetcher=lambda manifest: asyncio.sleep(0, result={}),
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )


@pytest.mark.asyncio
async def test_progress_failure_does_not_fail_resolution() -> None:
    async def broken_progress(
        phase: driver_module.ClosurePhase,
        round_number: int,
        items: tuple[str, ...],
    ) -> None:
        raise RuntimeError((phase, round_number, items))

    result = await drive_comfy_registry_wheel_closure(
        _resolution(),
        project_fetcher=lambda names: asyncio.sleep(0, result={}),
        metadata_fetcher=lambda manifest: asyncio.sleep(0, result={}),
        marker_environment=_environment(),
        supported_tags=(_TAG,),
        progress=broken_progress,
    )

    assert result.closure.complete is True


@pytest.mark.asyncio
async def test_driver_enforces_its_round_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha, alpha_metadata = _project("alpha", [("1.0", ["beta>=1"])])
    sources = _Sources({"alpha": alpha}, alpha_metadata)
    monkeypatch.setattr(driver_module, "MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS", 0)

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
        await drive_comfy_registry_wheel_closure(
            _resolution("alpha==1.0"),
            project_fetcher=sources.fetch_projects,
            metadata_fetcher=sources.fetch_metadata,
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )

    assert raised.value.code == "closure_round_limit"
    assert sources.project_calls == [("alpha",)]


def _metadata_manifest(content: bytes) -> ComfyRegistryWheelArtifactManifest:
    project, _ = _project("alpha", [("1.0", [])])
    record = project["files"][0]
    assert isinstance(record, dict)
    record["core-metadata"] = {"sha256": hashlib.sha256(content).hexdigest()}
    return resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(["alpha==1.0"]),
        {"alpha": project},
        marker_environment=_environment(),
        supported_tags=(_TAG,),
    )


@pytest.mark.asyncio
async def test_metadata_client_fetches_exact_hash_bound_document() -> None:
    content = _core_metadata("alpha", "1.0", [])
    manifest = _metadata_manifest(content)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(f"{manifest.artifacts[0].filename}.metadata")
        return httpx.Response(
            200,
            headers={"content-length": str(len(content))},
            content=content,
        )

    client = ComfyRegistryWheelMetadataClient(transport=httpx.MockTransport(handler))
    try:
        documents = await client.fetch(manifest)
    finally:
        await client.close()

    assert documents == {manifest.artifacts[0].filename: content}


@pytest.mark.asyncio
async def test_metadata_client_rejects_redirect_and_hash_mismatch() -> None:
    content = _core_metadata("alpha", "1.0", [])
    manifest = _metadata_manifest(content)
    statuses = iter(
        [
            httpx.Response(302, headers={"location": "https://example.com/metadata"}),
            httpx.Response(200, content=b"wrong"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(statuses)

    client = ComfyRegistryWheelMetadataClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as redirect:
            await client.fetch(manifest)
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as mismatch:
            await client.fetch(manifest)
    finally:
        await client.close()

    assert redirect.value.code == "metadata_http_error"
    assert mismatch.value.code == "core_metadata_hash_mismatch"


@pytest.mark.asyncio
async def test_metadata_client_reports_rate_limit_and_size_bound() -> None:
    content = _core_metadata("alpha", "1.0", [])
    manifest = _metadata_manifest(content)
    responses = iter(
        [
            httpx.Response(429, headers={"retry-after": "17"}),
            httpx.Response(
                200,
                headers={"content-length": str(driver_module.MAX_WHEEL_CORE_METADATA_BYTES + 1)},
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    client = ComfyRegistryWheelMetadataClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as limited:
            await client.fetch(manifest)
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as oversized:
            await client.fetch(manifest)
    finally:
        await client.close()

    assert limited.value.code == "metadata_rate_limited"
    assert limited.value.retry_after_seconds == 17
    assert oversized.value.code == "core_metadata_too_large"
