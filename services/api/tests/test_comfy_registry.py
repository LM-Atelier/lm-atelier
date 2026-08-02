from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from local_lm.comfy_registry import (
    MAX_REGISTRY_DEPENDENCIES,
    MAX_REGISTRY_RESPONSE_BYTES,
    ComfyRegistryClient,
)
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement


def requirement(
    package_id: str = "example-pack",
    versions: tuple[str, ...] = ("1.2.3",),
) -> WorkflowPackageRequirement:
    return WorkflowPackageRequirement(package_id, versions, ("ExampleNode",))


def node_payload(**updates: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": "example-pack",
        "repository": "https://github.com/example/example-pack",
        "status": "NodeStatusActive",
    }
    node.update(updates)
    return {"nodes": [node]}


def version_payload(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": "record-123",
        "node_id": "example-pack",
        "version": "1.2.3",
        "status": "NodeVersionStatusActive",
        "deprecated": False,
        "downloadUrl": "https://cdn.comfy.org/example-pack/1.2.3.zip",
        "dependencies": ["pillow>=12"],
        "tags_admin": [],
    }
    value.update(updates)
    return value


async def resolve_with(
    handler: Callable[[httpx.Request], httpx.Response],
    *requirements: WorkflowPackageRequirement,
) -> Any:
    client = ComfyRegistryClient(transport=httpx.MockTransport(handler))
    try:
        return await client.resolve(requirements)
    finally:
        await client.close()


async def test_semantic_registry_version_resolves_to_reviewable_archive() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/nodes":
            assert request.url.params["node_id"] == "example-pack"
            assert request.url.params["limit"] == "2"
            return httpx.Response(200, json=node_payload())
        return httpx.Response(200, json=version_payload(deprecated=True))

    result = await resolve_with(handler, requirement())
    package = result.packages[0]
    assert calls == ["/nodes", "/nodes/example-pack/versions/1.2.3"]
    assert result.metadata_resolved
    assert package.install_kind == "registry_archive"
    assert package.repository_url == "https://github.com/example/example-pack.git"
    assert package.registry_record_id == "record-123"
    assert package.download_url == "https://cdn.comfy.org/example-pack/1.2.3.zip"
    assert package.pip_dependencies == ("pillow>=12",)
    assert package.warnings == ("deprecated_version",)


async def test_commit_pin_uses_git_without_requesting_a_registry_version() -> None:
    commit = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=node_payload())

    result = await resolve_with(handler, requirement(versions=(commit,)))
    package = result.packages[0]
    assert calls == 1
    assert package.install_kind == "git_commit"
    assert package.declared_version == commit.lower()
    assert package.download_url is None
    assert package.warnings == (
        "source_review_required",
        "dependency_manifest_review_required",
    )


async def test_exact_local_package_resolution_skips_registry_network() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("resolved local packages must not reach the network")

    local = WorkflowPackageRequirement(
        "example-pack",
        ("1.2.3",),
        ("ExampleNode",),
        locally_resolved=True,
    )
    result = await resolve_with(handler, local)
    assert result.metadata_resolved
    assert result.packages[0].install_kind == "already_installed"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (requirement(versions=()), "unversioned_custom_node_package"),
        (
            requirement(versions=("1.0.0", "2.0.0")),
            "conflicting_custom_node_versions",
        ),
        (requirement(package_id="../pack"), "invalid_package_id"),
        (requirement(versions=("latest",)), "unsupported_package_version"),
        (requirement(versions=("1x2y3",)), "unsupported_package_version"),
    ],
)
async def test_invalid_requirements_fail_before_network(
    value: WorkflowPackageRequirement,
    code: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid requirements must not reach the network")

    result = await resolve_with(handler, value)
    assert result.packages[0].error_code == code
    assert not result.metadata_resolved


@pytest.mark.parametrize(
    ("node", "code"),
    [
        ({"nodes": []}, "registry_package_not_found"),
        (
            node_payload(status="NodeStatusBanned"),
            "registry_package_inactive",
        ),
        (
            node_payload(repository="https://attacker.invalid/package"),
            "unsupported_package_repository",
        ),
    ],
)
async def test_unusable_registry_packages_return_typed_errors(
    node: dict[str, Any],
    code: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=node)

    result = await resolve_with(handler, requirement())
    assert result.packages[0].error_code == code


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"node_id": "other-pack"}, "registry_identity_mismatch"),
        ({"version": "9.9.9"}, "registry_identity_mismatch"),
        ({"status": "NodeVersionStatusBanned"}, "registry_version_inactive"),
        ({"tags_admin": ["security-review"]}, "registry_security_warning"),
        (
            {"downloadUrl": "https://attacker.invalid/package.zip"},
            "invalid_registry_metadata",
        ),
        (
            {"dependencies": ["package"] * (MAX_REGISTRY_DEPENDENCIES + 1)},
            "invalid_registry_metadata",
        ),
    ],
)
async def test_unusable_registry_versions_return_typed_errors(
    updates: dict[str, Any],
    code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/nodes":
            return httpx.Response(200, json=node_payload())
        return httpx.Response(200, json=version_payload(**updates))

    result = await resolve_with(handler, requirement())
    assert result.packages[0].error_code == code


async def test_missing_registry_version_is_reported_without_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/nodes":
            return httpx.Response(200, json=node_payload())
        return httpx.Response(404, json={"message": "not found"})

    result = await resolve_with(handler, requirement())
    assert result.packages[0].error_code == "registry_version_not_found"


async def test_bounded_retry_after_is_honored_once() -> None:
    calls = 0
    delays: list[float] = []

    async def sleep(value: float) -> None:
        delays.append(value)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0.25"})
        if request.url.path == "/nodes":
            return httpx.Response(200, json=node_payload())
        return httpx.Response(200, json=version_payload())

    client = ComfyRegistryClient(
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.resolve([requirement()])
    finally:
        await client.close()
    assert result.metadata_resolved
    assert calls == 3
    assert delays == [0.25]


async def test_oversized_registry_response_is_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b" " * (MAX_REGISTRY_RESPONSE_BYTES + 1),
            headers={"content-type": "application/json"},
        )

    with pytest.raises(ValueError, match="size limit"):
        await resolve_with(handler, requirement())
