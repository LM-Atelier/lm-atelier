from __future__ import annotations

import hashlib
import re

import httpx
import pytest

from local_lm import (
    comfy_registry,
    comfy_registry_closure_driver,
    comfy_registry_sources,
    workflow_bindings,
)
from local_lm.comfy_registry import ComfyNodeResolution, ComfyRegistryClient
from local_lm.comfy_registry_sources import resolve_comfy_package_source
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement

# The Registry publishes some package ids in mixed case and finds them only in that case:
# a lower-cased lookup returns no node. Every stage that checks an id has to accept the
# published spelling, or a package that resolves is refused one step later.
_MIXED_CASE = "ExamplePack"
_COMMIT = "a" * 40
_REPOSITORY = "https://github.com/example/ExamplePack.git"


async def test_a_mixed_case_package_id_resolves_in_its_published_case() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["node_id"])
        node = {"id": _MIXED_CASE, "repository": _REPOSITORY, "status": "NodeStatusActive"}
        return httpx.Response(200, json={"nodes": [node]})

    client = ComfyRegistryClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.resolve(
            (WorkflowPackageRequirement(_MIXED_CASE, (_COMMIT,), ("ExampleNode",)),)
        )
    finally:
        await client.close()
    package = result.packages[0]
    assert seen == [_MIXED_CASE]
    assert package.error_code is None
    assert package.package_id == _MIXED_CASE
    assert package.install_kind == "git_commit"


def test_a_mixed_case_package_id_gets_an_exact_source_identity() -> None:
    record = (
        "github-commit:" + hashlib.sha256(f"{_REPOSITORY}{chr(0)}{_COMMIT}".encode()).hexdigest()
    )
    source = resolve_comfy_package_source(
        ComfyNodeResolution(
            package_id=_MIXED_CASE,
            declared_version=_COMMIT,
            node_types=("ExampleNode",),
            install_kind="git_commit",
            repository_url=_REPOSITORY,
            registry_record_id=record,
            download_url=f"https://codeload.github.com/example/ExamplePack/zip/{_COMMIT}",
        )
    )
    assert source.source_record_id == record


_CHECKS: list[re.Pattern[str]] = [
    comfy_registry._PACKAGE_ID,
    comfy_registry_closure_driver._PACKAGE_ID,
    comfy_registry_sources._PACKAGE_ID,
    workflow_bindings._PACKAGE_ID,
]


@pytest.mark.parametrize("check", _CHECKS, ids=lambda check: check.pattern)
@pytest.mark.parametrize("package_id", [_MIXED_CASE, "PACK2", "example-pack", "a.b_c-d"])
def test_every_package_id_check_accepts_published_spellings(
    check: re.Pattern[str], package_id: str
) -> None:
    assert check.fullmatch(package_id)


@pytest.mark.parametrize("check", _CHECKS, ids=lambda check: check.pattern)
@pytest.mark.parametrize(
    "package_id",
    ["", "../pack", "pack/../x", "-pack", ".pack", "Example Pack", "pack\n", "x" * 101],
)
def test_every_package_id_check_still_refuses_unsafe_ids(
    check: re.Pattern[str], package_id: str
) -> None:
    assert check.fullmatch(package_id) is None
