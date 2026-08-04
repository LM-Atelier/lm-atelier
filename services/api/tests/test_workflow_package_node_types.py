"""The node identities a prepared package promises to provide.

Preparation used to carry only a package id and a version. Composition then
built a requirement providing no node types at all, persistence correctly
refused an empty node identity, and so a prepared package could never reach
the activation contract honestly. What the analysis showed the user has to
survive the queue.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio

_OVERLONG = "N" * 201
_WITH_CONTROL = "Example\tNode"


async def test_the_queued_job_keeps_the_node_types_it_was_given(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from local_lm import api as api_module

    settings.comfy_executable = tmp_path / "python.exe"
    settings.comfy_executable.write_bytes(b"runtime")
    settings.comfy_directory = tmp_path / "ComfyUI"
    settings.comfy_directory.mkdir()

    captured: dict[str, object] = {}

    async def capture(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise RuntimeError("captured; the job may fail after this")

    monkeypatch.setattr(api_module, "prepare_workflow_package", capture)

    queued = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": "example-pack",
            "version": "1.2.3",
            "node_types": ["ExampleNode", "OtherNode"],
        },
    )

    assert queued.status_code == 202
    job_id = queued.json()["id"]
    listed = await client.get("/api/jobs")
    payload = next(job for job in listed.json() if job["id"] == job_id)["payload_json"]

    assert payload["node_types"] == ["ExampleNode", "OtherNode"]


@pytest.mark.parametrize(
    ("node_types", "code"),
    [
        ([""], "workflow-package-node-type-invalid"),
        (["   "], "workflow-package-node-type-invalid"),
        ([_WITH_CONTROL], "workflow-package-node-type-invalid"),
        ([_OVERLONG], "workflow-package-node-type-invalid"),
        (["ExampleNode", "ExampleNode"], "workflow-package-node-type-repeated"),
        (["ExampleNode", "  ExampleNode  "], "workflow-package-node-type-repeated"),
    ],
)
async def test_refuses_an_identity_persistence_would_reject(
    client: AsyncClient, node_types: list[str], code: str
) -> None:
    """The rules persistence enforces, applied before anything is queued.

    A job that must fail on its first step is a worse answer than a typed 422,
    and relaxing persistence to accept these instead would be worse still.
    """
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "node_types": node_types},
    )

    assert response.status_code == 422
    assert response.json()["code"] == code


async def test_refuses_a_request_naming_no_node_types_at_all(client: AsyncClient) -> None:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "node_types": []},
    )

    assert response.status_code == 422
