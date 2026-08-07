"""Package identities are derived from fresh workflow analysis before queueing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio


def _node(
    identifier: int,
    node_type: str,
    *,
    package: str = "example-pack",
    version: str = "1.2.3",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": node_type,
        "properties": {"cnr_id": package, "ver": version},
        "widgets_values": [],
    }


def _workflow(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": nodes,
        "links": [],
        "definitions": {"subgraphs": []},
    }


async def test_the_queued_job_uses_node_types_derived_from_the_graph(
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
    graph = _workflow([_node(1, "ExampleNode"), _node(2, "OtherNode")])

    queued = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "ui_graph": graph},
    )

    assert queued.status_code == 202
    job_id = queued.json()["id"]
    listed = await client.get("/api/jobs")
    payload = next(job for job in listed.json() if job["id"] == job_id)["payload_json"]
    assert payload["node_types"] == ["ExampleNode", "OtherNode"]


@pytest.mark.parametrize(
    ("package_id", "version", "code"),
    [
        ("different-pack", "1.2.3", "workflow-package-requirement-not-found"),
        ("example-pack", "9.9.9", "workflow-package-version-mismatch"),
    ],
)
async def test_refuses_a_selection_that_disagrees_with_the_graph(
    client: AsyncClient, package_id: str, version: str, code: str
) -> None:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": package_id,
            "version": version,
            "ui_graph": _workflow([_node(1, "ExampleNode")]),
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == code
    # Both sides named. A workflow can declare a version that cannot be
    # installed at all, and what the reader does next depends on knowing which
    # version was asked for and which the workflow wrote down - saying only
    # that they differ sends them off to find both.
    assert package_id in body["detail"]
    if code == "workflow-package-version-mismatch":
        assert version in body["detail"]
        assert "1.2.3" in body["detail"]
    jobs = (await client.get("/api/jobs")).json()
    assert all(job["kind"] != "registry_prepare" for job in jobs)


async def test_refuses_a_package_with_conflicting_workflow_versions(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": "example-pack",
            "version": "1.2.3",
            "ui_graph": _workflow(
                [
                    _node(1, "ExampleNode"),
                    _node(2, "OtherNode", version="2.0.0"),
                ]
            ),
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-package-version-mismatch"


async def test_refuses_browser_supplied_node_identities(client: AsyncClient) -> None:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": "example-pack",
            "version": "1.2.3",
            "ui_graph": _workflow([_node(1, "ExampleNode")]),
            "node_types": ["ClaimedByTheBrowser"],
        },
    )

    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden" and error["loc"][-1] == "node_types"
        for error in response.json()["detail"]
    )
