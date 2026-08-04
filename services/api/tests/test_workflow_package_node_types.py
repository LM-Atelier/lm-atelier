"""What a prepared package is recorded as providing, and who decides it.

Preparation used to carry only a package id and a version, so composition
built a requirement declaring no node types at all and persistence correctly
refused it. The first fix sent the identities from the browser, which traded
one defect for a worse one: those identities are persisted and later read as
the package's capability, so a caller could prepare one package while claiming
another's nodes and review would be judging a claim rather than a finding.

The graph is analyzed again on the server. The browser chooses which package
to prepare; it never says what that package provides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio


def _node(identifier: int, node_type: str, package: str, version: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": node_type,
        "properties": {"cnr_id": package, "ver": version},
        "widgets_values": [],
    }


def _graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            _node(1, "PowerLoraLoader", "rgthree-comfy", "1.2.3"),
            _node(2, "PowerPrompt", "rgthree-comfy", "1.2.3"),
            {"id": 3, "type": "SaveImage", "properties": {"cnr_id": "comfy-core"}},
        ],
        "links": [],
    }


@pytest.fixture
def configured(settings: Settings, tmp_path: Path) -> Settings:
    settings.comfy_executable = tmp_path / "python.exe"
    settings.comfy_executable.write_bytes(b"runtime")
    settings.comfy_directory = tmp_path / "ComfyUI"
    settings.comfy_directory.mkdir()
    return settings


async def test_the_queued_job_carries_what_the_server_analyzed(
    client: AsyncClient, configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import api as api_module

    async def capture(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("captured; the job may fail after this")

    monkeypatch.setattr(api_module, "prepare_workflow_package", capture)

    queued = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "rgthree-comfy", "version": "1.2.3", "ui_graph": _graph()},
    )

    assert queued.status_code == 202
    job_id = queued.json()["id"]
    listed = await client.get("/api/jobs")
    payload = next(job for job in listed.json() if job["id"] == job_id)["payload_json"]

    # Derived from the graph, and complete: both nodes, not whichever the
    # caller felt like naming.
    assert sorted(payload["node_types"]) == ["PowerLoraLoader", "PowerPrompt"]


async def test_a_node_list_from_the_caller_is_not_accepted_at_all(
    client: AsyncClient, configured: Settings
) -> None:
    """The field does not exist. A request carrying one is a 422, not a hint."""
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": "rgthree-comfy",
            "version": "1.2.3",
            "ui_graph": _graph(),
            "node_types": ["WhateverIWant"],
        },
    )

    assert response.status_code == 422


async def test_refuses_a_package_the_workflow_does_not_name(
    client: AsyncClient, configured: Settings
) -> None:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "some-other-pack", "version": "1.2.3", "ui_graph": _graph()},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-package-not-in-analysis"


async def test_refuses_a_version_the_workflow_does_not_require(
    client: AsyncClient, configured: Settings
) -> None:
    """Preparing 9.9.9 of a package the graph pins to 1.2.3 is not that graph."""
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "rgthree-comfy", "version": "9.9.9", "ui_graph": _graph()},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-package-version-not-required"


async def test_refuses_a_graph_that_names_no_packages(
    client: AsyncClient, configured: Settings
) -> None:
    empty = {
        "version": 0.4,
        "nodes": [{"id": 1, "type": "SaveImage", "properties": {}}],
        "links": [],
    }
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "rgthree-comfy", "version": "1.2.3", "ui_graph": empty},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-package-not-in-analysis"
