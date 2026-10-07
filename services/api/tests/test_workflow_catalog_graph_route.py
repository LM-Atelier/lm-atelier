"""Handing a discovered workflow's graph to the review a local file already gets.

A workflow found in Discover is a ComfyUI export somebody published. The route
returns that export and nothing else happens: no workflow is stored, trusted or
run. What makes it worth having is that its answer is exactly what the package
review's own first steps accept, so a catalog result and a downloaded file take
one road rather than two.

The source is the real CivitAI reader over a fake network wherever the reader's
own behaviour matters - which file is the graph, how an ambiguity is refused -
so these tests drive the route the way a person's click does rather than a
stand-in that returns whatever the test hands it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowRevision

pytestmark = pytest.mark.asyncio

EXPORT: dict[str, Any] = {
    "version": 0.4,
    "nodes": [
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": ["landscape_base.safetensors"],
        },
        {
            "id": 2,
            "type": "Note",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": ["Soft morning light over a valley."],
        },
    ],
    "links": [],
}
DOWNLOAD = "https://civitai.com/api/download/models/801"


def _file(name: str = "valley-workflow.json", **updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 901,
        "name": name,
        "type": "Workflow",
        "sizeKB": 1,
        "downloadUrl": DOWNLOAD,
    }
    value.update(updates)
    return value


class _Network:
    """The provider's two answers, and every address the reader asked for."""

    def __init__(self, files: list[dict[str, Any]], body: bytes) -> None:
        self.files = files
        self.body = body
        self.seen: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        if "model-versions" in str(request.url):
            return httpx.Response(200, json={"id": 801, "name": "v1", "files": self.files})
        return httpx.Response(200, content=self.body)


@pytest_asyncio.fixture
async def civitai(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[list[CivitaiCatalog]]:
    """Put a real CivitAI reader, over a fake network, behind the registry."""

    registry = app.state.services.catalog_sources
    registered = registry.get
    readers: list[CivitaiCatalog] = []

    def get(source_id: str) -> Any:
        if source_id == "civitai" and readers:
            return readers[-1]
        return registered(source_id)

    monkeypatch.setattr(registry, "get", get)
    yield readers
    for reader in readers:
        await reader.close()


def _serve(
    readers: list[CivitaiCatalog],
    tmp_path: Path,
    *,
    files: list[dict[str, Any]] | None = None,
    body: bytes | None = None,
) -> _Network:
    network = _Network(
        files if files is not None else [_file()],
        body if body is not None else json.dumps(EXPORT).encode("utf-8"),
    )
    readers.append(
        CivitaiCatalog(
            Settings(data_dir=tmp_path / "reader"),
            transport=httpx.MockTransport(network),
            sleep=asyncio.sleep,
        )
    )
    return network


def _stored_workflows() -> tuple[int, int]:
    with SessionLocal() as session:
        return (
            session.scalar(select(func.count(WorkflowDefinition.id))) or 0,
            session.scalar(select(func.count(WorkflowRevision.id))) or 0,
        )


async def test_a_published_export_comes_back_whole_and_nothing_is_stored(
    client: AsyncClient, civitai: list[CivitaiCatalog], tmp_path: Path
) -> None:
    network = _serve(civitai, tmp_path)
    before = _stored_workflows()

    response = await client.get("/api/workflow-catalog/versions/801/graph")

    assert response.status_code == 200
    assert response.json() == {"version_id": "801", "ui_graph": EXPORT}
    assert network.seen == ["https://civitai.com/api/v1/model-versions/801", DOWNLOAD]
    assert _stored_workflows() == before


async def test_what_comes_back_is_what_the_package_review_accepts(
    client: AsyncClient, civitai: list[CivitaiCatalog], tmp_path: Path
) -> None:
    """The review's first two calls take the returned graph exactly as given.

    Analysis reports on it, and the draft it stores is untrusted and not yet
    executable - the same state a downloaded export reaches through the same
    calls.
    """

    _serve(civitai, tmp_path)
    before = _stored_workflows()
    graph = (await client.get("/api/workflow-catalog/versions/801/graph")).json()["ui_graph"]

    analysis = await client.post("/api/workflows/packages/analyze", json={"ui_graph": graph})
    assert analysis.status_code == 200
    assert analysis.json()["node_count"] == 2
    assert [asset["filename"] for asset in analysis.json()["asset_references"]] == [
        "landscape_base.safetensors"
    ]

    draft = await client.post(
        "/api/workflows/packages/drafts",
        json={"ui_graph": graph, "name": "Valley", "operation": "text_to_image"},
    )
    assert draft.status_code == 201
    revision = draft.json()["revisions"][0]
    assert revision["trusted"] is False
    assert revision["api_graph_json"] == {}
    # The count the first case relies on does see what this app writes.
    assert _stored_workflows() == (before[0] + 1, before[1] + 1)


async def test_a_graph_that_is_not_an_export_is_refused_as_such(
    client: AsyncClient, civitai: list[CivitaiCatalog], tmp_path: Path
) -> None:
    """An API-format graph is valid JSON and a real workflow, but not one this can review."""

    api_format = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    _serve(civitai, tmp_path, body=json.dumps(api_format).encode("utf-8"))

    response = await client.get("/api/workflow-catalog/versions/801/graph")

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-catalog-graph-not-an-export"


async def test_an_ambiguous_version_is_refused_naming_what_it_carries(
    client: AsyncClient, civitai: list[CivitaiCatalog], tmp_path: Path
) -> None:
    network = _serve(
        civitai,
        tmp_path,
        files=[_file("first.json"), _file("second.json", id=902)],
    )

    response = await client.get("/api/workflow-catalog/versions/801/graph")

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-catalog-graph-unusable"
    assert "first.json" in response.json()["detail"]
    assert "second.json" in response.json()["detail"]
    # Refused before either file was read.
    assert network.seen == ["https://civitai.com/api/v1/model-versions/801"]


async def test_a_version_id_the_source_would_never_issue_is_refused_unasked(
    client: AsyncClient, civitai: list[CivitaiCatalog], tmp_path: Path
) -> None:
    network = _serve(civitai, tmp_path)

    response = await client.get("/api/workflow-catalog/versions/not-a-version/graph")

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-catalog-graph-unusable"
    assert network.seen == []


async def test_an_unreachable_provider_is_an_outage_not_a_bad_workflow(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fetch_workflow_graph(version_id: str) -> Any:
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(
        app.state.services.catalog_sources.get("civitai"),
        "fetch_workflow_graph",
        fetch_workflow_graph,
    )

    response = await client.get("/api/workflow-catalog/versions/801/graph")

    assert response.status_code == 503
    assert response.json()["code"] == "catalog-unavailable"


async def test_a_source_without_workflows_says_so(client: AsyncClient) -> None:
    """Hugging Face is registered and serves models, but no workflows."""

    response = await client.get(
        "/api/workflow-catalog/versions/801/graph", params={"source": "huggingface"}
    )

    assert response.status_code == 404
    assert response.json()["code"] == "catalog-source-serves-no-workflows"


async def test_an_unknown_source_is_refused_with_the_existing_shape(
    client: AsyncClient,
) -> None:
    response = await client.get(
        "/api/workflow-catalog/versions/801/graph", params={"source": "nowhere"}
    )

    assert response.status_code == 404
    assert response.json()["code"] == "catalog-source-not-found"


async def test_an_overlong_version_id_never_reaches_the_source(
    client: AsyncClient, civitai: list[CivitaiCatalog], tmp_path: Path
) -> None:
    network = _serve(civitai, tmp_path)

    response = await client.get(f"/api/workflow-catalog/versions/{'8' * 33}/graph")

    assert response.status_code == 422
    assert network.seen == []
