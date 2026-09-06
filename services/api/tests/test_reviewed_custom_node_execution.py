from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from httpx2 import AsyncClient
from test_custom_node_source_identity import _git
from test_custom_node_source_identity import installed_source as installed_source
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import CustomNodeInstall, WorkflowRevision

_NODE_SOURCE = """class ConstructedSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"aspect_ratio": (["square", "portrait"],)}}
    RETURN_TYPES = ("INT", "INT")
    FUNCTION = "size"
    CATEGORY = "constructed"
    def size(self, aspect_ratio):
        return (512, 768) if aspect_ratio == "portrait" else (512, 512)
"""
_GRAPH = {"1": {"class_type": "ConstructedSize", "inputs": {"aspect_ratio": "portrait"}}}


@pytest.fixture
async def custom_workflow(client: AsyncClient, app, installed_source, monkeypatch):
    _, install, root = installed_source
    (root / "node.py").write_text(_NODE_SOURCE, encoding="utf-8")
    (root / "__init__.py").write_text(
        "from .node import ConstructedSize\n"
        'NODE_CLASS_MAPPINGS = {"ConstructedSize": ConstructedSize}\n',
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "Define constructed node")
    install.revision = _git(root, "rev-parse", "HEAD")
    install.tree_hash = _git(root, "rev-parse", "HEAD^{tree}")
    install.security_json = {
        "trusted_by_local_user": True,
        "reviewed_at": "2026-09-06T00:00:00Z",
        "node_types": ["ConstructedSize"],
    }
    with SessionLocal() as session:
        session.add(install)
        session.commit()

    async def object_info():
        return {
            "ConstructedSize": {
                "python_module": "custom_nodes.constructed",
                "input": {"required": {"aspect_ratio": [["square", "portrait"], {}]}},
                "output": ["INT", "INT"],
            }
        }

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info, raising=False)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Constructed portrait workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": {"custom_nodes": [{"id": install.id, "revision": install.revision}]},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    return workflow, install.id, root


async def _approve_custom(client: AsyncClient, custom_workflow):
    workflow, _, _ = custom_workflow
    revision_id = workflow["current_revision_id"]
    url = f"/api/workflows/{workflow['id']}/revisions/{revision_id}/review"
    preview = await client.get(url)
    assert preview.status_code == 200, preview.text
    assert preview.json()["can_approve"] is True
    approved = await client.post(
        url,
        json={
            "action": "approve",
            "subject_sha256": preview.json()["subject_sha256"],
        },
    )
    assert approved.status_code == 200, approved.text
    return approved.json()


async def test_review_records_the_separately_pinned_custom_node(
    client: AsyncClient, custom_workflow
) -> None:
    _, install_id, _ = custom_workflow
    approved = await _approve_custom(client, custom_workflow)
    assert approved["trusted"] is True
    assert approved["packages"][0]["id"] == install_id
    with SessionLocal() as session:
        install = session.get(CustomNodeInstall, install_id)
        assert install is not None
        assert approved["packages"][0]["commit"] == install.revision
        assert approved["packages"][0]["tree"] == install.tree_hash
        assert approved["packages"][0]["source"] == install.source_url


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("source_url", "https://github.com/example/another-node.git"),
        ("revision", "a" * 40),
        ("tree_hash", "b" * 40),
        ("active", False),
        ("trusted", False),
        (
            "security_json",
            {
                "trusted_by_local_user": True,
                "reviewed_at": "changed",
                "node_types": ["ConstructedSize"],
            },
        ),
    ],
)
async def test_changed_node_identity_invalidates_workflow_execution(
    client: AsyncClient, custom_workflow, app, settings, field, replacement
) -> None:
    await _approve_custom(client, custom_workflow)
    from local_lm.workflow_review_runtime import verify_workflow_review_runtime

    workflow, install_id, _ = custom_workflow
    services = app.state.services
    with SessionLocal() as session:
        install = session.get(CustomNodeInstall, install_id)
        assert install is not None
        setattr(install, field, replacement)
        session.commit()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, workflow["current_revision_id"])
        assert revision is not None
        with pytest.raises(ValueError):
            await verify_workflow_review_runtime(
                settings, services.processes, services.engines.media, session, revision
            )


@pytest.mark.parametrize("change_source", [False, True])
async def test_explicit_reviewed_custom_workflow_reaches_dispatch_only_with_current_source(
    custom_workflow, client: AsyncClient, app, settings, monkeypatch, change_source: bool
) -> None:
    workflow, _, root = custom_workflow
    await _approve_custom(client, custom_workflow)
    services = app.state.services
    settings.media_engine = "comfyui"
    # Keep the real selection, review, source and submission checks. The fixture
    # replaces only worker startup and the adapter boundary, not review authority.
    monkeypatch.setattr(services.orchestrator, "_ensure_media_worker", AsyncMock())
    captured = []
    generate = services.engines.media.generate

    async def capture(request):
        captured.append(request)
        async for event in generate(request):
            yield event

    monkeypatch.setattr(services.engines.media, "generate", capture)
    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Constructed media profile",
            "role": "image",
            "engine": "comfyui",
        },
    )
    assert profile.status_code == 201, profile.text
    chat = (await client.post("/api/chats", json={"title": "Exact custom workflow"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={
            "active_image_profile_id": profile.json()["id"],
        },
    )
    assert selected.status_code == 200, selected.text
    async with services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Draw a blue bowl",
                "mode": "image",
                "workflow_revision_id": workflow["current_revision_id"],
            },
        )
        if change_source:
            (root / "node.py").write_text(_NODE_SOURCE + "\nCHANGED = True\n", encoding="utf-8")
    assert accepted.status_code == 202, accepted.text
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == workflow["current_revision_id"]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        current = (await client.get(f"/api/runs/{run['id']}")).json()
        if current["status"] in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.03)
    else:
        raise AssertionError("constructed run did not terminate")
    if change_source:
        assert current["status"] == "failed", current
        assert captured == []
    else:
        assert current["status"] == "complete", current
        assert len(captured) == 1
        assert captured[0].workflow == _GRAPH
