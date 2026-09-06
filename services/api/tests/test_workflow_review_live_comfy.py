from __future__ import annotations

import asyncio
import io
import os
import socket
import uuid
from pathlib import Path

import pytest
from httpx2 import AsyncClient
from PIL import Image
from test_custom_node_source_identity import _git

from local_lm.db import SessionLocal
from local_lm.models import CustomNodeInstall

_NODE_SOURCE = """import torch

class ConstructedSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"aspect_ratio": (["square", "portrait"],)}}
    RETURN_TYPES = ("INT", "INT")
    FUNCTION = "size"
    CATEGORY = "constructed"
    def size(self, aspect_ratio):
        return (64, 96) if aspect_ratio == "portrait" else (64, 64)

class ConstructedImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT",), "height": ("INT",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image"
    CATEGORY = "constructed"
    def image(self, width, height):
        return (torch.full((1, height, width, 3), 0.4),)
"""

_GRAPH = {
    "1": {"class_type": "ConstructedSize", "inputs": {"aspect_ratio": "portrait"}},
    "2": {
        "class_type": "ConstructedImage",
        "inputs": {"width": ["1", 0], "height": ["1", 1]},
    },
    "3": {
        "class_type": "SaveImage",
        "inputs": {"images": ["2", 0], "filename_prefix": "constructed-review"},
    },
}


@pytest.fixture
def settings(settings):
    executable = os.environ.get("LM_ATELIER_TEST_COMFY_EXECUTABLE")
    directory = os.environ.get("LM_ATELIER_TEST_COMFY_DIRECTORY")
    if not executable or not directory:
        pytest.skip("An isolated ComfyUI runtime is required for live workflow acceptance")
    settings.comfy_executable = Path(executable)
    settings.comfy_directory = Path(directory)
    assert settings.comfy_executable.is_file()
    assert (settings.comfy_directory / "main.py").is_file()
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    settings.comfy_url = f"http://127.0.0.1:{port}"
    settings.media_engine = "comfyui"
    settings.worker_startup_seconds = 120
    return settings


@pytest.mark.parametrize("change_source", [False, True], ids=["render", "changed-code-refusal"])
async def test_reviewed_revision_uses_real_managed_comfy(
    client: AsyncClient, app, settings, monkeypatch, change_source: bool
) -> None:
    # This opt-in test uses a disposable runtime and profile. The real supervisor,
    # HTTP adapter, review API, dispatch, renderer and artifact ingest all run.
    identifier = "node_live_" + uuid.uuid4().hex[:20]
    source = settings.custom_node_dir / ("lm-atelier-" + identifier)
    source.mkdir(parents=True)
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "ajccarlson")
    _git(source, "config", "user.email", "32660587+ajccarlson@users.noreply.github.com")
    _git(source, "config", "core.autocrlf", "false")
    (source / "node.py").write_text(_NODE_SOURCE, encoding="utf-8")
    (source / "__init__.py").write_text(
        "from .node import ConstructedSize, ConstructedImage\n"
        "NODE_CLASS_MAPPINGS = {"
        "'ConstructedSize': ConstructedSize, 'ConstructedImage': ConstructedImage}\n",
        encoding="utf-8",
    )
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "Add constructed image nodes")
    revision = _git(source, "rev-parse", "HEAD")
    with SessionLocal() as session:
        session.add(
            CustomNodeInstall(
                id=identifier,
                name="Constructed image nodes",
                source_url="https://github.com/example/constructed-image-nodes.git",
                revision=revision,
                installed_path=source.name,
                tree_hash=_git(source, "rev-parse", "HEAD^{tree}"),
                trusted=True,
                active=True,
                security_json={
                    "trusted_by_local_user": True,
                    "reviewed_at": "2026-09-06T00:00:00Z",
                    "node_types": ["ConstructedSize", "ConstructedImage"],
                },
            )
        )
        session.commit()
    services = app.state.services
    real_replace = services.processes._replace

    async def cpu_replace(name, command, *args, **kwargs):
        if name == "media":
            command = [*command, "--cpu"]
        return await real_replace(name, command, *args, **kwargs)

    monkeypatch.setattr(services.processes, "_replace", cpu_replace)
    async with services.scheduler.lease("primary"):
        await services.processes.start_media()
    worker = next(item for item in services.processes.statuses() if item.name == "media")
    assert worker.managed and worker.running and worker.state == "ready" and worker.pid
    assert (await services.engines.media.capabilities()).healthy

    created = await client.post(
        "/api/workflows",
        json={
            "name": "Constructed portrait review",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": {"custom_nodes": [{"id": identifier, "revision": revision}]},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    revision_id = workflow["current_revision_id"]
    review_url = f"/api/workflows/{workflow['id']}/revisions/{revision_id}/review"
    preview = await client.get(review_url)
    assert preview.status_code == 200, preview.text
    assert preview.json()["can_approve"] is True
    approved = await client.post(
        review_url,
        json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]},
    )
    assert approved.status_code == 200 and approved.json()["trusted"] is True

    submitted = []
    generate = services.engines.media.generate

    async def capture(request):
        submitted.append(request)
        async for event in generate(request):
            yield event

    monkeypatch.setattr(services.engines.media, "generate", capture)
    profile = await client.post(
        "/api/profiles",
        json={"name": "Constructed CPU image", "role": "image", "engine": "comfyui"},
    )
    assert profile.status_code == 201, profile.text
    chat = (await client.post("/api/chats", json={"title": "Constructed CPU review"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": profile.json()["id"]},
    )
    assert selected.status_code == 200, selected.text
    async with services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Create a plain grey test image",
                "mode": "image",
                "workflow_revision_id": revision_id,
            },
        )
        if change_source:
            (source / "node.py").write_text(_NODE_SOURCE + "\nCHANGED = True\n", encoding="utf-8")
    assert accepted.status_code == 202, accepted.text
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    deadline = asyncio.get_running_loop().time() + 45
    while asyncio.get_running_loop().time() < deadline:
        current = (await client.get(f"/api/runs/{run['id']}")).json()
        if current["status"] in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Constructed ComfyUI run did not terminate")
    if change_source:
        assert current["status"] == "failed", current
        assert submitted == []
    else:
        assert current["status"] == "complete", current
        assert len(submitted) == 1 and submitted[0].workflow == _GRAPH
        outputs = current["provenance_json"]["outputs"]
        assert len(outputs) == 1
        content = await client.get(f"/api/artifacts/{outputs[0]['artifact_id']}/content")
        assert content.status_code == 200
        with Image.open(io.BytesIO(content.content)) as rendered:
            assert rendered.size == (64, 96)
            assert rendered.mode == "RGB"
