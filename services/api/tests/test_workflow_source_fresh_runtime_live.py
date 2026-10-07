"""Install a missing runtime through one source approval and render its workflow."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database
from local_lm.models import (
    Job,
    ModelInstall,
    WorkflowActivation,
    WorkflowInstallOffer,
    WorkflowRevision,
)
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner


def _source_graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "EmptyImage",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": [64, 96, 1, 0x666666],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "widgets_values": ["constructed-installation"],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
    }


@pytest.fixture
def settings(
    tmp_path_factory: pytest.TempPathFactory, migrated_database_template: Path
) -> Settings:
    if os.name != "nt" or os.environ.get("LM_ATELIER_TEST_PROVISION_COMFY_SOURCE") != "1":
        pytest.skip("An isolated Windows workflow runtime installation must be requested")
    data = tmp_path_factory.mktemp("runtime") / "data"
    state = data / "state"
    state.mkdir(parents=True)
    shutil.copyfile(migrated_database_template, state / "local-lm.sqlite3")
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    configured = Settings(
        data_dir=data,
        dev=True,
        chat_engine="mock",
        media_engine="comfyui",
        comfy_executable=None,
        comfy_directory=None,
        comfy_url=f"http://127.0.0.1:{port}",
        worker_startup_seconds=120,
    )
    configure_database(configured)
    return configured


def _cache_approved_archives(runtimes: RuntimeProvisioner) -> None:
    definition = runtimes._definition("comfyui")
    asset = definition["runtime_assets"]["windows-x86_64-nvidia-cu13"]
    for variable, selected in (
        ("LM_ATELIER_TEST_RUNTIME_ARCHIVE_FILE", asset),
        ("LM_ATELIER_TEST_RUNTIME_OVERLAY_FILE", asset["security_overlays"][0]),
    ):
        cached = os.environ.get(variable)
        assert cached, "Previously verified archive inputs must be supplied"
        source = Path(cached)
        assert source.is_file() and not source.is_symlink()
        assert source.stat().st_size == selected["size_bytes"]
        with source.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == selected["sha256"]
        destination = runtimes._archive_path("comfyui", definition, selected)
        assert not destination.exists()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


@pytest.mark.asyncio
async def test_one_source_approval_provisions_activates_and_renders_a_fresh_runtime(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    runtimes = services.runtimes
    runtimes._platform_keys["comfyui"] = "windows-x86_64-nvidia-cu13"
    assert settings.comfy_executable is None and settings.comfy_directory is None
    assert not runtime_config_path(settings.data_dir).exists()
    assert not any(item.name == "media" and item.running for item in services.processes.statuses())
    with SessionLocal() as session:
        assert list(session.scalars(select(ModelInstall))) == []
    await asyncio.to_thread(_cache_approved_archives, runtimes)
    approved = await asyncio.to_thread(runtimes.preflight, "comfyui")
    assert approved.operation == "install_managed"
    original_replace = services.processes._replace

    async def cpu_replace(name: str, command: list[str], *args: Any, **kwargs: Any) -> None:
        await original_replace(
            name, [*command, "--cpu"] if name == "media" else command, *args, **kwargs
        )

    monkeypatch.setattr(services.processes, "_replace", cpu_replace)
    requests: list[str] = []

    def no_download(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        raise AssertionError("The approved runtime archives are already cached")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_download)) as archive_client:
        monkeypatch.setattr(runtimes, "_client", archive_client)
        try:
            graph = _source_graph()
            preview = await client.post(
                "/api/workflows/packages/install-plans",
                json={
                    "name": "Constructed fresh portrait workflow",
                    "operation": "text_to_image",
                    "ui_graph": graph,
                    "dependencies": {"version": 1, "slots": []},
                    "selections": [],
                },
            )
            assert preview.status_code == 201, preview.text
            plan = preview.json()
            assert plan["can_accept"] and plan["blockers"] == [], plan
            assert plan["runtime_plan"] == asdict(approved)
            assert plan["total_download_bytes"] == approved.download_bytes > 0
            install_url = f"/api/workflow-install-offers/{plan['id']}/install"
            async with services.scheduler.lease("primary"):
                accepted = await client.post(install_url)
                assert accepted.status_code == 202, accepted.text
                jobs = accepted.json()
                assert [job["kind"] for job in jobs] == ["workflow_install"]
                assert settings.comfy_executable is None
                with SessionLocal() as session:
                    offer = session.scalar(select(WorkflowInstallOffer))
                    assert offer is not None and offer.source_plan_id == plan["id"]
                    offer_id, draft_id = offer.id, offer.workflow_revision_id
                    draft = session.get(WorkflowRevision, draft_id)
                    assert draft is not None and not draft.trusted and draft.api_graph_json == {}
                    assert list(session.scalars(select(WorkflowActivation))) == []
            pending = services.downloads._offer_tasks.get(offer_id)
            assert pending is not None
            await asyncio.wait_for(asyncio.shield(pending), timeout=1200)
            progress = await client.get(f"/api/workflow-install-offers/{plan['id']}/progress")
            assert progress.status_code == 200, progress.text
            assert progress.json()["phase"] == "completed", progress.text
            revision_id = progress.json()["workflow_revision_id"]
            assert revision_id != draft_id
            verified = await asyncio.to_thread(runtimes.verify_status, "comfyui")
            assert verified.state == "ready" and verified.managed
            assert verified.release == approved.release
            assert runtime_config_path(settings.data_dir).is_file()
            assert settings.comfy_executable is not None and settings.comfy_executable.is_file()
            assert settings.comfy_directory is not None
            assert not any(
                item.name == "media" and item.running for item in services.processes.statuses()
            )
            with SessionLocal() as session:
                revision = session.get(WorkflowRevision, revision_id)
                assert revision is not None and revision.trusted and revision.ui_graph_json == graph
                assert revision.api_graph_json["1"]["inputs"]["width"] == 64
                assert revision.api_graph_json["1"]["inputs"]["height"] == 96
                activation = session.scalar(
                    select(WorkflowActivation).where(
                        WorkflowActivation.workflow_revision_id == revision_id
                    )
                )
                assert activation is not None and activation.state == "ready"
                job = session.get(Job, jobs[0]["id"])
                assert job is not None and job.status == "complete"
            repeated = await client.post(install_url)
            assert repeated.status_code == 202, repeated.text
            assert [job["id"] for job in repeated.json()] == [job["id"] for job in jobs]
            profile = await client.post(
                "/api/profiles",
                json={"name": "Constructed source CPU image", "role": "image", "engine": "comfyui"},
            )
            assert profile.status_code == 201, profile.text
            chat = (
                await client.post("/api/chats", json={"title": "Constructed source render"})
            ).json()
            selected = await client.patch(
                f"/api/chats/{chat['id']}", json={"active_image_profile_id": profile.json()["id"]}
            )
            assert selected.status_code == 200, selected.text
            submitted = await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={
                    "text": "Create a plain grey test image",
                    "mode": "image",
                    "workflow_revision_id": revision_id,
                },
            )
            assert submitted.status_code == 202, submitted.text
            run = submitted.json()["run"]
            assert run["workflow_revision_id"] == revision_id
            deadline = asyncio.get_running_loop().time() + 120
            while asyncio.get_running_loop().time() < deadline:
                current = (await client.get(f"/api/runs/{run['id']}")).json()
                if current["status"] in {"complete", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("The constructed source render did not terminate")
            assert current["status"] == "complete", current
            outputs = current["provenance_json"]["outputs"]
            assert len(outputs) == 1
            content = await client.get(f"/api/artifacts/{outputs[0]['artifact_id']}/content")
            assert content.status_code == 200
            with Image.open(io.BytesIO(content.content)) as rendered:
                assert rendered.size == (64, 96) and rendered.mode == "RGB"
                assert rendered.getpixel((32, 48)) == (102, 102, 102)
            assert requests == []
            result = {
                "source_plan_id": plan["id"],
                "approved": asdict(approved),
                "verified": verified.model_dump(mode="json"),
                "data_dir": str(settings.data_dir),
                "executable": str(settings.comfy_executable),
                "directory": str(settings.comfy_directory),
                "workflow_revision_id": revision_id,
                "rendered_size": [64, 96],
                "rendered_pixel": [102, 102, 102],
                "archive_http_requests": requests,
            }
            (settings.data_dir.parent / "source-runtime-installed.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        finally:
            await services.processes.stop("media")
