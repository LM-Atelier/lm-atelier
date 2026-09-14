"""Approve an extension source through the API and render it with the real worker."""

from __future__ import annotations

import asyncio
import hashlib
import io
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select
from test_comfy_registry_closure_driver import _project, _Sources
from test_comfy_registry_downloads import archive_bytes
from test_comfy_registry_wheel_environments import _wheel_content, _wheel_files
from test_workflow_package_execution_plan import _inputs
from test_workflow_package_extension_preflight import _configure_clients
from test_workflow_package_preparation import _Registry
from test_workflow_review_live_comfy import settings as settings
from test_workflow_source_live_comfy import _source_graph
from workflow_asset_transfer_fixture import configure_asset_transfer

from local_lm import models, workflow_source_runtime
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_interpreter import probe_comfy_registry_runtime_target
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.db import SessionLocal
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_source_extensions import (
    ExtensionPreparationServices,
    prepare_workflow_source_extensions,
)

_CODE = b"""import torch

class ExampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 64, "min": 1}),
            "height": ("INT", {"default": 96, "min": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1}),
            "color": ("INT", {"default": 0}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image"
    CATEGORY = "constructed"
    def image(self, width, height, batch_size, color):
        return (torch.full((batch_size, height, width, 3), 0.6),)

NODE_CLASS_MAPPINGS = {"ExampleNode": ExampleNode}
"""


async def _settle(app: FastAPI, offer_id: str) -> None:
    pending = app.state.services.downloads._offer_tasks.get(offer_id)
    if pending is not None:
        await asyncio.wait_for(asyncio.shield(pending), timeout=300)


async def _render(client: AsyncClient, revision_id: str, *, expected_intensity: int = 153) -> None:
    profile = await client.post(
        "/api/profiles",
        json={"name": "Constructed extension image", "role": "image", "engine": "comfyui"},
    )
    assert profile.status_code == 201, profile.text
    chat = (await client.post("/api/chats", json={"title": "Constructed extension render"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}", json={"active_image_profile_id": profile.json()["id"]}
    )
    assert selected.status_code == 200, selected.text
    submitted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a plain grey image",
            "mode": "image",
            "workflow_revision_id": revision_id,
        },
    )
    assert submitted.status_code == 202, submitted.text
    run_id = submitted.json()["run"]["id"]
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        current = (await client.get(f"/api/runs/{run_id}")).json()
        if current["status"] in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Constructed extension render did not terminate")
    assert current["status"] == "complete", current
    outputs = current["provenance_json"]["outputs"]
    assert len(outputs) == 1
    image = await client.get(f"/api/artifacts/{outputs[0]['artifact_id']}/content")
    assert image.status_code == 200
    with Image.open(io.BytesIO(image.content)) as rendered:
        assert rendered.size == (64, 96) and rendered.mode == "RGB"
        assert rendered.getpixel((32, 48)) == (expected_intensity,) * 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("commit", "dependencies", "asset_kind"),
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, True, "lora"),
        (True, True, "lora"),
        (False, True, "background_removal"),
    ],
    ids=[
        "release",
        "review-commit",
        "release-wheel",
        "review-commit-wheel",
        "release-wheel-assets",
        "review-commit-wheel-assets",
        "release-wheel-background-removal",
    ],
)
async def test_prepared_extension_runs_only_after_its_accepted_trust_decision(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit: bool,
    dependencies: bool,
    asset_kind: Literal["lora", "background_removal"] | None,
) -> None:
    services = app.state.services
    replace_worker = services.processes._replace

    async def cpu_replace(name: str, command: list[str], *args: Any, **kwargs: Any) -> None:
        await replace_worker(
            name, [*command, "--cpu"] if name == "media" else command, *args, **kwargs
        )

    monkeypatch.setattr(services.processes, "_replace", cpu_replace)
    async with services.scheduler.lease("primary"):
        await services.processes.start_media()
    inventory = await services.processes.comfy_node_inventory()
    assets = asset_kind is not None
    node_type = (
        "ExampleBackgroundImage"
        if asset_kind == "background_removal"
        else "ExampleLoraImage"
        if asset_kind == "lora"
        else "ExampleNode"
    )
    assert node_type not in inventory
    if asset_kind == "background_removal":
        assert "LoadBackgroundRemovalModel" in inventory
    context = PreparationContext.from_settings(services.settings)
    inputs = _inputs(commit=commit)
    # Each fresh database gets a separate package path in the shared disposable runtime.
    package_id = "example-pack-" + uuid4().hex[:12]
    inputs["selected"] = replace(inputs["selected"], package_id=package_id, node_types=(node_type,))
    inputs["requirement"] = replace(
        inputs["requirement"], package_id=package_id, node_types=(node_type,)
    )
    inputs["python_executable"] = context.python_executable
    inputs["interpreter_probe"] = probe_comfy_registry_runtime_target
    prefix = "package-root/" if commit else ""
    entries = {prefix + "__init__.py": _CODE}
    wheel_content = b""
    wheel_metadata = b""
    if dependencies:
        wheel_content = _wheel_content()
        wheel_metadata = _wheel_files(wheel_content)["alpha-1.0.dist-info/METADATA"]
        project, documents = _project("alpha", [("1.0", [])])
        records = project["files"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0].update(
            hashes={"sha256": hashlib.sha256(wheel_content).hexdigest()},
            size=len(wheel_content),
            **{"core-metadata": {"sha256": hashlib.sha256(wheel_metadata).hexdigest()}},
        )
        documents["alpha-1.0-py3-none-any.whl"] = wheel_metadata
        sources = _Sources({"alpha": project}, documents)
        inputs["project_client"] = SimpleNamespace(fetch=sources.fetch_projects)
        inputs["metadata_client"] = SimpleNamespace(fetch=sources.fetch_metadata)
        entries[prefix + "__init__.py"] = _CODE.replace(
            b"import torch", b"import torch\nimport alpha"
        ).replace(b"0.6)", b"0.6 * alpha.VALUE)")
        if commit:
            entries[prefix + "requirements.txt"] = b"alpha==1.0\n"
        else:
            inputs["selected"] = replace(inputs["selected"], pip_dependencies=("alpha==1.0",))
    asset = (
        configure_asset_transfer(app, monkeypatch, tmp_path, asset_kind=asset_kind)
        if asset_kind is not None
        else None
    )
    if asset is not None:
        asset_code = (
            entries[prefix + "__init__.py"]
            .replace(b"ExampleNode", node_type.encode())
            .replace(
                b"import torch",
                b"import torch\nimport folder_paths\nfrom safetensors.torch import load_file",
            )
        )
        if asset_kind == "background_removal":
            entries[prefix + "__init__.py"] = asset_code.replace(
                b"0.6 * alpha.VALUE",
                (
                    "0.6 * alpha.VALUE * load_file("
                    f'folder_paths.get_full_path("{asset.runtime_folder}", "{asset.filename}"))'
                    f'["{asset.tensor_name}"].item()'
                ).encode(),
            )
        else:
            input_name = "lora_name"
            entries[prefix + "__init__.py"] = (
                asset_code.replace(
                    b'"color":',
                    (
                        f'"{input_name}": '
                        f'(folder_paths.get_filename_list("{asset.runtime_folder}"),),\n'
                        '            "color":'
                    ).encode(),
                )
                .replace(
                    b"batch_size, color):",
                    f"batch_size, {input_name}, color):".encode(),
                )
                .replace(
                    b"0.6 * alpha.VALUE",
                    (
                        "0.6 * alpha.VALUE * load_file("
                        f'folder_paths.get_full_path("{asset.runtime_folder}", {input_name}))'
                        f'["{asset.tensor_name}"].item()'
                    ).encode(),
                )
            )
    inputs["registry_client"] = _Registry(inputs["selected"])
    inputs["state"]["content"] = archive_bytes(entries)
    graph = _source_graph()
    graph["nodes"][0].update(
        type=node_type,
        properties={
            "cnr_id": inputs["selected"].package_id,
            "ver": inputs["selected"].declared_version,
        },
    )
    if asset is not None and asset_kind == "lora":
        graph["nodes"][0]["widgets_values"].insert(3, asset.filename)
    elif asset is not None:
        graph["nodes"].append(
            {
                "id": 99,
                "type": "LoadBackgroundRemovalModel",
                "mode": 4,
                "inputs": [],
                "outputs": [],
                "widgets_values": [asset.filename],
                "properties": {"cnr_id": "comfy-core", "ver": "0.28.0"},
            }
        )
    payload = {
        "name": "Constructed extension source",
        "operation": "text_to_image",
        "ui_graph": graph,
        "dependencies": {"version": 1, "slots": []},
        "selections": [asset.selection()] if asset is not None else [],
    }
    requests: list[httpx.Request] = []
    wheel_requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=inputs["state"]["content"])

    def serve_wheel(request: httpx.Request) -> httpx.Response:
        assert dependencies and request.url.host == "files.pythonhosted.org"
        wheel_requests.append(request)
        return httpx.Response(
            200,
            content=wheel_metadata if request.url.path.endswith(".metadata") else wheel_content,
        )

    async with AsyncExitStack() as cleanup:
        _configure_clients(inputs, monkeypatch)
        cleanup.push_async_callback(inputs["unused_archive"].close)
        archive = ComfyRegistryArchiveDownloader(transport=httpx.MockTransport(serve))
        cleanup.push_async_callback(archive.close)
        wheels = ComfyRegistryWheelDownloader(transport=httpx.MockTransport(serve_wheel))
        cleanup.push_async_callback(wheels.close)

        async def prepare(*args: Any, **kwargs: Any) -> Any:
            return await prepare_workflow_source_extensions(
                *args,
                **kwargs,
                services=ExtensionPreparationServices(
                    probe_comfy_registry_runtime_target,
                    inputs["registry_client"],
                    inputs["project_client"],
                    inputs["metadata_client"],
                    archive,
                    wheels,
                ),
            )

        monkeypatch.setattr(workflow_source_runtime, "prepare_workflow_source_extensions", prepare)
        preview = await client.post("/api/workflows/packages/install-plans", json=payload)
        assert preview.status_code == 201, preview.text
        plan = preview.json()
        assert plan["can_accept"] and plan["blockers"] == [], plan
        assert plan["missing_node_types"] == [node_type]
        assert plan["total_download_bytes"] == (
            len(inputs["state"]["content"])
            + len(wheel_content)
            + (len(asset.content) if asset is not None else 0)
        )
        install_url = f"/api/workflow-install-offers/{plan['id']}/install"
        async with services.scheduler.lease("primary"):
            accepted = await client.post(install_url)
            assert accepted.status_code == 202, accepted.text
            jobs = accepted.json()
            assert sorted(job["kind"] for job in jobs) == (
                ["download", "workflow_install"] if assets else ["workflow_install"]
            )
            completion_id = next(job["id"] for job in jobs if job["kind"] == "workflow_install")
            with SessionLocal() as session:
                offer = session.scalar(
                    select(models.WorkflowInstallOffer).where(
                        models.WorkflowInstallOffer.source_plan_id == plan["id"]
                    )
                )
                assert offer is not None
                offer_id = offer.id
                draft = session.get(models.WorkflowRevision, offer.workflow_revision_id)
                assert draft is not None and not draft.trusted and draft.api_graph_json == {}
                assert list(session.scalars(select(models.ComfyRegistryInstall))) == []
                assert list(session.scalars(select(models.WorkflowActivation))) == []
        if asset is not None:
            for download in list(services.downloads._tasks.values()):
                await asyncio.wait_for(asyncio.shield(download), timeout=120)
            asset.assert_transferred()
            with SessionLocal() as session:
                installed_asset = session.scalar(select(models.ModelAssetInstall))
                assert installed_asset is not None
                assert (
                    Path(installed_asset.local_path) / asset.filename
                ).read_bytes() == asset.content
        await _settle(app, offer_id)
        if commit:
            progress = await client.get(f"/api/workflow-install-offers/{offer_id}/progress")
            assert progress.status_code == 200 and progress.json()["phase"] == "paused", (
                progress.text
            )
            with SessionLocal() as session:
                extension = session.scalar(select(models.ComfyRegistryInstall))
                assert extension is not None and not extension.trusted and not extension.active
                extension_id = extension.id
                assert list(session.scalars(select(models.WorkflowActivation))) == []
            assert node_type not in await services.processes.comfy_node_inventory()
            reviewed = await client.post(
                f"/api/workflows/packages/installs/{extension_id}/review", json={"trusted": True}
            )
            assert reviewed.status_code == 200, reviewed.text
            await _settle(app, offer_id)
    assert len(requests) == 1
    assert len(inputs["clients"]) == 1 and inputs["clients"][0]._client.is_closed
    assert len(wheel_requests) == 2 * int(dependencies)
    assert sum(request.url.path.endswith(".whl") for request in wheel_requests) == int(dependencies)
    progress = await client.get(f"/api/workflow-install-offers/{offer_id}/progress")
    assert progress.status_code == 200 and progress.json()["phase"] == "completed", progress.text
    with SessionLocal() as session:
        extension = session.scalar(select(models.ComfyRegistryInstall))
        assert extension is not None and extension.trusted and extension.active
        assert extension.review_json["activation_batch_v1"]["state"] == "complete"
        job = session.get(models.Job, completion_id)
        assert job is not None and job.status == "complete" and job.attempt == (2 if commit else 1)
    repeated = await client.post(install_url)
    assert repeated.status_code == 202, repeated.text
    assert {job["id"] for job in repeated.json()} == {job["id"] for job in jobs}
    await _render(
        client, progress.json()["workflow_revision_id"], expected_intensity=76 if assets else 153
    )
