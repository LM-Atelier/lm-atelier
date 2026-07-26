from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import zipfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session

import local_lm.api as api_module
from local_lm import __version__
from local_lm.adapters.base import ChatEvent, ChatRequest, MediaEvent, MediaRequest
from local_lm.adapters.mock import MockChatAdapter, MockMediaAdapter
from local_lm.auxiliary_assets import checkpoint_lora_extension
from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.hardware import hardware_capability_class
from local_lm.incognito import INCOGNITO_HEADER
from local_lm.main import create_app
from local_lm.models import (
    Artifact,
    Chat,
    GenerationPreset,
    Job,
    Message,
    MessagePart,
    ModelAssetInstall,
    ModelCapabilityEvidence,
    ModelInstall,
    ModelProfile,
    ModelSource,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    TurnCreationClaim,
    WorkflowDefinition,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import (
    DownloadRequest,
    EngineCapabilities,
    RuntimeStatus,
    SettingField,
    TurnRequest,
)

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def wait_for_assistant(client: AsyncClient, chat_id: str, expected_type: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/chats/{chat_id}")
        assert response.status_code == 200
        chat = response.json()
        assistant = [message for message in chat["messages"] if message["role"] == "assistant"][-1]
        if assistant["status"] in {"complete", "failed", "cancelled"}:
            assert assistant["status"] == "complete", assistant
            assert any(part["type"] == expected_type for part in assistant["parts"])
            return assistant
        await asyncio.sleep(0.03)
    raise AssertionError("assistant run did not complete")


async def wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            assert run["started_at"] is not None
            assert run["completed_at"] is not None
            assert isinstance(run["duration_ms"], int)
            assert run["duration_ms"] >= 0
            assert run["provenance_json"]["timings"]["duration_ms"] == run["duration_ms"]
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("run did not complete")


def extend_capability_role(
    capabilities: EngineCapabilities,
    role: str,
    *fields: SettingField,
) -> EngineCapabilities:
    by_role = {
        mapped_role: list(mapped_fields)
        for mapped_role, mapped_fields in capabilities.settings_by_role.items()
    }
    by_role[role] = [*by_role.get(role, []), *fields]
    return capabilities.model_copy(
        update={
            "settings": [*capabilities.settings, *fields],
            "settings_by_role": by_role,
        }
    )


def create_managed_model(
    *,
    model_id: str,
    path: Path,
    files: list[str],
    role: str = "chat",
    engine: str = "mock",
) -> None:
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id=model_id,
                name=model_id,
                role=role,
                engine=engine,
                local_path=str(path),
                size_bytes=sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
                if path.is_dir()
                else path.stat().st_size,
                compatibility="likely",
                manifest_json={"files": files},
                active=True,
            )
        )
        session.commit()


def project_manifest(archive_bytes: bytes) -> dict:  # type: ignore[type-arg]
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return json.loads(archive.read("manifest.json"))


def rewrite_project_archive(
    archive_bytes: bytes,
    manifest: dict,  # type: ignore[type-arg]
    *,
    extras: dict[str, bytes] | None = None,
) -> bytes:
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(archive_bytes)) as source,
        zipfile.ZipFile(output, "w") as destination,
    ):
        destination.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for info in source.infolist():
            if info.is_dir() or info.filename == "manifest.json":
                continue
            destination.writestr(info.filename, source.read(info), info.compress_type)
        for name, payload in (extras or {}).items():
            destination.writestr(name, payload)
    return output.getvalue()


async def test_health_probes_the_database(client: AsyncClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    healthy = await client.get("/api/health")
    assert healthy.status_code == 200
    assert healthy.json()["database"] is True
    assert healthy.json()["version"] == __version__

    def fail_probe(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        from sqlalchemy.exc import OperationalError

        raise OperationalError("SELECT 1", {}, Exception("database unavailable"))

    monkeypatch.setattr("sqlalchemy.orm.Session.execute", fail_probe)
    degraded = await client.get("/api/health")
    assert degraded.status_code == 200
    assert degraded.json()["status"] == "degraded"
    assert degraded.json()["database"] is False


async def test_fast_readiness_probe_reports_version(client: AsyncClient) -> None:
    response = await client.get("/api/ready")
    assert response.status_code == 200
    assert response.json() == {"version": __version__}


async def test_model_readiness_requires_matching_capability_evidence(
    client: AsyncClient,
    settings: Settings,
) -> None:
    current_hardware_class = hardware_capability_class(settings)
    with SessionLocal() as session:
        verified = ModelInstall(
            id="model_verified",
            name="Verified",
            role="chat",
            engine="llama.cpp",
            local_path="C:/models/verified",
            manifest_json={
                "files": ["verified.gguf"],
                "expected_sha256": {"verified.gguf": "b" * 64},
            },
            active=True,
        )
        unverified = ModelInstall(
            id="model_unverified",
            name="Unverified",
            role="chat",
            engine="llama.cpp",
            local_path="C:/models/unverified",
            manifest_json={"files": ["unverified.gguf"]},
            active=True,
        )
        session.add_all([verified, unverified])
        session.flush()
        session.add(
            ModelCapabilityEvidence(
                model_install_id=verified.id,
                evidence_key="a" * 64,
                result="ready",
                component_hashes_json={"verified.gguf": "b" * 64},
                runtime_build="llama-test",
                adapter_contract_version=1,
                launch_contract_version="worker-launch-v1",
                workflow_contract_version=None,
                hardware_class=current_hardware_class,
                probe_version="activation-probe-v2",
            )
        )
        session.commit()

    response = await client.get("/api/models")
    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()}
    assert models["model_verified"]["readiness"] == "ready"
    assert models["model_verified"]["capability_evidence"]["runtime_build"] == "llama-test"
    assert models["model_unverified"]["readiness"] == "unverified"
    assert models["model_unverified"]["capability_evidence"] is None


async def test_about_reports_version_and_local_support_paths(
    client: AsyncClient, settings: Settings
) -> None:
    response = await client.get("/api/about")

    assert response.status_code == 200
    assert response.json() == {
        "version": __version__,
        "data_directory": str(settings.data_dir.resolve()),
        "log_directory": str(settings.log_dir.resolve()),
    }


async def test_project_and_chat_management_contract(client: AsyncClient) -> None:
    project = (await client.post("/api/projects", json={"name": "Research Lab"})).json()
    chat = (
        await client.post("/api/chats", json={"title": "Model notes", "project_id": project["id"]})
    ).json()

    project_search = await client.get("/api/projects", params={"query": "research"})
    chat_search = await client.get("/api/chats", params={"query": "notes"})
    assert [item["id"] for item in project_search.json()] == [project["id"]]
    assert [item["id"] for item in chat_search.json()] == [chat["id"]]

    archived = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"title": "Archived notes", "archived": True, "project_id": None},
    )
    assert archived.status_code == 200
    assert archived.json()["project_id"] is None
    assert (await client.get("/api/chats")).json() == []
    archived_list = await client.get("/api/chats", params={"include_archived": True})
    assert [item["id"] for item in archived_list.json()] == [chat["id"]]

    restored = await client.patch(f"/api/chats/{chat['id']}", json={"archived": False})
    assert restored.status_code == 200
    assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 204
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["project_id"] is None
    assert (await client.delete(f"/api/chats/{chat['id']}")).status_code == 204
    assert (await client.get(f"/api/chats/{chat['id']}")).status_code == 404


async def test_project_chat_text_and_inline_image_flow(client: AsyncClient) -> None:
    project_response = await client.post("/api/projects", json={"name": "Demo"})
    assert project_response.status_code == 201
    project = project_response.json()

    chat_response = await client.post(
        "/api/chats", json={"title": "New chat", "project_id": project["id"]}
    )
    assert chat_response.status_code == 201
    chat = chat_response.json()

    text_response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Explain local inference", "mode": "auto", "idempotency_key": "text-1"},
    )
    assert text_response.status_code == 202
    assert text_response.json()["run"]["operation"] == "text"
    assistant = await wait_for_assistant(client, chat["id"], "text")
    assert "Mock local response" in assistant["parts"][0]["text"]
    text_metadata = next(
        part for part in assistant["parts"] if part["type"] == "generation_metadata"
    )["metadata_json"]
    assert text_metadata["provenance"]["output"]["kind"] == "text"
    assert len(text_metadata["provenance"]["output"]["sha256"]) == 64
    assert "resolved_settings" in text_metadata["provenance"]

    image_response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a purple observatory",
            "mode": "auto",
            "idempotency_key": "image-1",
        },
    )
    assert image_response.status_code == 202
    assert image_response.json()["run"]["operation"] == "text_to_image"
    image_message = await wait_for_assistant(client, chat["id"], "image")
    image_part = next(part for part in image_message["parts"] if part["type"] == "image")
    image_metadata = next(
        part for part in image_message["parts"] if part["type"] == "generation_metadata"
    )["metadata_json"]
    image_output = image_metadata["provenance"]["outputs"][0]
    assert image_output["artifact_id"] == image_part["artifact_id"]
    assert image_output["sha256"] == image_part["artifact"]["sha256"]
    assert image_part["artifact"]["kind"] == "image"
    assert image_part["artifact"]["metadata_json"].get("temporary_preview") is None
    content = await client.get(f"/api/artifacts/{image_part['artifact_id']}/content")
    assert content.status_code == 200
    assert content.headers["content-type"].startswith("image/svg+xml")
    partial = await client.get(
        f"/api/artifacts/{image_part['artifact_id']}/content", headers={"range": "bytes=0-15"}
    )
    assert partial.status_code == 206
    assert partial.headers["content-range"].startswith("bytes 0-15/")
    assert len(partial.content) == 16

    followup = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Animate that image gently", "mode": "auto"},
    )
    assert followup.status_code == 202
    assert followup.json()["run"]["operation"] == "image_to_video"
    assert followup.json()["run"]["provenance_json"]["input_artifact_ids"] == [
        image_part["artifact_id"]
    ]
    await wait_for_assistant(client, chat["id"], "video")


async def test_image_turn_resolves_verified_lora_stack_and_provenance(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.operation == "text_to_image")
        )
        assert definition and definition.current_revision_id
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        assert revision
        graph = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "mock.safetensors"},
            },
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        }
        extension = checkpoint_lora_extension(graph)
        assert extension
        revision.api_graph_json = graph
        revision.input_schema_json = {
            "type": "object",
            "properties": {
                "loras": {
                    "type": "array",
                    "default": [],
                    "maxItems": 8,
                }
            },
        }
        revision.dependencies_json = {"extensions": {"lora": extension}}
        lora = ModelAssetInstall(
            name="Ink",
            kind="lora",
            family="sdxl",
            local_path="C:/managed/ink",
            size_bytes=1024,
            manifest_json={
                "sha256": "a" * 64,
                "comfy_name": "ink.safetensors",
                "metadata": {"trigger_words": ["ink"]},
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(lora)
        session.commit()
        lora_id = lora.id

    listed = await client.get("/api/model-assets", params={"kind": "lora"})
    assert listed.status_code == 200
    assert [asset["id"] for asset in listed.json()] == [lora_id]
    chat = (await client.post("/api/chats", json={"title": "LoRA run"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of an ink workshop",
            "mode": "image",
            "settings": {
                "loras": [
                    {
                        "asset_id": lora_id,
                        "model_strength": 0.8,
                        "clip_strength": 0.65,
                        "enabled": True,
                    }
                ]
            },
        },
    )
    assert response.status_code == 202
    run = await wait_for_run(client, response.json()["run"]["id"])
    assert run["status"] == "complete"
    auxiliary = run["provenance_json"]["auxiliary_assets"]
    assert auxiliary["graph_transform_version"] == "lora-graph-v1"
    assert len(auxiliary["effective_graph_sha256"]) == 64
    assert auxiliary["lora_stack"] == [
        {
            "asset_id": lora_id,
            "position": 0,
            "name": "Ink",
            "family": "sdxl",
            "sha256": "a" * 64,
            "comfy_name": "ink.safetensors",
            "trigger_words": ["ink"],
            "model_strength": 0.8,
            "clip_strength": 0.65,
            "enabled": True,
        }
    ]


async def test_verified_model_asset_can_be_toggled_and_deleted_safely(
    client: AsyncClient,
    settings: Settings,
) -> None:
    asset_root = settings.model_dir / "asset-lifecycle"
    asset_root.mkdir(parents=True)
    (asset_root / "ink.safetensors").write_bytes(b"verified")
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name="Lifecycle LoRA",
            kind="lora",
            family="sdxl",
            local_path=str(asset_root),
            size_bytes=8,
            manifest_json={
                "sha256": "d" * 64,
                "comfy_name": "ink.safetensors",
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        asset_id = asset.id

    disabled = await client.patch(f"/api/model-assets/{asset_id}", json={"active": False})
    assert disabled.status_code == 200
    assert disabled.json()["active"] is False
    enabled = await client.patch(f"/api/model-assets/{asset_id}", json={"active": True})
    assert enabled.status_code == 200
    assert enabled.json()["active"] is True

    deleted = await client.delete(f"/api/model-assets/{asset_id}")
    assert deleted.status_code == 204
    assert not asset_root.exists()
    with SessionLocal() as session:
        assert session.get(ModelAssetInstall, asset_id) is None


async def test_inline_video_and_project_export(client: AsyncClient) -> None:
    project = (await client.post("/api/projects", json={"name": "Film lab"})).json()
    chat = (
        await client.post("/api/chats", json={"title": "Storyboard", "project_id": project["id"]})
    ).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a short video of dawn", "mode": "video"},
    )
    assert response.status_code == 202
    message = await wait_for_assistant(client, chat["id"], "video")
    video_part = next(part for part in message["parts"] if part["type"] == "video")
    assert video_part["artifact_id"]
    metadata = next(part for part in message["parts"] if part["type"] == "generation_metadata")[
        "metadata_json"
    ]
    output = metadata["provenance"]["outputs"][0]
    assert output["artifact_id"] == video_part["artifact_id"]
    assert output["sha256"] == video_part["artifact"]["sha256"]
    assert output["kind"] == video_part["artifact"]["kind"]

    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201
    archive = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert "manifest.json" in bundle.namelist()
        assert any(name.startswith("artifacts/") for name in bundle.namelist())
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["version"] == 6
        assert set(manifest["dependencies"]) == {"profiles", "presets", "workflows"}

    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "film-lab.lm-atelier.zip",
                archive.content,
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    imported_chats = await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    assert [item["title"] for item in imported_chats.json()] == ["Storyboard"]
    imported_detail = (await client.get(f"/api/chats/{imported_chats.json()[0]['id']}")).json()
    imported_video = next(
        part
        for message in imported_detail["messages"]
        for part in message["parts"]
        if part["type"] == "video"
    )
    assert imported_video["artifact_id"] == video_part["artifact_id"]


async def test_project_archive_uses_immutable_auxiliary_requirements_without_weights(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "LoRA archive"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Portable LoRA", "project_id": project["id"]},
        )
    ).json()
    digest = "c" * 64
    with SessionLocal() as session:
        source = ModelSource(
            provider="huggingface",
            remote_id="atelier/ink-lora",
            revision="immutable-revision",
        )
        session.add(source)
        session.flush()
        asset = ModelAssetInstall(
            source_id=source.id,
            name="Atelier Ink",
            kind="lora",
            family="sdxl",
            local_path=r"C:\private-models\ink.safetensors",
            size_bytes=12_345,
            manifest_json={
                "sha256": digest,
                "metadata": {
                    "network_type": "LoRA",
                    "rank": 16,
                    "trigger_words": ["ink wash"],
                },
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.flush()
        chat_row = session.get(Chat, chat["id"])
        assert chat_row
        chat_row.generation_settings_json = {
            "image": {
                "loras": [
                    {
                        "asset_id": asset.id,
                        "model_strength": 0.8,
                        "clip_strength": 0.7,
                        "enabled": True,
                    }
                ]
            }
        }
        asset_id = asset.id
        session.commit()

    exported = await client.post(
        f"/api/projects/{project['id']}/export",
        params={"include_media": False},
    )
    archive_response = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        assert all(not name.endswith(".safetensors") for name in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    reference = f"auxiliary:lora:sha256:{digest}"
    assert manifest["version"] == 6
    assert manifest["auxiliary_requirements"] == [
        {
            "id": reference,
            "kind": "lora",
            "name": "Atelier Ink",
            "family": "sdxl",
            "sha256": digest,
            "size_bytes": 12_345,
            "metadata": {
                "network_type": "LoRA",
                "rank": 16,
                "trigger_words": ["ink wash"],
            },
            "source": {
                "provider": "huggingface",
                "remote_id": "atelier/ink-lora",
                "revision": "immutable-revision",
            },
        }
    ]
    assert (
        manifest["chats"][0]["generation_settings_json"]["image"]["loras"][0]["asset_id"]
        == reference
    )
    assert asset_id not in json.dumps(manifest)
    assert "private-models" not in json.dumps(manifest)

    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "lora-project.lm-atelier.zip",
                archive_response.content,
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    imported_chat = (
        await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    ).json()[0]
    assert imported_chat["generation_settings_json"]["image"]["loras"][0]["asset_id"] == asset_id


async def test_metadata_only_project_export_import_marks_missing_media(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Portable"})).json()
    chat = (
        await client.post("/api/chats", json={"title": "Images", "project_id": project["id"]})
    ).json()
    await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a portable image", "mode": "image"},
    )
    await wait_for_assistant(client, chat["id"], "image")
    exported = await client.post(
        f"/api/projects/{project['id']}/export", params={"include_media": False}
    )
    archive = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert not any(name.startswith("artifacts/") for name in bundle.namelist())

    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("metadata.lm-atelier.zip", archive.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    imported_chat = (
        await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    ).json()[0]
    detail = (await client.get(f"/api/chats/{imported_chat['id']}")).json()
    image_part = next(
        part
        for message in detail["messages"]
        for part in message["parts"]
        if part["type"] == "image"
    )
    assert image_part["artifact_id"] is None
    assert image_part["metadata_json"]["missing_import_artifact_id"].startswith("sha256:")


async def test_project_import_rejects_unsafe_archive_paths(client: AsyncClient) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("../outside", b"unsafe")
        bundle.writestr("manifest.json", b"{}")
    response = await client.post(
        "/api/projects/import",
        files={"archive": ("unsafe.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 422
    assert "unsafe path" in response.json()["detail"]


async def test_project_import_remains_compatible_with_v1_and_v2(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Legacy source"})).json()
    await client.post(
        "/api/chats",
        json={"title": "Legacy chat", "project_id": project["id"]},
    )
    exported = (
        await client.post(
            f"/api/projects/{project['id']}/export",
            params={"include_media": False},
        )
    ).json()
    archive = await client.get(exported["url"])
    current = project_manifest(archive.content)

    for version in (1, 2):
        legacy = json.loads(json.dumps(current))
        legacy["version"] = version
        legacy.pop("dependencies")
        legacy["project"].pop("generation_settings_json")
        legacy["project"].pop("generation_preset_ids_json")
        for chat in legacy["chats"]:
            chat.pop("generation_settings_json")
            chat.pop("generation_preset_ids_json")
        response = await client.post(
            "/api/projects/import",
            files={
                "archive": (
                    f"legacy-v{version}.lm-atelier.zip",
                    rewrite_project_archive(archive.content, legacy),
                    "application/zip",
                )
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["generation_settings_json"] == {}
        imported_chats = (
            await client.get(
                "/api/chats",
                params={"project_id": response.json()["id"]},
            )
        ).json()
        assert imported_chats[0]["active_chat_profile_id"] == "__auto__"


async def test_project_v3_import_rejects_malformed_dependencies_and_object_abuse(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Hostile input"})).json()
    await client.post(
        "/api/chats",
        json={"title": "Validation target", "project_id": project["id"]},
    )
    exported = (
        await client.post(
            f"/api/projects/{project['id']}/export",
            params={"include_media": False},
        )
    ).json()
    archive = await client.get(exported["url"])
    baseline = project_manifest(archive.content)

    missing_profile = json.loads(json.dumps(baseline))
    missing_profile["chats"][0]["active_chat_profile_id"] = "profile_not_embedded"

    extra_dependency_field = json.loads(json.dumps(baseline))
    extra_dependency_field["dependencies"]["profiles"].append(
        {
            "source_id": "profile_extra",
            "name": "Unexpected field",
            "use_case": "",
            "role": "chat",
            "engine": "mock",
            "load_settings": {},
            "request_settings": {},
            "absolute_path": "C:\\secrets",
        }
    )

    duplicate_dependency = json.loads(json.dumps(baseline))
    profile_record = {
        "source_id": "profile_duplicate",
        "name": "Duplicate",
        "use_case": "",
        "role": "chat",
        "engine": "mock",
        "load_settings": {},
        "request_settings": {},
    }
    duplicate_dependency["dependencies"]["profiles"] = [
        profile_record,
        dict(profile_record),
    ]

    invalid_workflow_head = json.loads(json.dumps(baseline))
    invalid_workflow_head["dependencies"]["workflows"] = [
        {
            "source_id": "workflow_hostile",
            "name": "Hostile workflow",
            "operation": "text_to_image",
            "description": "",
            "current_revision_source_id": "wfrev_missing",
            "revisions": [
                {
                    "source_id": "wfrev_present",
                    "source_version": 1,
                    "engine": "mock",
                    "engine_version": None,
                    "ui_graph": {},
                    "api_graph": {},
                    "input_schema": {},
                    "dependencies": {},
                    "trusted": True,
                }
            ],
        }
    ]

    deeply_nested = json.loads(json.dumps(baseline))
    nested: dict = {}  # type: ignore[type-arg]
    cursor = nested
    for _ in range(70):
        child: dict = {}  # type: ignore[type-arg]
        cursor["nested"] = child
        cursor = child
    deeply_nested["dependencies"]["profiles"].append(
        {
            "source_id": "profile_deep",
            "name": "Deep object",
            "use_case": "",
            "role": "chat",
            "engine": "mock",
            "load_settings": nested,
            "request_settings": {},
        }
    )

    non_finite = json.loads(json.dumps(baseline))
    non_finite["project"]["description"] = float("nan")

    cases = [
        (missing_profile, None, "incompatible role"),
        (extra_dependency_field, None, "invalid portable dependencies"),
        (duplicate_dependency, None, "duplicate profile dependency ids"),
        (invalid_workflow_head, None, "invalid current workflow revision"),
        (deeply_nested, None, "nested too deeply"),
        (non_finite, None, "invalid numeric value"),
        (baseline, {"undeclared/payload.exe": b"MZ"}, "not declared"),
    ]
    project_count = len((await client.get("/api/projects")).json())
    for manifest, extras, expected in cases:
        response = await client.post(
            "/api/projects/import",
            files={
                "archive": (
                    "hostile.lm-atelier.zip",
                    rewrite_project_archive(archive.content, manifest, extras=extras),
                    "application/zip",
                )
            },
        )
        assert response.status_code == 422, response.text
        assert expected in response.json()["detail"]
        assert len((await client.get("/api/projects")).json()) == project_count


async def test_project_v3_round_trip_remaps_portable_dependencies_in_a_fresh_database(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    chat_profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Portable conversationalist",
                "use_case": "Concise research answers",
                "role": "chat",
                "engine": "mock",
                "load_settings": {"context_length": 4096},
                "request_settings": {"temperature": 0.35},
            },
        )
    ).json()
    image_profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Portable illustrator",
                "use_case": "Editorial illustrations",
                "role": "image",
                "engine": "mock",
                "request_settings": {"steps": 12},
            },
        )
    ).json()
    preset = (
        await client.post(
            "/api/presets",
            json={
                "name": "Portable project voice",
                "role": "chat",
                "settings": {"temperature": 0.2, "max_tokens": 333},
            },
        )
    ).json()
    workflow_response = await client.post(
        "/api/workflows",
        json={
            "name": "Portable illustration workflow",
            "operation": "text_to_image",
            "description": "Two immutable portable revisions",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "PortableV1"}},
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "default": 12}},
            },
            "dependencies": {"models": ["portable-model"]},
            "trusted": True,
        },
    )
    assert workflow_response.status_code == 201
    workflow = workflow_response.json()
    revision_one = workflow["revisions"][0]
    revision_two_response = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "api_graph": {"node": {"class_type": "PortableV2"}},
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "default": 16}},
            },
            "dependencies": {"models": ["portable-model-v2"]},
            "trusted": True,
        },
    )
    assert revision_two_response.status_code == 201
    revision_two = revision_two_response.json()

    project = (
        await client.post(
            "/api/projects",
            json={
                "name": "Fully portable project",
                "image_workflow_revision_id": revision_one["id"],
                "generation_preset_ids_json": {"chat": preset["id"]},
                "generation_settings_json": {"chat": {"temperature": 0.1}},
            },
        )
    ).json()
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Portable dependency chat",
                "project_id": project["id"],
                "generation_preset_ids_json": {"chat": preset["id"]},
                "generation_settings_json": {"chat": {"max_tokens": 222}},
            },
        )
    ).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={
            "active_chat_profile_id": chat_profile["id"],
            "active_image_profile_id": image_profile["id"],
        },
    )
    assert selected.status_code == 200
    text_turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Describe a portable archive", "mode": "text"},
    )
    assert text_turn.status_code == 202
    await wait_for_run(client, text_turn.json()["run"]["id"])
    image_turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image of a portable archive", "mode": "image"},
    )
    assert image_turn.status_code == 202
    await wait_for_run(client, image_turn.json()["run"]["id"])

    exported = (
        await client.post(
            f"/api/projects/{project['id']}/export",
            params={"include_media": False},
        )
    ).json()
    archive = await client.get(exported["url"])
    manifest = project_manifest(archive.content)
    assert manifest["version"] == 6
    assert {item["source_id"] for item in manifest["dependencies"]["profiles"]} == {
        chat_profile["id"],
        image_profile["id"],
    }
    assert [item["source_id"] for item in manifest["dependencies"]["presets"]] == [preset["id"]]
    portable_workflow = manifest["dependencies"]["workflows"][0]
    assert portable_workflow["source_id"] == workflow["id"]
    assert [item["source_version"] for item in portable_workflow["revisions"]] == [1, 2]
    assert portable_workflow["current_revision_source_id"] == revision_two["id"]
    with zipfile.ZipFile(io.BytesIO(archive.content)) as metadata_archive:
        assert not any(name.startswith("artifacts/") for name in metadata_archive.namelist())

    target_settings = Settings(
        data_dir=tmp_path / "portable-target",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    target_app = create_app(target_settings)
    async with target_app.router.lifespan_context(target_app):  # noqa: SIM117
        async with AsyncClient(
            transport=ASGITransport(app=target_app),
            base_url="http://testserver",
        ) as target:
            session_response = await target.post("/api/session")
            target.headers["x-local-lm-csrf"] = session_response.json()["csrf_token"]

            collision_profile = await target.post(
                "/api/profiles",
                json={
                    "name": chat_profile["name"],
                    "use_case": chat_profile["use_case"],
                    "role": "chat",
                    "engine": "mock",
                    "request_settings": {"temperature": 0.9},
                },
            )
            assert collision_profile.status_code == 201
            collision_preset = await target.post(
                "/api/presets",
                json={
                    "name": preset["name"],
                    "role": "chat",
                    "settings": {"temperature": 0.95},
                },
            )
            assert collision_preset.status_code == 201
            collision_workflow = await target.post(
                "/api/workflows",
                json={
                    "name": workflow["name"],
                    "operation": "text_to_image",
                    "description": workflow["description"],
                    "engine": "mock",
                    "api_graph": {"node": {"class_type": "Collision"}},
                },
            )
            assert collision_workflow.status_code == 201

            imported = await target.post(
                "/api/projects/import",
                files={
                    "archive": (
                        "portable-project.lm-atelier.zip",
                        archive.content,
                        "application/zip",
                    )
                },
            )
            assert imported.status_code == 201, imported.text
            imported_project = imported.json()
            imported_chats = (
                await target.get(
                    "/api/chats",
                    params={"project_id": imported_project["id"]},
                )
            ).json()
            imported_chat = imported_chats[0]

            assert imported_project["image_workflow_revision_id"] != revision_one["id"]
            assert imported_project["generation_preset_ids_json"]["chat"] != preset["id"]
            assert imported_project["generation_settings_json"]["chat"] == {
                "temperature": 0.1,
                "max_tokens": 333,
            }
            assert imported_chat["active_chat_profile_id"] != chat_profile["id"]
            assert imported_chat["active_image_profile_id"] != image_profile["id"]
            assert imported_chat["generation_settings_json"]["chat"] == {
                "temperature": 0.2,
                "max_tokens": 222,
            }
            assert (
                imported_chat["generation_preset_ids_json"]["chat"]
                == (imported_project["generation_preset_ids_json"]["chat"])
            )

            with SessionLocal() as target_session:
                chat_row = target_session.get(Chat, imported_chat["id"])
                assert chat_row
                imported_chat_profile = target_session.get(
                    ModelProfile, chat_row.active_chat_profile_id
                )
                imported_image_profile = target_session.get(
                    ModelProfile, chat_row.active_image_profile_id
                )
                assert imported_chat_profile
                assert imported_chat_profile.request_settings_json == {"temperature": 0.35}
                assert imported_chat_profile.model_install_id is None
                assert imported_image_profile
                assert imported_image_profile.request_settings_json == {"steps": 12}

                imported_preset = target_session.get(
                    GenerationPreset,
                    imported_project["generation_preset_ids_json"]["chat"],
                )
                assert imported_preset
                assert imported_preset.name == f"{preset['name']} (imported)"
                assert imported_preset.settings_json == {
                    "temperature": 0.2,
                    "max_tokens": 333,
                }

                pinned = target_session.get(
                    WorkflowRevision,
                    imported_project["image_workflow_revision_id"],
                )
                assert pinned
                assert pinned.version == 1
                imported_definition = target_session.get(WorkflowDefinition, pinned.workflow_id)
                assert imported_definition
                imported_revisions = list(
                    target_session.scalars(
                        select(WorkflowRevision)
                        .where(WorkflowRevision.workflow_id == imported_definition.id)
                        .order_by(WorkflowRevision.version)
                    ).all()
                )
                assert [item.version for item in imported_revisions] == [1, 2]
                assert all(item.trusted is False for item in imported_revisions)
                assert imported_definition.current_revision_id == imported_revisions[1].id
                imported_runs = list(
                    target_session.scalars(select(Run).where(Run.chat_id == chat_row.id)).all()
                )
                assert {run.profile_id for run in imported_runs if run.operation == "text"} == {
                    imported_chat_profile.id
                }
                image_run = next(run for run in imported_runs if run.operation == "text_to_image")
                assert image_run.profile_id == imported_image_profile.id
                assert image_run.workflow_revision_id == pinned.id

            repeated = await target.post(
                "/api/projects/import",
                files={
                    "archive": (
                        "portable-project-again.lm-atelier.zip",
                        archive.content,
                        "application/zip",
                    )
                },
            )
            assert repeated.status_code == 201, repeated.text
            assert (
                repeated.json()["image_workflow_revision_id"]
                == imported_project["image_workflow_revision_id"]
            )
            assert (
                repeated.json()["generation_preset_ids_json"]
                == (imported_project["generation_preset_ids_json"])
            )
            profiles = (await target.get("/api/profiles?role=chat")).json()
            assert len([item for item in profiles if item["name"] == chat_profile["name"]]) == 2
            workflows = (await target.get("/api/workflows")).json()
            assert len([item for item in workflows if item["name"] == workflow["name"]]) == 2


async def test_backup_create_verify_and_restore_marker(
    client: AsyncClient, settings: Settings
) -> None:
    created = await client.post("/api/backups")
    assert created.status_code == 201
    name = created.json()["name"]
    verified = await client.post(f"/api/backups/{name}/verify")
    assert verified.status_code == 200
    assert verified.json()["verified"] is True
    restore = await client.post(f"/api/backups/{name}/restore")
    assert restore.status_code == 200
    assert restore.json()["restore_pending"] is True

    await client.post(
        "/api/artifacts",
        files={"file": ("backup-image.png", b"backup-media", "image/png")},
    )
    media_backup = await client.post("/api/backups", params={"include_media": True})
    assert media_backup.status_code == 201
    assert media_backup.json()["media_included"] is True
    assert media_backup.json()["media_size_bytes"] > 0
    assert (settings.backup_dir / f"{media_backup.json()['name']}.media.zip").is_file()
    assert (await client.post(f"/api/backups/{media_backup.json()['name']}/verify")).json()[
        "verified"
    ] is True


async def test_diagnostic_bundle_is_redacted(client: AsyncClient) -> None:
    secret_prompt = "diagnostic-secret-prompt-that-must-not-leak"
    chat = (await client.post("/api/chats", json={"title": "Private"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": secret_prompt, "mode": "text"}
    )
    await wait_for_run(client, turn.json()["run"]["id"])

    created = await client.post("/api/diagnostics")
    assert created.status_code == 201
    archive = await client.get(created.json()["url"])
    assert secret_prompt.encode() not in archive.content
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        payload = bundle.read("diagnostics.json")
        assert secret_prompt.encode() not in payload
        assert b'"prompts_included": false' in payload
        assert b'"tokens_included": false' in payload
        assert b'"comfy_inactivity_seconds": 600' in payload


async def test_worker_management_reports_missing_local_binaries(client: AsyncClient) -> None:
    workers = await client.get("/api/workers")
    assert workers.status_code == 200
    assert {item["name"] for item in workers.json()} == {"chat", "media"}
    assert all(item["state"] == "stopped" for item in workers.json())
    assert all(item["active_jobs"] == 0 and item["queued_jobs"] == 0 for item in workers.json())
    assert all(item["current_memory_bytes"] is None for item in workers.json())
    assert all(item["failure_detail"] is None for item in workers.json())
    assert all(item["stderr_tail"] is None for item in workers.json())
    assert {item["log_path"] for item in workers.json()} == {
        "logs/chat-worker.log",
        "logs/media-worker.log",
    }
    media = await client.post("/api/workers/media/start")
    assert media.status_code == 422


async def test_runtime_status_exposes_pinned_external_setup(client: AsyncClient) -> None:
    response = await client.get("/api/runtimes")

    assert response.status_code == 200
    runtimes = {item["engine"]: item for item in response.json()}
    assert set(runtimes) == {"llama.cpp", "comfyui"}
    assert runtimes["llama.cpp"]["release"] == "b9637"
    assert runtimes["comfyui"]["release"] == "v0.28.0"
    assert runtimes["comfyui"]["distribution"] == "external-gpl-3.0"
    assert runtimes["comfyui"]["license"] == "GPL-3.0-only"
    assert runtimes["comfyui"]["state"] in {"missing", "unsupported"}
    if runtimes["comfyui"]["security_status"] == "blocked":
        assert runtimes["comfyui"]["state"] == "unsupported"
        assert runtimes["comfyui"]["supported"] is False
        assert runtimes["comfyui"]["security_message"]
    else:
        assert runtimes["comfyui"]["security_status"] == "checksum-pinned"


def test_downloads_share_the_generation_compute_scheduler(app: FastAPI) -> None:
    services = app.state.services
    assert services.downloads.scheduler is services.scheduler


async def test_worker_changes_are_rejected_while_generation_is_pending(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        session.add(
            Job(
                kind="chat",
                status="queued",
                phase="queued",
                payload_json={},
            )
        )
        session.commit()

    stopped = await client.post("/api/workers/chat/stop")
    assert stopped.status_code == 409
    assert "active or queued job" in stopped.json()["detail"]

    profiles = (await client.get("/api/profiles?role=chat")).json()
    loaded = await client.post(f"/api/workers/chat/load/{profiles[0]['id']}")
    assert loaded.status_code == 409


async def test_model_deletion_rechecks_pending_generation_inside_compute_lease(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
) -> None:
    model_path = settings.model_dir / "delete-race"
    model_path.mkdir(parents=True)
    (model_path / "model.safetensors").write_bytes(b"model")
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_delete_race",
            name="Delete race",
            role="image",
            engine="comfyui",
            local_path=str(model_path),
            manifest_json={"files": ["model.safetensors"]},
            active=True,
        )
        session.add(install)
        session.add(
            ModelProfile(
                id="profile_delete_race",
                name="Delete race",
                role="image",
                engine="comfyui",
                model_install_id=install.id,
            )
        )
        session.commit()

    async with app.state.services.scheduler.lease("primary"):
        deletion = asyncio.create_task(
            client.delete(
                "/api/models/model_delete_race",
                params={"delete_profiles": "true"},
            )
        )
        await asyncio.sleep(0.03)
        assert deletion.done() is False
        with SessionLocal() as session:
            session.add(
                Job(
                    id="job_delete_race",
                    kind="image",
                    status="queued",
                    phase="queued",
                )
            )
            session.commit()

    response = await asyncio.wait_for(deletion, timeout=2)
    assert response.status_code == 409
    assert "active or queued job" in response.json()["detail"]
    with SessionLocal() as session:
        assert session.get(ModelInstall, "model_delete_race")
        assert session.get(ModelProfile, "profile_delete_race")
    assert (model_path / "model.safetensors").read_bytes() == b"model"


async def test_chat_tool_capability_probe_executes_declared_schema(client: AsyncClient) -> None:
    response = await client.post("/api/engines/chat/tool-probe")
    assert response.status_code == 200
    assert response.json()["passed"] is True
    assert response.json()["arguments"] == {"mode": "image", "confidence": 1}


async def test_engine_api_isolates_media_settings_by_role(client: AsyncClient) -> None:
    response = await client.get("/api/engines")
    assert response.status_code == 200
    media = next(item for item in response.json() if {"image", "video"} <= set(item["roles"]))

    image_keys = [field["key"] for field in media["settings_by_role"]["image"]]
    video_keys = [field["key"] for field in media["settings_by_role"]["video"]]
    assert image_keys == [
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "denoise",
        "batch_size",
    ]
    assert video_keys == [
        "seed",
        "width",
        "height",
        "frames",
        "fps",
        "steps",
        "guidance",
        "motion_strength",
        "codec",
    ]
    assert "frames" not in image_keys
    assert "negative_prompt" not in video_keys
    assert len(media["settings"]) == len(image_keys) + len(video_keys)


async def test_random_media_seed_is_resolved_and_persisted(
    client: AsyncClient, monkeypatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", lambda _upper: 8675309)
    chat = (await client.post("/api/chats", json={"title": "Seed provenance"})).json()

    random_seed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image of a seed vault", "mode": "image"},
    )
    assert random_seed.status_code == 202
    random_run = random_seed.json()["run"]
    assert random_run["settings_json"]["seed"] == 8675309
    assert random_run["provenance_json"]["resolved_settings"]["seed"] == 8675309

    explicit_seed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a second image of a seed vault",
            "mode": "image",
            "settings": {"seed": 42},
        },
    )
    assert explicit_seed.status_code == 202
    explicit_run = explicit_seed.json()["run"]
    assert explicit_run["settings_json"]["seed"] == 42
    assert explicit_run["provenance_json"]["resolved_settings"]["seed"] == 42


async def test_uncertain_auto_media_requires_confirmation(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Routing"})).json()
    payload = {
        "text": "Maybe create an image of a quiet harbor",
        "mode": "auto",
        "idempotency_key": "uncertain-media-route",
    }
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "route_confirmation_required"
    assert detail["plan"]["operation"] == "text_to_image"

    confirmed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "confirm_media": True},
    )
    assert confirmed.status_code == 202
    assert confirmed.json()["run"]["operation"] == "text_to_image"


async def test_expensive_auto_video_reports_estimate_before_queueing(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Video estimate"})).json()
    payload = {
        "text": "Generate a video of a lighthouse through the night",
        "mode": "auto",
        "settings": {"width": 1024, "height": 576, "frames": 121, "steps": 30, "fps": 24},
        "idempotency_key": "expensive-video-route",
    }
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    estimate = detail["plan"]["parameter_overrides"]["_generation_estimate"]
    assert detail["plan"]["operation"] == "text_to_video"
    assert estimate["duration_seconds"] == 5.04
    assert estimate["estimated_intermediate_bytes"] > estimate["estimated_output_bytes"]

    confirmed = await client.post(
        f"/api/chats/{chat['id']}/turns", json={**payload, "confirm_media": True}
    )
    assert confirmed.status_code == 202
    assert confirmed.json()["run"]["provenance_json"]["generation_estimate"] == estimate


async def test_turn_idempotency_returns_original_run(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Idempotency"})).json()
    payload = {"text": "Hello", "mode": "text", "idempotency_key": "stable-key"}
    first = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    second = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "text": "A replay must not create a replacement turn"},
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["run"]["id"] == second.json()["run"]["id"]
    assert (
        second.json()["user_message"]["parts"][0]["text"]
        == first.json()["user_message"]["parts"][0]["text"]
    )


async def test_turn_idempotency_key_is_scoped_to_each_chat(client: AsyncClient) -> None:
    chats = [
        (await client.post("/api/chats", json={"title": title})).json()
        for title in ("First scope", "Second scope")
    ]
    responses = [
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": f"Request for {chat['title']}",
                "mode": "text",
                "idempotency_key": "shared-client-key",
            },
        )
        for chat in chats
    ]

    assert [response.status_code for response in responses] == [202, 202]
    runs = [response.json()["run"] for response in responses]
    assert runs[0]["id"] != runs[1]["id"]
    assert [run["chat_id"] for run in runs] == [chat["id"] for chat in chats]


async def test_turn_idempotency_never_bypasses_chat_existence(
    client: AsyncClient,
) -> None:
    source = (await client.post("/api/chats", json={"title": "Source"})).json()
    payload = {
        "text": "Private source request",
        "mode": "text",
        "idempotency_key": "chat-existence-key",
    }
    original = await client.post(f"/api/chats/{source['id']}/turns", json=payload)
    assert original.status_code == 202

    missing = await client.post("/api/chats/chat_missing/turns", json=payload)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "chat not found"

    deleted = (await client.post("/api/chats", json={"title": "Deleted"})).json()
    assert (await client.delete(f"/api/chats/{deleted['id']}")).status_code == 204
    replay_to_deleted = await client.post(
        f"/api/chats/{deleted['id']}/turns",
        json=payload,
    )
    assert replay_to_deleted.status_code == 404
    assert replay_to_deleted.json()["detail"] == "chat not found"


async def test_concurrent_orchestrators_converge_before_expensive_turn_planning(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Concurrent claim"})).json()
    services = app.state.services
    first: ConversationOrchestrator = services.orchestrator
    second = ConversationOrchestrator(
        services.engines,
        services.artifacts,
        services.events,
        services.scheduler,
        services.processes,
    )
    planner_started = asyncio.Event()
    release_planner = asyncio.Event()
    plan_calls = 0
    dispatched: list[tuple[str, str]] = []
    original_plan = first.router.plan_with_model

    async def counted_plan(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal plan_calls
        plan_calls += 1
        planner_started.set()
        await release_planner.wait()
        return await original_plan(**kwargs)

    def record_start(
        _orchestrator: ConversationOrchestrator,
        job_id: str,
        run_id: str,
    ) -> None:
        dispatched.append((job_id, run_id))

    monkeypatch.setattr(first.router, "plan_with_model", counted_plan)
    monkeypatch.setattr(second.router, "plan_with_model", counted_plan)
    monkeypatch.setattr(ConversationOrchestrator, "start", record_start)
    request = TurnRequest(
        text="Plan this exactly once",
        mode="text",
        idempotency_key="concurrent-cross-orchestrator",
    )

    with SessionLocal() as first_session, SessionLocal() as second_session:
        first_task = asyncio.create_task(first.create_turn(first_session, chat["id"], request))
        await asyncio.wait_for(planner_started.wait(), timeout=2)
        second_task = asyncio.create_task(second.create_turn(second_session, chat["id"], request))
        await asyncio.sleep(0.05)
        assert plan_calls == 1
        release_planner.set()
        first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result.run.id == second_result.run.id
    assert plan_calls == 1
    assert len(dispatched) == 1
    with SessionLocal() as session:
        runs = session.scalars(
            select(Run).where(
                Run.chat_id == chat["id"],
                Run.idempotency_key == request.idempotency_key,
            )
        ).all()
        jobs = session.scalars(select(Job).where(Job.run_id == first_result.run.id)).all()
        claims = session.scalars(
            select(TurnCreationClaim).where(TurnCreationClaim.chat_id == chat["id"])
        ).all()
    assert len(runs) == 1
    assert len(jobs) == 1
    assert claims == []


async def test_chat_turn_exposes_startup_status_until_text_arrives(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Startup status"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Reply briefly", "mode": "text"},
    )
    assert turn.status_code == 202
    accepted = turn.json()
    progress = next(
        part for part in accepted["assistant_message"]["parts"] if part["type"] == "progress"
    )
    assert progress["text"] == "Queued"
    assert progress["metadata_json"] == {
        "activity": "chat",
        "progress": 0,
        "phase": "queued",
    }

    await wait_for_run(client, accepted["run"]["id"])
    assistant = (await client.get(f"/api/messages/{accepted['assistant_message']['id']}")).json()
    assert any(part["type"] == "text" and part["text"] for part in assistant["parts"])
    assert not any(part["type"] == "progress" for part in assistant["parts"])


async def test_legacy_turn_creates_one_durable_work_plan_and_step(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Durable plan"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create one durable step",
            "mode": "text",
            "idempotency_key": "durable-plan-key",
        },
    )
    assert turn.status_code == 202
    accepted = turn.json()
    plan_id = accepted["run"]["work_plan_id"]
    step_id = accepted["run"]["work_step_id"]
    assert plan_id
    assert step_id
    await wait_for_run(client, accepted["run"]["id"])

    plan = await client.get(f"/api/work-plans/{plan_id}")
    assert plan.status_code == 200
    assert plan.json()["source_action"] == "send"
    assert plan.json()["status"] == "complete"
    assert plan.json()["transcript_sequence"] == 1
    assert len(plan.json()["steps"]) == 1
    assert plan.json()["steps"][0]["id"] == step_id
    assert plan.json()["steps"][0]["run_id"] == accepted["run"]["id"]
    assert plan.json()["steps"][0]["status"] == "complete"

    step = await client.get(f"/api/work-steps/{step_id}")
    assert step.status_code == 200
    assert step.json()["queue_class"] == "interactive_compute"
    jobs = (await client.get("/api/jobs")).json()
    job = next(item for item in jobs if item["run_id"] == accepted["run"]["id"])
    assert job["work_plan_id"] == plan_id
    assert job["work_step_id"] == step_id
    assert job["progress_json"]["version"] == 2
    assert job["progress_json"]["overall_progress"] == 1

    replay = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create one durable step",
            "mode": "text",
            "idempotency_key": "durable-plan-key",
        },
    )
    assert replay.status_code == 202
    assert replay.json()["run"]["id"] == accepted["run"]["id"]
    assert replay.json()["run"]["work_plan_id"] == plan_id
    listed = (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json()
    assert [item["id"] for item in listed] == [plan_id]


async def test_media_variations_create_ordered_independent_output_slots(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Four variations"})).json()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Create four variations of a blue ceramic apple",
                "mode": "image",
                "idempotency_key": "four-media-outputs",
            },
        )
        assert response.status_code == 202
        accepted = response.json()
        plan = (await client.get(f"/api/work-plans/{accepted['run']['work_plan_id']}")).json()
        assert plan["planner_version"] == "media-outputs-v1"
        assert plan["summary_json"]["output_count"] == 4
        assert plan["summary_json"]["status_counts"] == {"queued": 4}
        assert [step["ordinal"] for step in plan["steps"]] == [1, 2, 3, 4]
        assert [step["output_contract_json"][0]["slot"] for step in plan["steps"]] == [
            "output-1",
            "output-2",
            "output-3",
            "output-4",
        ]
        assert all(step["display_group"] == "media_outputs" for step in plan["steps"])
        assert all(step["settings_json"].get("batch_size") == 1 for step in plan["steps"])

        with SessionLocal() as session:
            runs = session.scalars(
                select(Run).where(Run.work_plan_id == plan["id"]).order_by(Run.created_at, Run.id)
            ).all()
            steps = session.scalars(
                select(WorkStep).where(WorkStep.plan_id == plan["id"]).order_by(WorkStep.ordinal)
            ).all()
            jobs = session.scalars(
                select(Job).where(Job.work_plan_id == plan["id"]).order_by(Job.queue_ticket)
            ).all()
            assert len(runs) == len(steps) == len(jobs) == 4
            assert len({run.assistant_message_id for run in runs}) == 4
            assert [job.payload_json["output_index"] for job in jobs] == [1, 2, 3, 4]
            assert [job.payload_json["output_count"] for job in jobs] == [4, 4, 4, 4]
            assert runs[0].idempotency_key == "four-media-outputs"
            assert all(run.idempotency_key is None for run in runs[1:])
            assistant_ids = plan["summary_json"]["assistant_message_ids"]
            messages = [session.get(Message, message_id) for message_id in assistant_ids]
            assert all(message is not None for message in messages)
            assert messages[0].parent_id == accepted["user_message"]["id"]
            assert [message.parent_id for message in messages[1:]] == assistant_ids[:-1]

        second_step_id = plan["steps"][1]["id"]
        cancelled = await client.post(f"/api/work-steps/{second_step_id}/cancel")
        assert cancelled.status_code == 200
        updated = (await client.get(f"/api/work-plans/{plan['id']}")).json()
        assert [step["status"] for step in updated["steps"]] == [
            "queued",
            "cancelled",
            "queued",
            "queued",
        ]
        assert updated["status"] == "queued"
        assert updated["summary_json"]["status_counts"] == {
            "queued": 3,
            "cancelled": 1,
        }

        retried = await client.post(f"/api/work-steps/{second_step_id}/retry")
        assert retried.status_code == 200
        updated = (await client.get(f"/api/work-plans/{plan['id']}")).json()
        assert [step["status"] for step in updated["steps"]] == ["queued"] * 4

    for run_id in plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)
    completed = (await client.get(f"/api/work-plans/{plan['id']}")).json()
    assert completed["status"] == "complete"
    assert completed["summary_json"]["status_counts"] == {"complete": 4}
    assert (await client.delete(f"/api/chats/{chat['id']}")).status_code == 204
    with SessionLocal() as session:
        assert session.get(WorkPlan, plan["id"]) is None
        assert not session.scalar(select(Job.id).where(Job.work_plan_id == plan["id"]))
        assert not session.scalar(select(Run.id).where(Run.work_plan_id == plan["id"]))


async def test_media_output_count_is_bounded_before_any_turn_is_written(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Output bound"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create nine variations of a red square",
            "mode": "image",
        },
    )
    assert response.status_code == 422
    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert detail["messages"] == []
    assert (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json() == []


async def test_ordered_text_image_video_text_plan_resolves_typed_outputs(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Ordered work"})).json()
    payload = {
        "text": (
            "Write a short story about a paper boat, then create an image based on it, "
            "then animate the image into a video, then summarize the video"
        ),
        "mode": "auto",
        "ordered_settings": {
            "chat": {"max_tokens": 64},
            "image": {"width": 512},
            "video": {"frames": 49},
        },
        "idempotency_key": "ordered-work-chain",
    }
    preview = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert preview.status_code == 409
    assert preview.json()["detail"]["code"] == "ordered_plan_confirmation_required"
    assert [step["mode"] for step in preview.json()["detail"]["plan"]["steps"]] == [
        "text",
        "image",
        "video",
        "text",
    ]
    assert preview.json()["detail"]["estimate"]["video_duration_seconds"] > 0
    assert preview.json()["detail"]["estimate"]["estimated_bytes"] > 0
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["messages"] == []
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "confirm_media": True},
    )
    assert response.status_code == 202
    accepted = response.json()
    plan_id = accepted["run"]["work_plan_id"]
    plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert plan["planner_version"] == "ordered-work-v1"
    assert plan["failure_policy"] == "preserve_completed_block_dependents"
    assert [step["operation"] for step in plan["steps"]] == [
        "text",
        "text_to_image",
        "image_to_video",
        "text",
    ]
    assert [step["ordinal"] for step in plan["steps"]] == [1, 2, 3, 4]
    assert plan["summary_json"]["intent"]["planner_version"] == "ordered-work-v1"

    for run_id in plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)

    with SessionLocal() as session:
        steps = session.scalars(
            select(WorkStep).where(WorkStep.plan_id == plan_id).order_by(WorkStep.ordinal)
        ).all()
        runs = [session.get(Run, step.run_id) for step in steps]
        assert all(run is not None for run in runs)
        assert runs[0].settings_json["max_tokens"] == 64
        assert runs[1].settings_json["width"] == 512
        assert "frames" not in runs[1].settings_json
        assert runs[2].settings_json["frames"] == 49
        assert runs[3].settings_json["max_tokens"] == 64
        assert steps[0].input_bindings_json == []
        assert steps[1].input_bindings_json == [
            {"type": "step_output.text", "source_step_id": steps[0].id}
        ]
        assert steps[2].input_bindings_json == [
            {"type": "step_output.artifact", "source_step_id": steps[1].id}
        ]
        assert steps[3].input_bindings_json == [
            {"type": "step_output.artifact", "source_step_id": steps[2].id}
        ]
        assert (
            runs[1].provenance_json["resolved_dependency_text"][0]["source_step_id"] == steps[0].id
        )
        image_artifact_ids = runs[2].provenance_json["resolved_dependency_artifact_ids"]
        video_artifact_ids = runs[3].provenance_json["resolved_dependency_artifact_ids"]
        assert len(image_artifact_ids) == 1
        assert len(video_artifact_ids) == 1
        image_artifact = session.get(Artifact, image_artifact_ids[0])
        video_artifact = session.get(Artifact, video_artifact_ids[0])
        assert image_artifact and image_artifact.media_type.startswith("image/")
        assert video_artifact and video_artifact.media_type.startswith("video/")
        for resolved_step, resolved_run in zip(steps, runs, strict=True):
            assert resolved_run
            if resolved_run.profile_id:
                profile = session.get(ModelProfile, resolved_run.profile_id)
                assert profile
                expected_role = (
                    "chat"
                    if resolved_step.operation == "text"
                    else "video"
                    if "video" in resolved_step.operation
                    else "image"
                )
                assert profile.role == expected_role

    completed = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert completed["status"] == "complete"
    assert completed["summary_json"]["status_counts"] == {"complete": 4}


async def test_ordered_plan_blocks_dependents_and_resumes_after_retry(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_stream = MockChatAdapter.stream

    async def fail_first_step(
        self: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        del self, request
        yield ChatEvent(type="error", data={"error": "Synthetic story failure"})

    monkeypatch.setattr(MockChatAdapter, "stream", fail_first_step)
    chat = (await client.post("/api/chats", json={"title": "Blocked chain"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": (
                "Write a short story, then create an image based on it, then describe the image"
            ),
            "mode": "auto",
            "confirm_media": True,
        },
    )
    assert response.status_code == 202
    plan_id = response.json()["run"]["work_plan_id"]
    deadline = asyncio.get_running_loop().time() + 5
    blocked_plan: dict = {}
    while asyncio.get_running_loop().time() < deadline:
        blocked_plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
        if [step["status"] for step in blocked_plan["steps"]] == [
            "failed",
            "blocked",
            "blocked",
        ]:
            break
        await asyncio.sleep(0.03)
    assert [step["status"] for step in blocked_plan["steps"]] == [
        "failed",
        "blocked",
        "blocked",
    ]
    assert blocked_plan["status"] == "blocked"
    assert blocked_plan["summary_json"]["status_counts"] == {
        "blocked": 2,
        "failed": 1,
    }
    jobs = (await client.get("/api/jobs")).json()
    blocked_jobs = [
        job for job in jobs if job["work_plan_id"] == plan_id and job["status"] == "queued"
    ]
    assert len(blocked_jobs) == 2
    assert all(job["progress_json"]["blocked_by"] for job in blocked_jobs)

    monkeypatch.setattr(MockChatAdapter, "stream", original_stream)
    retry = await client.post(f"/api/work-steps/{blocked_plan['steps'][0]['id']}/retry")
    assert retry.status_code == 200
    for run_id in blocked_plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)
    completed = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert completed["status"] == "complete"
    assert completed["summary_json"]["status_counts"] == {"complete": 3}
    jobs = (await client.get("/api/jobs")).json()
    attempts = {
        job["work_step_id"]: job["attempt"] for job in jobs if job["work_plan_id"] == plan_id
    }
    assert attempts[blocked_plan["steps"][0]["id"]] == 2
    assert attempts[blocked_plan["steps"][1]["id"]] == 1
    assert attempts[blocked_plan["steps"][2]["id"]] == 1


async def test_ordered_retry_preserves_completed_predecessor(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_generate = MockMediaAdapter.generate

    async def fail_media_step(
        self: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        del self, request
        raise RuntimeError("Synthetic media failure")
        yield MediaEvent(type="complete")  # pragma: no cover

    monkeypatch.setattr(MockMediaAdapter, "generate", fail_media_step)
    chat = (await client.post("/api/chats", json={"title": "Preserved chain"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": (
                "Write a short scene, then create an image based on it, then describe the image"
            ),
            "mode": "auto",
            "confirm_media": True,
        },
    )
    plan_id = response.json()["run"]["work_plan_id"]
    deadline = asyncio.get_running_loop().time() + 5
    failed_plan: dict = {}
    while asyncio.get_running_loop().time() < deadline:
        failed_plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
        if [step["status"] for step in failed_plan["steps"]] == [
            "complete",
            "failed",
            "blocked",
        ]:
            break
        await asyncio.sleep(0.03)
    assert [step["status"] for step in failed_plan["steps"]] == [
        "complete",
        "failed",
        "blocked",
    ]
    first_message_id = failed_plan["summary_json"]["assistant_message_ids"][0]
    first_message = (await client.get(f"/api/messages/{first_message_id}")).json()

    monkeypatch.setattr(MockMediaAdapter, "generate", original_generate)
    retry = await client.post(f"/api/work-steps/{failed_plan['steps'][1]['id']}/retry")
    assert retry.status_code == 200
    for run_id in failed_plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)
    assert (await client.get(f"/api/messages/{first_message_id}")).json() == first_message
    jobs = (await client.get("/api/jobs")).json()
    attempts = {
        job["work_step_id"]: job["attempt"] for job in jobs if job["work_plan_id"] == plan_id
    }
    assert attempts[failed_plan["steps"][0]["id"]] == 1
    assert attempts[failed_plan["steps"][1]["id"]] == 2
    assert attempts[failed_plan["steps"][2]["id"]] == 1


async def test_invalid_ordered_plan_settings_fail_before_writes(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Invalid ordered"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Write a short story, then create an image based on it",
            "mode": "auto",
            "confirm_media": True,
            "ordered_settings": {"image": {"arbitrary_node": "execute"}},
        },
    )
    assert response.status_code == 422
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["messages"] == []
    assert (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json() == []


async def test_turn_remains_truthfully_queued_until_compute_lease(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Queue truth"})).json()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "Wait for the real lease", "mode": "text"},
        )
        assert response.status_code == 202
        accepted = response.json()
        await asyncio.sleep(0.05)
        run = (await client.get(f"/api/runs/{accepted['run']['id']}")).json()
        jobs = (await client.get("/api/jobs")).json()
        job = next(item for item in jobs if item["run_id"] == run["id"])
        assert run["status"] == "queued"
        assert run["started_at"] is None
        assert job["status"] == "queued"
        assert job["started_at"] is None
        assert job["progress_json"]["stage"] == "queued"
        assert job["progress_json"]["queue_position"] == 0

    await wait_for_run(client, accepted["run"]["id"])


async def test_three_rapid_turns_are_admitted_once_in_transcript_order(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Rapid turns"})).json()
    prompts = ["First prompt", "Second prompt", "Third prompt"]
    async with app.state.services.scheduler.lease("primary"):
        responses = [
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={
                    "text": prompt,
                    "mode": "text",
                    "idempotency_key": f"rapid-{index}",
                },
            )
            for index, prompt in enumerate(prompts, start=1)
        ]
        assert all(response.status_code == 202 for response in responses)
        detail = (await client.get(f"/api/chats/{chat['id']}")).json()
        user_text = [
            part["text"]
            for message in detail["messages"]
            if message["role"] == "user"
            for part in message["parts"]
            if part["type"] == "text"
        ]
        assert user_text == prompts
        plans = (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json()
        assert [plan["transcript_sequence"] for plan in reversed(plans)] == [1, 2, 3]
        assert len({response.json()["run"]["id"] for response in responses}) == 3
        with SessionLocal() as session:
            third_run = session.get(Run, responses[2].json()["run"]["id"])
            assert third_run
            assert ConversationOrchestrator._context_messages(session, third_run) == [
                {"role": "user", "content": prompt} for prompt in prompts
            ]

    for response in responses:
        await wait_for_run(client, response.json()["run"]["id"])


async def test_pending_output_reference_creates_a_dispatch_dependency(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Pending dependency"})).json()
    async with app.state.services.scheduler.lease("primary"):
        first = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "Write a tiny story", "mode": "text"},
            )
        ).json()
        second = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "Summarize that response", "mode": "text"},
            )
        ).json()
        with SessionLocal() as session:
            first_run = session.get(Run, first["run"]["id"])
            second_run = session.get(Run, second["run"]["id"])
            assert first_run and second_run
            second_plan = session.get(WorkPlan, second_run.work_plan_id)
            dependency = session.scalar(
                select(WorkStepDependency).where(
                    WorkStepDependency.step_id == second_run.work_step_id
                )
            )
            assert second_plan
            assert second_plan.context_head_message_id == first["assistant_message"]["id"]
            assert dependency
            assert dependency.depends_on_step_id == first_run.work_step_id
            eligible = ResourceScheduler._eligible_jobs(session, "primary", utcnow())
            assert [job.run_id for job in eligible] == [first_run.id]

    await wait_for_run(client, first["run"]["id"])
    await wait_for_run(client, second["run"]["id"])


async def test_independent_text_is_prioritized_ahead_of_untouched_media(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Interactive priority"})).json()
    async with app.state.services.scheduler.lease("primary"):
        image = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "Create an abstract image", "mode": "image"},
            )
        ).json()
        text_turn = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "Explain binary search", "mode": "text"},
            )
        ).json()
        with SessionLocal() as session:
            eligible = ResourceScheduler._eligible_jobs(session, "primary", utcnow())
            assert [job.run_id for job in eligible] == [
                text_turn["run"]["id"],
                image["run"]["id"],
            ]
            text_run = session.get(Run, text_turn["run"]["id"])
            image_run = session.get(Run, image["run"]["id"])
            assert text_run and image_run
            text_plan = session.get(WorkPlan, text_run.work_plan_id)
            image_job = session.scalar(select(Job).where(Job.run_id == image_run.id))
            assert text_plan
            assert text_plan.context_head_message_id == image["user_message"]["id"]
            assert image_job and image_job.status == JobStatus.QUEUED.value
            assert not session.scalar(
                select(WorkStepDependency).where(
                    WorkStepDependency.step_id == text_run.work_step_id
                )
            )

    await wait_for_run(client, text_turn["run"]["id"])
    await wait_for_run(client, image["run"]["id"])


async def test_stop_and_send_cancels_current_item_then_admits_replacement(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Stop and send"})).json()
    async with app.state.services.scheduler.lease("primary"):
        first = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "Original queued prompt", "mode": "text"},
            )
        ).json()
        replacement = await client.post(
            f"/api/chats/{chat['id']}/stop-and-send",
            json={"text": "Replacement prompt", "mode": "text"},
        )
        assert replacement.status_code == 202
        jobs = (await client.get("/api/jobs")).json()
        first_job = next(job for job in jobs if job["run_id"] == first["run"]["id"])
        assert first_job["status"] == "cancelled"
        plan = (
            await client.get(f"/api/work-plans/{replacement.json()['run']['work_plan_id']}")
        ).json()
        assert plan["source_action"] == "stop_and_send"

    await wait_for_run(client, replacement.json()["run"]["id"])


async def test_plan_cancel_and_retry_control_the_queued_step(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Plan controls"})).json()
    async with app.state.services.scheduler.lease("primary"):
        accepted = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "Queue this plan", "mode": "text"},
            )
        ).json()
        plan_id = accepted["run"]["work_plan_id"]
        cancelled = await client.post(f"/api/work-plans/{plan_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert cancelled.json()["steps"][0]["status"] == "cancelled"

        retried = await client.post(f"/api/work-plans/{plan_id}/retry")
        assert retried.status_code == 200
        assert retried.json()["status"] == "queued"
        assert retried.json()["steps"][0]["status"] == "queued"

        step_id = retried.json()["steps"][0]["id"]
        step_cancelled = await client.post(f"/api/work-steps/{step_id}/cancel")
        assert step_cancelled.status_code == 200
        assert step_cancelled.json()["status"] == "cancelled"

        step_retried = await client.post(f"/api/work-steps/{step_id}/retry")
        assert step_retried.status_code == 200
        assert step_retried.json()["status"] == "queued"

    await wait_for_run(client, accepted["run"]["id"])


async def test_chat_pending_admission_limit_is_bounded(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Admission bound"})).json()
    async with app.state.services.scheduler.lease("primary"):
        accepted = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": "The real queued turn", "mode": "text"},
            )
        ).json()
        with SessionLocal() as session:
            session.add_all(
                [
                    Job(
                        id=f"job_admission_{index}",
                        kind="chat",
                        status=JobStatus.QUEUED.value,
                        run_id=accepted["run"]["id"],
                    )
                    for index in range(31)
                ]
            )
            session.commit()
        rejected = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "One too many", "mode": "text"},
        )
        assert rejected.status_code == 422
        assert "32 pending items" in rejected.json()["detail"]

    await wait_for_run(client, accepted["run"]["id"])


async def test_incognito_conversation_and_artifact_never_enter_durable_storage(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch,
) -> None:
    marker = "INCOGNITO-MARKER-7c4ef64c"
    observation_marker = "INCOGNITO-VISION-OBSERVATION-a91d8c2e"
    original_capabilities = MockChatAdapter.capabilities

    async def vision_capabilities(adapter: MockChatAdapter) -> EngineCapabilities:
        capabilities = await original_capabilities(adapter)
        return capabilities.model_copy(update={"input_modalities": ["text", "image"]})

    async def private_stream(
        _adapter: MockChatAdapter,
        _request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="delta", text=observation_marker)
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    monkeypatch.setattr(MockChatAdapter, "capabilities", vision_capabilities)
    monkeypatch.setattr(MockChatAdapter, "stream", private_stream)
    durable = (await client.post("/api/chats", json={"title": "Durable control"})).json()
    started = await client.post("/api/incognito/session")
    assert started.status_code == 201
    token = started.json()["token"]
    assert "forensic erasure" in started.json()["disclosure"]
    assert app.state.services.incognito
    scope = app.state.services.incognito.require(token)
    scope_root = scope.root

    client.headers[INCOGNITO_HEADER] = token
    private_chat = (await client.post("/api/chats", json={"title": "Private session"})).json()
    accepted = (
        await client.post(
            f"/api/chats/{private_chat['id']}/turns",
            json={"text": marker, "mode": "text"},
        )
    ).json()
    await wait_for_run(client, accepted["run"]["id"])
    private_image = await client.post(
        "/api/artifacts",
        files={"file": ("private-pixel.png", ONE_PIXEL_PNG, "image/png")},
    )
    assert private_image.status_code == 201
    vision_turn = (
        await client.post(
            f"/api/chats/{private_chat['id']}/turns",
            json={
                "text": "Inspect this private image",
                "mode": "text",
                "input_artifact_ids": [private_image.json()["id"]],
            },
        )
    ).json()
    await wait_for_run(client, vision_turn["run"]["id"])
    private_run = (await client.get(f"/api/runs/{vision_turn['run']['id']}")).json()
    assert private_run["provenance_json"]["context"]["vision"]["visual_contents_inspected"] is True
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("marker.txt", marker.encode(), "text/plain")},
    )
    assert uploaded.status_code == 201
    content = await client.get(f"/api/artifacts/{uploaded.json()['id']}/content")
    assert content.content == marker.encode()
    assert content.headers["cache-control"] == "no-store"

    del client.headers[INCOGNITO_HEADER]
    durable_chats = (await client.get("/api/chats?include_archived=true")).json()
    assert [chat["id"] for chat in durable_chats] == [durable["id"]]
    with SessionLocal() as session:
        assert not session.scalar(select(MessagePart).where(MessagePart.text.contains(marker)))
        assert not session.scalar(
            select(MessagePart).where(MessagePart.text.contains(observation_marker))
        )
        assert session.get(Run, vision_turn["run"]["id"]) is None
        assert session.get(Artifact, private_image.json()["id"]) is None
    durable_artifact_bytes = b"".join(
        path.read_bytes() for path in settings.artifact_dir.rglob("*") if path.is_file()
    )
    assert marker.encode() not in durable_artifact_bytes
    assert ONE_PIXEL_PNG not in durable_artifact_bytes

    client.headers[INCOGNITO_HEADER] = token
    ended = await client.delete("/api/incognito/session")
    assert ended.status_code == 204
    assert not scope_root.exists()
    assert (await client.get(f"/api/chats/{private_chat['id']}")).status_code == 404
    del client.headers[INCOGNITO_HEADER]
    assert (await client.get(f"/api/chats/{durable['id']}")).status_code == 200


async def test_ordered_plan_remains_inside_incognito_scope(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    started = await client.post("/api/incognito/session")
    token = started.json()["token"]
    scope = app.state.services.incognito.require(token)
    scope_root = scope.root
    client.headers[INCOGNITO_HEADER] = token
    chat = (await client.post("/api/chats", json={"title": "Private ordered"})).json()
    accepted = (
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Write a private scene, then create an image based on it",
                "mode": "auto",
                "confirm_media": True,
            },
        )
    ).json()
    plan = (await client.get(f"/api/work-plans/{accepted['run']['work_plan_id']}")).json()
    for run_id in plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)
    private_run_ids = plan["summary_json"]["run_ids"]
    private_plan_id = plan["id"]

    del client.headers[INCOGNITO_HEADER]
    with SessionLocal() as session:
        assert session.get(WorkPlan, private_plan_id) is None
        assert all(session.get(Run, run_id) is None for run_id in private_run_ids)
    client.headers[INCOGNITO_HEADER] = token
    assert (await client.delete("/api/incognito/session")).status_code == 204
    assert not scope_root.exists()
    del client.headers[INCOGNITO_HEADER]


async def test_active_chat_run_can_be_cancelled_directly(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Stop response"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write a response long enough to stop", "mode": "text"},
    )
    assert turn.status_code == 202
    assistant_id = turn.json()["assistant_message"]["id"]
    deadline = asyncio.get_running_loop().time() + 5
    streamed_text = ""
    while asyncio.get_running_loop().time() < deadline:
        assistant = (await client.get(f"/api/messages/{assistant_id}")).json()
        streamed_text = "".join(
            part["text"] or "" for part in assistant["parts"] if part["type"] == "text"
        )
        if streamed_text:
            break
        await asyncio.sleep(0.01)
    assert streamed_text

    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    run = (await client.get(f"/api/runs/{turn.json()['run']['id']}")).json()
    assert run["status"] == "cancelled"
    assistant = (await client.get(f"/api/messages/{assistant_id}")).json()
    assert assistant["status"] == "cancelled"
    final_text = "".join(
        part["text"] or "" for part in assistant["parts"] if part["type"] == "text"
    )
    assert final_text.startswith(streamed_text.rstrip())
    assert not any(part["type"] == "error" for part in assistant["parts"])


async def test_failed_chat_run_preserves_streamed_text_and_reports_error(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def fail_after_delta(
        _adapter: MockChatAdapter, _request: ChatRequest
    ) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="delta", text="Partial response that should remain")
        yield ChatEvent(type="error", data={"error": ""})

    monkeypatch.setattr(MockChatAdapter, "stream", fail_after_delta)
    chat = (await client.post("/api/chats", json={"title": "Failed response"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Start a response", "mode": "text"},
    )
    assert turn.status_code == 202

    deadline = asyncio.get_running_loop().time() + 5
    run = turn.json()["run"]
    while asyncio.get_running_loop().time() < deadline:
        run = (await client.get(f"/api/runs/{run['id']}")).json()
        if run["status"] == "failed":
            break
        await asyncio.sleep(0.01)
    assert run["status"] == "failed"
    assert run["error"] == "Chat engine stream failed"

    assistant = (await client.get(f"/api/messages/{turn.json()['assistant_message']['id']}")).json()
    assert assistant["status"] == "failed"
    assert [
        (part["type"], part["text"])
        for part in assistant["parts"]
        if part["type"] in {"text", "error"}
    ] == [
        ("text", "Partial response that should remain"),
        ("error", "Chat engine stream failed"),
    ]


async def test_restart_recovery_preserves_partial_text_and_appends_error(
    client: AsyncClient,
    app: FastAPI,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Restart recovery"})).json()
    turn = (
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "Begin a response", "mode": "text"},
        )
    ).json()
    await wait_for_run(client, turn["run"]["id"])

    with SessionLocal() as session:
        run = session.get(Run, turn["run"]["id"])
        job = session.scalar(select(Job).where(Job.run_id == turn["run"]["id"]))
        message = session.get(Message, turn["assistant_message"]["id"])
        assert run and job and message
        run.status = "running"
        run.error = None
        run.completed_at = None
        job.status = JobStatus.RUNNING.value
        job.error = None
        job.completed_at = None
        message.status = "pending"
        message.parts.clear()
        session.flush()
        message.parts.extend(
            [
                MessagePart(position=0, type="text", text="Durable partial response"),
                MessagePart(
                    position=1,
                    type="progress",
                    text="Waiting",
                    metadata_json={"activity": "chat"},
                ),
            ]
        )
        session.commit()

    app.state.services.orchestrator.recover_interrupted()

    assistant = (await client.get(f"/api/messages/{turn['assistant_message']['id']}")).json()
    assert assistant["status"] == "failed"
    assert [
        (part["type"], part["text"])
        for part in assistant["parts"]
        if part["type"] in {"text", "error"}
    ] == [
        ("text", "Durable partial response"),
        ("error", "The application restarted before this job completed."),
    ]
    assert not any(part["type"] == "progress" for part in assistant["parts"])


async def test_restart_recovery_requeues_work_that_never_started(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Queued recovery"})).json()
    turn = (
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "Remain queued", "mode": "text"},
        )
    ).json()
    await wait_for_run(client, turn["run"]["id"])
    with SessionLocal() as session:
        run = session.get(Run, turn["run"]["id"])
        job = session.scalar(select(Job).where(Job.run_id == turn["run"]["id"]))
        assert run and job
        run.status = "queued"
        run.started_at = None
        run.completed_at = None
        job.status = JobStatus.QUEUED.value
        job.started_at = None
        job.completed_at = None
        job.claim_owner = "stale-dispatcher"
        session.commit()
        job_id = job.id

    restarted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator,
        "start",
        lambda candidate_job_id, candidate_run_id: restarted.append(
            (candidate_job_id, candidate_run_id)
        ),
    )

    app.state.services.orchestrator.recover_interrupted()

    with SessionLocal() as session:
        recovered = session.get(Job, job_id)
        assert recovered
        assert recovered.status == JobStatus.QUEUED.value
        assert recovered.claim_owner is None
        assert recovered.progress_json["stage"] == "queued"
    assert restarted == [(job_id, turn["run"]["id"])]


async def test_restart_recovery_preserves_queued_transcript_order(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Ordered recovery"})).json()
    turns = []
    for index in range(3):
        turn = (
            await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={"text": f"Queued item {index + 1}", "mode": "text"},
            )
        ).json()
        await wait_for_run(client, turn["run"]["id"])
        turns.append(turn)

    with SessionLocal() as session:
        expected: list[tuple[str, str]] = []
        for turn in turns:
            run = session.get(Run, turn["run"]["id"])
            job = session.scalar(select(Job).where(Job.run_id == turn["run"]["id"]))
            assert run and job
            run.status = JobStatus.QUEUED.value
            run.started_at = None
            run.completed_at = None
            job.status = JobStatus.QUEUED.value
            job.started_at = None
            job.completed_at = None
            job.claim_owner = "stale-dispatcher"
            expected.append((job.id, run.id))
        session.commit()

    restarted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator,
        "start",
        lambda candidate_job_id, candidate_run_id: restarted.append(
            (candidate_job_id, candidate_run_id)
        ),
    )

    app.state.services.orchestrator.recover_interrupted()

    assert restarted == expected


async def test_restart_recovery_preserves_media_output_order(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Media recovery"})).json()
    turn = (
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Create three variations of a paper boat",
                "mode": "image",
            },
        )
    ).json()
    plan_id = turn["run"]["work_plan_id"]
    plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
    for run_id in plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)

    with SessionLocal() as session:
        jobs = session.scalars(
            select(Job).where(Job.work_plan_id == plan_id).order_by(Job.queue_ticket)
        ).all()
        expected = [(job.id, job.run_id) for job in jobs]
        for job in jobs:
            run = session.get(Run, job.run_id)
            step = session.get(WorkStep, job.work_step_id)
            assert run and step
            job.status = JobStatus.QUEUED.value
            job.started_at = None
            job.completed_at = None
            job.claim_owner = "stale-dispatcher"
            run.status = JobStatus.QUEUED.value
            run.started_at = None
            run.completed_at = None
            step.status = JobStatus.QUEUED.value
        work_plan = session.get(WorkPlan, plan_id)
        assert work_plan
        work_plan.status = JobStatus.QUEUED.value
        session.commit()

    restarted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator,
        "start",
        lambda job_id, run_id: restarted.append((job_id, run_id)),
    )
    app.state.services.orchestrator.recover_interrupted()

    assert restarted == expected
    recovered = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert recovered["status"] == "queued"
    assert recovered["summary_json"]["status_counts"] == {"queued": 3}


async def test_restart_recovery_does_not_replay_completed_ordered_steps(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Ordered restart"})).json()
    turn = (
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": (
                    "Write a short scene, then create an image based on it, then describe the image"
                ),
                "mode": "auto",
                "confirm_media": True,
            },
        )
    ).json()
    plan_id = turn["run"]["work_plan_id"]
    plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
    for run_id in plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)

    first_message_id = plan["summary_json"]["assistant_message_ids"][0]
    first_message = (await client.get(f"/api/messages/{first_message_id}")).json()
    with SessionLocal() as session:
        steps = session.scalars(
            select(WorkStep).where(WorkStep.plan_id == plan_id).order_by(WorkStep.ordinal)
        ).all()
        assert len(steps) == 3
        expected: list[tuple[str, str]] = []
        for step in steps[1:]:
            run = session.get(Run, step.run_id)
            job = session.scalar(select(Job).where(Job.work_step_id == step.id))
            assert run and job
            step.status = JobStatus.QUEUED.value
            step.error = None
            run.status = JobStatus.QUEUED.value
            run.started_at = None
            run.completed_at = None
            run.error = None
            job.status = JobStatus.QUEUED.value
            job.started_at = None
            job.completed_at = None
            job.error = None
            job.claim_owner = "stale-dispatcher"
            expected.append((job.id, run.id))
        work_plan = session.get(WorkPlan, plan_id)
        assert work_plan
        work_plan.status = JobStatus.QUEUED.value
        session.commit()

    restarted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator,
        "start",
        lambda job_id, run_id: restarted.append((job_id, run_id)),
    )
    app.state.services.orchestrator.recover_interrupted()

    assert restarted == expected
    assert (await client.get(f"/api/messages/{first_message_id}")).json() == first_message
    recovered = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert [step["status"] for step in recovered["steps"]] == [
        "complete",
        "queued",
        "queued",
    ]
    assert recovered["summary_json"]["status_counts"] == {
        "complete": 1,
        "queued": 2,
    }


async def test_retry_clears_stale_error_before_dispatch(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Clean retry"})).json()
    turn = (
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "Try once", "mode": "text"},
        )
    ).json()
    await wait_for_run(client, turn["run"]["id"])
    with SessionLocal() as session:
        run = session.get(Run, turn["run"]["id"])
        job = session.scalar(select(Job).where(Job.run_id == turn["run"]["id"]))
        message = session.get(Message, turn["assistant_message"]["id"])
        assert run and job and message
        run.status = "failed"
        run.error = "Prior failure"
        job.status = JobStatus.FAILED.value
        job.error = "Prior failure"
        message.status = "failed"
        message.parts.append(
            MessagePart(
                position=max(part.position for part in message.parts) + 1,
                type="error",
                text="Prior failure",
            )
        )
        session.commit()
        job_id = job.id

    dispatched: list[tuple[str, str]] = []

    def start(job_id: str, run_id: str) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            message = session.get(Message, run.assistant_message_id) if run else None
            assert job and run and message
            assert job.status == JobStatus.QUEUED.value
            assert job.error is None
            assert run.status == "queued"
            assert run.error is None
            assert message.status == "pending"
            assert not any(part.type == "error" for part in message.parts)
        dispatched.append((job_id, run_id))

    monkeypatch.setattr(app.state.services.orchestrator, "start", start)
    retried = await client.post(f"/api/jobs/{job_id}/retry")

    assert retried.status_code == 200
    assert dispatched == [(job_id, turn["run"]["id"])]
    message = (await client.get(f"/api/messages/{turn['assistant_message']['id']}")).json()
    assert message["status"] == "pending"
    assert [(part["type"], part["text"]) for part in message["parts"]] == [
        ("text", ""),
        ("progress", "Queued"),
    ]


async def test_cancelled_media_run_can_be_retried(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Retry cancelled media"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a retryable video", "mode": "video"},
    )
    assert turn.status_code == 202

    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    retried = await client.post(f"/api/jobs/{cancelled.json()['id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] in {"queued", "running"}

    run = await wait_for_run(client, turn.json()["run"]["id"])
    assert run["operation"] == "text_to_video"
    jobs = (await client.get("/api/jobs")).json()
    completed = next(job for job in jobs if job["id"] == cancelled.json()["id"])
    assert completed["status"] == "complete"
    assert completed["attempt"] == 2


async def test_editing_user_message_creates_new_active_branch(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Edit branch"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Original question", "mode": "text"}
    )
    assert first.status_code == 202
    await wait_for_run(client, first.json()["run"]["id"])
    second = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Old branch follow-up", "mode": "text"}
    )
    assert second.status_code == 202
    await wait_for_run(client, second.json()["run"]["id"])

    branched = await client.post(
        f"/api/messages/{first.json()['user_message']['id']}/branch",
        json={"text": "Edited question"},
    )
    assert branched.status_code == 202
    payload = branched.json()
    assert payload["run"]["operation"] == "text"
    assert payload["user_message"]["parent_id"] == first.json()["user_message"]["parent_id"]
    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert detail["active_head_message_id"] == payload["assistant_message"]["id"]


async def test_editing_image_request_as_text_does_not_inherit_media_role(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Change edited mode"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a blue apple",
            "mode": "image",
            "settings": {"steps": 7},
        },
    )
    assert original.status_code == 202
    await wait_for_run(client, original.json()["run"]["id"])

    branched = await client.post(
        f"/api/messages/{original.json()['user_message']['id']}/branch",
        json={
            "text": "Explain why apples can look blue",
            "mode": "text",
            "input_artifact_ids": [],
            "settings": {},
        },
    )
    assert branched.status_code == 202
    run = branched.json()["run"]
    assert run["operation"] == "text"
    assert run["workflow_revision_id"] is None
    assert run["provenance_json"]["input_artifact_ids"] == []
    assert "steps" not in run["settings_json"]


async def test_regenerating_older_response_preserves_later_history_and_revisions(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Revision project"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Revision history", "project_id": project["id"]},
        )
    ).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write the first answer", "mode": "text"},
    )
    assert first.status_code == 202
    await wait_for_run(client, first.json()["run"]["id"])
    first_assistant_id = first.json()["assistant_message"]["id"]

    second = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write the later answer", "mode": "text"},
    )
    assert second.status_code == 202
    await wait_for_run(client, second.json()["run"]["id"])
    later_head = second.json()["assistant_message"]["id"]
    before = (await client.get(f"/api/chats/{chat['id']}")).json()
    visible_before = [
        message["id"] for message in before["messages"] if message["transcript_visible"]
    ]

    regenerated = await client.post(
        f"/api/messages/{first_assistant_id}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    accepted = regenerated.json()
    assert accepted["user_message"]["transcript_visible"] is False
    assert accepted["assistant_message"]["transcript_visible"] is False
    assert (await client.get(f"/api/chats/{chat['id']}")).json()[
        "active_head_message_id"
    ] == later_head
    await wait_for_run(client, accepted["run"]["id"])

    after = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert after["active_head_message_id"] == later_head
    assert [
        message["id"] for message in after["messages"] if message["transcript_visible"]
    ] == visible_before
    original = next(message for message in after["messages"] if message["id"] == first_assistant_id)
    completed_revisions = [
        revision for revision in original["response_revisions"] if revision["status"] == "complete"
    ]
    assert len(completed_revisions) == 2
    assert original["active_response_revision_id"] == completed_revisions[-1]["id"]

    latest_export = (await client.post(f"/api/projects/{project['id']}/export")).json()
    latest_archive = await client.get(latest_export["url"])
    latest_import = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "latest-revision-project.lm-atelier.zip",
                latest_archive.content,
                "application/zip",
            )
        },
    )
    assert latest_import.status_code == 201, latest_import.text
    latest_imported_chat = (
        await client.get("/api/chats", params={"project_id": latest_import.json()["id"]})
    ).json()[0]
    latest_imported_detail = (await client.get(f"/api/chats/{latest_imported_chat['id']}")).json()
    latest_imported_first = next(
        message
        for message in latest_imported_detail["messages"]
        if message["role"] == "assistant" and message["transcript_visible"]
    )
    assert latest_imported_first["active_response_revision_id"] == next(
        revision["id"]
        for revision in latest_imported_first["response_revisions"]
        if revision["sequence"] == 2
    )

    selected = await client.post(
        f"/api/messages/{first_assistant_id}/revisions/{completed_revisions[0]['id']}/select"
    )
    assert selected.status_code == 200
    assert selected.json()["active_response_revision_id"] == completed_revisions[0]["id"]
    assert (await client.get(f"/api/chats/{chat['id']}")).json()[
        "active_head_message_id"
    ] == later_head

    exported = (await client.post(f"/api/projects/{project['id']}/export")).json()
    archive = await client.get(exported["url"])
    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "revision-project.lm-atelier.zip",
                archive.content,
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    imported_chat = (
        await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    ).json()[0]
    imported_detail = (await client.get(f"/api/chats/{imported_chat['id']}")).json()
    imported_first = next(
        message
        for message in imported_detail["messages"]
        if message["role"] == "assistant" and message["transcript_visible"]
    )
    assert len(imported_first["response_revisions"]) == 2
    assert imported_first["active_response_revision_id"] == next(
        revision["id"]
        for revision in imported_first["response_revisions"]
        if revision["sequence"] == 1
    )


async def test_failed_regeneration_keeps_the_selected_response(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Failed revision"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write a durable answer", "mode": "text"},
    )
    assert original.status_code == 202
    await wait_for_run(client, original.json()["run"]["id"])
    message_id = original.json()["assistant_message"]["id"]
    before = (await client.get(f"/api/messages/{message_id}")).json()
    selected_before = before["active_response_revision_id"]
    parts_before = [(part["type"], part["text"], part["artifact_id"]) for part in before["parts"]]

    async def fail_after_delta(
        _adapter: MockChatAdapter,
        _request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="delta", text="Replacement that must not be selected")
        yield ChatEvent(type="error", data={"error": ""})

    monkeypatch.setattr(MockChatAdapter, "stream", fail_after_delta)
    regenerated = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    run_id = regenerated.json()["run"]["id"]
    deadline = asyncio.get_running_loop().time() + 5
    run = regenerated.json()["run"]
    while asyncio.get_running_loop().time() < deadline:
        run = (await client.get(f"/api/runs/{run_id}")).json()
        if run["status"] == "failed":
            break
        await asyncio.sleep(0.03)
    assert run["status"] == "failed"

    after = (await client.get(f"/api/messages/{message_id}")).json()
    assert after["active_response_revision_id"] == selected_before
    assert [
        (part["type"], part["text"], part["artifact_id"]) for part in after["parts"]
    ] == parts_before
    assert [revision["status"] for revision in after["response_revisions"]] == [
        "complete",
        "failed",
    ]


async def test_cancelled_regeneration_keeps_the_selected_response(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Cancelled revision"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write a durable answer", "mode": "text"},
    )
    assert original.status_code == 202
    await wait_for_run(client, original.json()["run"]["id"])
    message_id = original.json()["assistant_message"]["id"]
    before = (await client.get(f"/api/messages/{message_id}")).json()
    selected_before = before["active_response_revision_id"]
    parts_before = [(part["type"], part["text"], part["artifact_id"]) for part in before["parts"]]

    regenerated = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    after = (await client.get(f"/api/messages/{message_id}")).json()
    assert after["active_response_revision_id"] == selected_before
    assert [
        (part["type"], part["text"], part["artifact_id"]) for part in after["parts"]
    ] == parts_before
    assert [revision["status"] for revision in after["response_revisions"]] == [
        "complete",
        "cancelled",
    ]


async def test_regeneration_reuses_the_selected_revision_contract(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Selected contract"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Write a stable answer",
            "mode": "text",
            "settings": {"temperature": 0.1},
        },
    )
    assert original.status_code == 202
    await wait_for_run(client, original.json()["run"]["id"])
    message_id = original.json()["assistant_message"]["id"]

    first_replacement = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {"temperature": 0.25}},
    )
    assert first_replacement.status_code == 202
    await wait_for_run(client, first_replacement.json()["run"]["id"])

    second_replacement = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {}},
    )
    assert second_replacement.status_code == 202
    assert second_replacement.json()["run"]["settings_json"]["temperature"] == 0.25
    duplicate = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {}},
    )
    assert duplicate.status_code == 409
    assert "already being regenerated" in duplicate.json()["detail"]
    await wait_for_run(client, second_replacement.json()["run"]["id"])


async def test_new_turn_uses_persisted_active_branch_head(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Branch head"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "First", "mode": "text"},
    )
    first_run = await wait_for_run(client, first.json()["run"]["id"])
    first_assistant_id = first_run["assistant_message_id"]

    refreshed = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert refreshed["active_head_message_id"] == first_assistant_id

    second = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Second", "mode": "text"},
    )
    assert second.status_code == 202
    assert second.json()["user_message"]["parent_id"] == first_assistant_id
    assert (await client.get(f"/api/chats/{chat['id']}")).json()[
        "active_head_message_id"
    ] == second.json()["assistant_message"]["id"]


async def test_long_chat_records_visible_context_truncation(client: AsyncClient) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    profile = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{profile['id']}",
        json={
            "load_settings": {"context_length": 512},
            "request_settings": {"max_tokens": 64},
        },
    )
    assert updated.status_code == 200
    chat = (await client.post("/api/chats", json={"title": "Long context"})).json()

    final_run: dict = {}
    for index in range(6):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": f"Turn {index}: " + ("context detail " * 12),
                "mode": "text",
            },
        )
        assert response.status_code == 202
        final_run = await wait_for_run(client, response.json()["run"]["id"])

    context = final_run["provenance_json"]["context"]
    assert context["messages_omitted"] > 0
    assert context["input_tokens"] <= context["input_budget"]
    assert final_run["settings_json"]["context_length"] == 512
    assert final_run["provenance_json"]["resolved_settings"]["context_length"] == 512

    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assistant = [item for item in detail["messages"] if item["role"] == "assistant"][-1]
    metadata = next(part for part in assistant["parts"] if part["type"] == "generation_metadata")
    assert metadata["metadata_json"]["context"]["messages_omitted"] > 0


async def test_turn_rejects_load_only_setting_overrides(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Load settings"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Do not restart the worker",
            "mode": "text",
            "settings": {"context_length": 512},
        },
    )
    assert response.status_code == 422
    assert "unsupported settings: context_length" in response.json()["detail"]


async def test_artifact_upload_deduplicates_content(client: AsyncClient) -> None:
    first = await client.post(
        "/api/artifacts",
        files={"file": ("one.png", b"same-bytes", "image/png")},
    )
    second = await client.post(
        "/api/artifacts",
        files={"file": ("two.png", b"same-bytes", "image/png")},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]


async def test_media_library_reports_references_and_storage(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Gallery"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image for the gallery", "mode": "image"},
    )
    message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in message["parts"] if part["type"] == "image")

    gallery = await client.get("/api/artifacts", params={"kind": "image"})
    assert gallery.status_code == 200
    item = next(artifact for artifact in gallery.json() if artifact["id"] == image["artifact_id"])
    assert item["reference_count"] == 1
    assert item["chat_ids"] == [chat["id"]]

    storage = await client.get("/api/artifacts/storage")
    assert storage.status_code == 200
    assert storage.json()["referenced_count"] >= 1
    assert storage.json()["retention_pending_count"] == 0
    assert storage.json()["retention_days"] == 30
    assert turn.status_code == 202


async def test_text_turn_after_image_keeps_alternating_chat_context(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Mixed media context"})).json()
    await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a reference image", "mode": "image"},
    )
    await wait_for_assistant(client, chat["id"], "image")

    with SessionLocal() as session:
        persisted_chat = session.get(Chat, chat["id"])
        assert persisted_chat
        routing_context = ConversationOrchestrator._routing_context(
            session,
            persisted_chat,
            persisted_chat.active_head_message_id,
        )
    assert routing_context == [
        {"role": "user", "content": "Create a reference image"},
        {
            "role": "assistant",
            "content": (
                "Generated image requested with this prompt "
                '(visual contents not inspected): "Create a reference image".'
            ),
        },
    ]

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Now answer a text question", "mode": "text"},
    )
    assert accepted.status_code == 202
    await wait_for_assistant(client, chat["id"], "text")

    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run
        generation_context = ConversationOrchestrator._context_messages(session, run)
    assert generation_context == [
        {"role": "user", "content": "Create a reference image"},
        {
            "role": "assistant",
            "content": (
                "Generated image requested with this prompt "
                '(visual contents not inspected): "Create a reference image".'
            ),
        },
        {"role": "user", "content": "Now answer a text question"},
    ]


async def test_vision_chat_receives_verified_local_image_bytes(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_capabilities = MockChatAdapter.capabilities
    captured: list[ChatRequest] = []

    async def vision_capabilities(adapter: MockChatAdapter) -> EngineCapabilities:
        capabilities = await original_capabilities(adapter)
        return capabilities.model_copy(update={"input_modalities": ["text", "image"]})

    async def capture_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        captured.append(request)
        yield ChatEvent(type="delta", text="I inspected the local image.")
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    monkeypatch.setattr(MockChatAdapter, "capabilities", vision_capabilities)
    monkeypatch.setattr(MockChatAdapter, "stream", capture_stream)
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("pixel.png", ONE_PIXEL_PNG, "image/png")},
    )
    assert uploaded.status_code == 201
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Vision context"})).json()

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "What does this image show?",
            "mode": "text",
            "input_artifact_ids": [artifact_id],
        },
    )
    assert accepted.status_code == 202
    await wait_for_assistant(client, chat["id"], "text")

    assert len(captured) == 1
    content = captured[0].messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "text",
        "text": "What does this image show?\n[Attached image: pixel.png]",
    }
    data_url = content[1]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)
    assert base64.b64decode(data_url.removeprefix(prefix)) == ONE_PIXEL_PNG
    run = (await client.get(f"/api/runs/{accepted.json()['run']['id']}")).json()
    vision = run["provenance_json"]["context"]["vision"]
    assert vision["available"] is True
    assert vision["mode"] == "direct"
    assert vision["visual_contents_inspected"] is True
    assert vision["images_included"] == 1
    assert vision["artifact_ids"] == [artifact_id]
    assert vision["bytes_included"] == len(ONE_PIXEL_PNG)
    assert vision["images_skipped"] == 0
    assert vision["artifact_hashes"] == {artifact_id: artifact_id.removeprefix("sha256:")}


async def test_text_only_chat_never_receives_attached_image_bytes(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: list[ChatRequest] = []

    async def capture_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        captured.append(request)
        yield ChatEvent(type="delta", text="I only received textual context.")
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    monkeypatch.setattr(MockChatAdapter, "stream", capture_stream)
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("pixel.png", ONE_PIXEL_PNG, "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Text fallback"})).json()

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Describe this attachment",
            "mode": "text",
            "input_artifact_ids": [artifact_id],
        },
    )
    assert accepted.status_code == 202
    await wait_for_assistant(client, chat["id"], "text")

    assert len(captured) == 1
    content = captured[0].messages[-1]["content"]
    assert isinstance(content, str)
    assert "Attached image: pixel.png" in content
    run = (await client.get(f"/api/runs/{accepted.json()['run']['id']}")).json()
    assert run["provenance_json"]["context"]["vision"] == {
        "available": False,
        "mode": "none",
        "visual_contents_inspected": False,
        "images_included": 0,
        "artifact_ids": [],
        "images_skipped": 1,
        "reason": "No runtime-verified vision profile is available.",
    }


async def test_vision_chat_reuses_the_newest_prior_generated_image(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_capabilities = MockChatAdapter.capabilities
    captured: list[ChatRequest] = []

    async def vision_capabilities(adapter: MockChatAdapter) -> EngineCapabilities:
        capabilities = await original_capabilities(adapter)
        return capabilities.model_copy(update={"input_modalities": ["text", "image"]})

    async def capture_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        captured.append(request)
        yield ChatEvent(type="delta", text="Done.")
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    monkeypatch.setattr(MockChatAdapter, "capabilities", vision_capabilities)
    monkeypatch.setattr(MockChatAdapter, "stream", capture_stream)
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("generated.png", ONE_PIXEL_PNG, "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Prior visual"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Start the scene", "mode": "text"},
    )
    await wait_for_assistant(client, chat["id"], "text")
    with SessionLocal() as session:
        assistant = session.get(Message, first.json()["assistant_message"]["id"])
        assert assistant
        assistant.parts.append(
            MessagePart(
                position=max(part.position for part in assistant.parts) + 1,
                type="image",
                artifact_id=artifact_id,
            )
        )
        session.commit()
    captured.clear()

    followup = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "What changed in the image?", "mode": "text"},
    )
    assert followup.status_code == 202
    await wait_for_assistant(client, chat["id"], "text")

    assert len(captured) == 1
    content = captured[0].messages[-1]["content"]
    assert isinstance(content, list)
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    run = (await client.get(f"/api/runs/{followup.json()['run']['id']}")).json()
    assert run["provenance_json"]["context"]["vision"]["artifact_ids"] == [artifact_id]


async def test_vision_chat_uses_the_poster_for_a_prior_generated_video(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_capabilities = MockChatAdapter.capabilities
    captured: list[ChatRequest] = []

    async def vision_capabilities(adapter: MockChatAdapter) -> EngineCapabilities:
        capabilities = await original_capabilities(adapter)
        return capabilities.model_copy(update={"input_modalities": ["text", "image"]})

    async def capture_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        captured.append(request)
        yield ChatEvent(type="delta", text="Done.")
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    monkeypatch.setattr(MockChatAdapter, "capabilities", vision_capabilities)
    monkeypatch.setattr(MockChatAdapter, "stream", capture_stream)
    poster_response = await client.post(
        "/api/artifacts",
        files={"file": ("poster.png", ONE_PIXEL_PNG, "image/png")},
    )
    video_response = await client.post(
        "/api/artifacts",
        files={
            "file": (
                "clip.mp4",
                b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2",
                "video/mp4",
            )
        },
    )
    poster_id = poster_response.json()["id"]
    video_id = video_response.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Video poster context"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Start the clip", "mode": "text"},
    )
    await wait_for_assistant(client, chat["id"], "text")
    with SessionLocal() as session:
        video = session.get(Artifact, video_id)
        assistant = session.get(Message, first.json()["assistant_message"]["id"])
        assert video and assistant
        video.metadata_json = {**video.metadata_json, "poster_artifact_id": poster_id}
        assistant.parts.append(
            MessagePart(
                position=max(part.position for part in assistant.parts) + 1,
                type="video",
                artifact_id=video_id,
            )
        )
        session.commit()
    captured.clear()

    followup = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Describe the clip", "mode": "text"},
    )
    assert followup.status_code == 202
    await wait_for_assistant(client, chat["id"], "text")

    assert len(captured) == 1
    content = captured[0].messages[-1]["content"]
    assert isinstance(content, list)
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    run = (await client.get(f"/api/runs/{followup.json()['run']['id']}")).json()
    assert run["provenance_json"]["context"]["vision"]["artifact_ids"] == [poster_id]


async def test_image_request_uses_referenced_text_from_the_active_chat_branch(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Story illustration"})).json()
    story_request = "Write a short story about a silver fox crossing a glass city at dusk."
    text_turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": story_request, "mode": "text"},
    )
    assert text_turn.status_code == 202
    story_message = await wait_for_assistant(client, chat["id"], "text")
    story_text = "\n".join(
        part["text"] for part in story_message["parts"] if part["type"] == "text"
    ).strip()

    image_turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make an image based on the previous story", "mode": "auto"},
    )
    assert image_turn.status_code == 202
    image_run = image_turn.json()["run"]
    assert image_run["operation"] == "text_to_image"
    assert "Source chat text:" in image_run["standalone_prompt"]
    assert story_request in image_run["standalone_prompt"]
    assert story_text in image_run["standalone_prompt"]

    image_message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in image_message["parts"] if part["type"] == "image")
    assert (
        image["artifact"]["metadata_json"]["semantic_description"] == image_run["standalone_prompt"]
    )
    assert image["artifact"]["metadata_json"]["semantic_description_source"] == (
        "generation_prompt"
    )
    assert image["artifact"]["metadata_json"]["semantic_description_confidence"] == "intent-only"
    assert image["artifact"]["metadata_json"]["visual_contents_inspected"] is False


async def test_prior_image_edit_falls_back_to_accumulated_text_prompt(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        for workflow in session.query(WorkflowDefinition).filter_by(operation="image_to_image"):
            session.delete(workflow)
        session.commit()

    chat = (await client.post("/api/chats", json={"title": "Semantic image edit"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make me an image of an apple", "mode": "auto"},
    )
    assert first.status_code == 202
    await wait_for_assistant(client, chat["id"], "image")

    edited = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make it green", "mode": "auto"},
    )
    assert edited.status_code == 202
    edited_run = await wait_for_run(client, edited.json()["run"]["id"])

    assert edited_run["operation"] == "text_to_image"
    assert edited_run["standalone_prompt"] == (
        "Make me an image of an apple. Follow-up instruction: Make it green"
    )
    edited_message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in edited_message["parts"] if part["type"] == "image")
    assert image["artifact"]["metadata_json"]["semantic_description"] == (
        "Make me an image of an apple. Follow-up instruction: Make it green"
    )


async def test_retention_cleanup_only_removes_expired_unreferenced_artifacts(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.bin", b"expired-orphan", "application/octet-stream")},
    )
    artifact_id = uploaded.json()["id"]
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact
        artifact.metadata_json = {
            **artifact.metadata_json,
            "unreferenced_at": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
        }
        session.commit()

    preview = await client.post("/api/artifacts/cleanup", json={"dry_run": True})
    assert preview.status_code == 200
    assert preview.json()["removed_count"] == 1
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200

    cleaned = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleaned.status_code == 200
    assert cleaned.json()["removed_count"] == 1
    assert cleaned.json()["reclaimed_bytes"] == len(b"expired-orphan")
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 404


async def test_cleanup_marks_new_unreferenced_artifacts_for_recovery(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.png", b"new-orphan", "image/png")},
    )
    artifact_id = uploaded.json()["id"]

    cleaned = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleaned.status_code == 200
    assert cleaned.json()["marked_count"] == 1
    assert cleaned.json()["removed_count"] == 0

    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact
        assert isinstance(artifact.metadata_json.get("unreferenced_at"), str)


async def test_retention_keeps_artifacts_referenced_only_by_response_revisions(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Retained revision"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a revision owner", "mode": "text"},
    )
    assert turn.status_code == 202
    await wait_for_run(client, turn.json()["run"]["id"])
    message = (await client.get(f"/api/messages/{turn.json()['assistant_message']['id']}")).json()
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("retained.png", ONE_PIXEL_PNG, "image/png")},
    )
    artifact_id = uploaded.json()["id"]

    with SessionLocal() as session:
        revision = session.get(ResponseRevision, message["active_response_revision_id"])
        artifact = session.get(Artifact, artifact_id)
        assert revision and artifact
        revision.parts.append(
            ResponseRevisionPart(
                position=len(revision.parts),
                type="image",
                artifact_id=artifact_id,
            )
        )
        artifact.metadata_json = {
            **artifact.metadata_json,
            "unreferenced_at": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
        }
        session.commit()

    cleaned = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleaned.status_code == 200
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200


async def test_media_library_can_delete_a_referenced_artifact(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Delete media"})).json()
    await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image to delete", "mode": "image"},
    )
    message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in message["parts"] if part["type"] == "image")
    artifact_id = image["artifact_id"]

    deleted = await client.delete(f"/api/artifacts/{artifact_id}")
    assert deleted.status_code == 200
    assert deleted.json()["artifact_id"] == artifact_id
    assert deleted.json()["reference_count"] == 1
    assert deleted.json()["removed_count"] == 1
    assert deleted.json()["reclaimed_bytes"] > 0
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 404

    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    deleted_part = next(
        part for item in detail["messages"] for part in item["parts"] if part["id"] == image["id"]
    )
    assert deleted_part["artifact_id"] is None


async def test_chat_delete_can_remove_exclusive_generated_media(client: AsyncClient) -> None:
    keep_chat = (await client.post("/api/chats", json={"title": "Keep media"})).json()
    await client.post(
        f"/api/chats/{keep_chat['id']}/turns",
        json={"text": "Create an image to keep", "mode": "image"},
    )
    keep_message = await wait_for_assistant(client, keep_chat["id"], "image")
    keep_artifact_id = next(
        part["artifact_id"] for part in keep_message["parts"] if part["type"] == "image"
    )

    assert (await client.delete(f"/api/chats/{keep_chat['id']}")).status_code == 204
    assert (await client.get(f"/api/artifacts/{keep_artifact_id}")).status_code == 200

    delete_chat = (await client.post("/api/chats", json={"title": "Delete media"})).json()
    await client.post(
        f"/api/chats/{delete_chat['id']}/turns",
        json={"text": "Create an image to remove", "mode": "image"},
    )
    delete_message = await wait_for_assistant(client, delete_chat["id"], "image")
    delete_artifact_id = next(
        part["artifact_id"] for part in delete_message["parts"] if part["type"] == "image"
    )

    deleted = await client.delete(
        f"/api/chats/{delete_chat['id']}",
        params={"delete_generated_media": True},
    )
    assert deleted.status_code == 204
    assert (await client.get(f"/api/artifacts/{delete_artifact_id}")).status_code == 404


async def test_chat_delete_keeps_generated_media_referenced_by_another_chat(
    client: AsyncClient,
) -> None:
    source_chat = (await client.post("/api/chats", json={"title": "Source"})).json()
    await client.post(
        f"/api/chats/{source_chat['id']}/turns",
        json={"text": "Create a shared image", "mode": "image"},
    )
    source_message = await wait_for_assistant(client, source_chat["id"], "image")
    artifact_id = next(
        part["artifact_id"] for part in source_message["parts"] if part["type"] == "image"
    )

    other_chat = (await client.post("/api/chats", json={"title": "Other"})).json()
    attached = await client.post(
        f"/api/chats/{other_chat['id']}/turns",
        json={
            "text": "Describe this image",
            "mode": "text",
            "input_artifact_ids": [artifact_id],
        },
    )
    assert attached.status_code == 202

    deleted = await client.delete(
        f"/api/chats/{source_chat['id']}",
        params={"delete_generated_media": True},
    )
    assert deleted.status_code == 204
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200


async def test_chat_delete_cancels_all_queued_runs_and_cleans_up_tasks(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    started: set[str] = set()
    finished: set[str] = set()

    async def remain_queued(
        _orchestrator: ConversationOrchestrator,
        job_id: str,
        _run_id: str,
    ) -> None:
        started.add(job_id)
        try:
            await asyncio.Event().wait()
        finally:
            finished.add(job_id)

    monkeypatch.setattr(ConversationOrchestrator, "_execute", remain_queued)
    chat = (await client.post("/api/chats", json={"title": "Queued deletion"})).json()
    turn_responses = [
        await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": f"Queued request {index}", "mode": "text"},
        )
        for index in range(2)
    ]
    assert all(response.status_code == 202 for response in turn_responses), [
        response.text for response in turn_responses
    ]
    turns = [response.json() for response in turn_responses]
    run_ids = {turn["run"]["id"] for turn in turns}
    orchestrator: ConversationOrchestrator = app.state.services.orchestrator
    deadline = asyncio.get_running_loop().time() + 5
    job_ids: set[str] = set()
    while asyncio.get_running_loop().time() < deadline:
        jobs = (await client.get("/api/jobs")).json()
        job_ids = {job["id"] for job in jobs if job["run_id"] in run_ids}
        if len(job_ids) == 2 and started == job_ids:
            break
        await asyncio.sleep(0.01)
    assert len(job_ids) == 2
    assert started == job_ids
    assert job_ids <= orchestrator._tasks.keys()

    deleted = await client.delete(f"/api/chats/{chat['id']}")

    assert deleted.status_code == 204
    await asyncio.sleep(0)
    assert finished == job_ids
    assert not (job_ids & orchestrator._tasks.keys())
    with SessionLocal() as session:
        jobs = list(session.scalars(select(Job).where(Job.id.in_(job_ids))).all())
        assert len(jobs) == 2
        assert all(job.status == JobStatus.CANCELLED.value for job in jobs)
        assert all(job.run_id is None for job in jobs)
        assert not session.scalars(select(Run).where(Run.chat_id == chat["id"])).all()


async def test_chat_delete_awaits_active_run_cleanup_before_database_deletion(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    stream_started = asyncio.Event()
    cancellation_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def blocked_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="delta", text="Partial response")
        stream_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_started.set()
            await allow_cleanup.wait()
            with SessionLocal() as session:
                run = session.get(Run, request.run_id)
                assert run is not None
                run.error = "task cleanup completed before chat deletion"
                session.commit()
            cleanup_finished.set()
            raise

    monkeypatch.setattr(MockChatAdapter, "stream", blocked_stream)
    chat = (await client.post("/api/chats", json={"title": "Active deletion"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Start a response", "mode": "text"},
    )
    assert turn.status_code == 202
    await asyncio.wait_for(stream_started.wait(), timeout=5)
    jobs = (await client.get("/api/jobs")).json()
    job = next(item for item in jobs if item["run_id"] == turn.json()["run"]["id"])
    assert job["status"] == JobStatus.RUNNING.value

    deletion = asyncio.create_task(client.delete(f"/api/chats/{chat['id']}"))
    await asyncio.wait_for(cancellation_started.wait(), timeout=5)
    await asyncio.sleep(0)
    assert not deletion.done()
    with SessionLocal() as session:
        assert session.get(Chat, chat["id"]) is not None
        assert session.get(Run, turn.json()["run"]["id"]) is not None

    allow_cleanup.set()
    deleted = await asyncio.wait_for(deletion, timeout=5)

    assert deleted.status_code == 204
    assert cleanup_finished.is_set()
    orchestrator: ConversationOrchestrator = app.state.services.orchestrator
    await asyncio.sleep(0)
    assert job["id"] not in orchestrator._tasks
    with SessionLocal() as session:
        remaining_job = session.get(Job, job["id"])
        assert remaining_job is not None
        assert remaining_job.status == JobStatus.CANCELLED.value
        assert remaining_job.run_id is None
        assert session.get(Run, turn.json()["run"]["id"]) is None
        assert session.get(Chat, chat["id"]) is None


async def test_workflow_revisions_and_validation(client: AsyncClient) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Custom",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "default": 20}},
            },
            "trusted": True,
        },
    )
    assert created.status_code == 201
    workflow = created.json()
    revision = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={"api_graph": {"node": {"class_type": "MockV2"}}, "trusted": True},
    )
    assert revision.status_code == 201
    assert revision.json()["version"] == 2
    validation = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    exported = await client.get(f"/api/workflows/{workflow['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-workflow"
    assert bundle["source_revision"] == 2

    cloned = await client.post(
        f"/api/workflows/{workflow['id']}/clone", json={"name": "Custom copy"}
    )
    assert cloned.status_code == 201
    assert cloned.json()["name"] == "Custom copy"
    assert cloned.json()["revisions"][0]["trusted"] is True

    bundle["name"] = "Imported workflow"
    bundle["trusted"] = True
    imported = await client.post("/api/workflows/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["revisions"][0]["api_graph_json"] == {"node": {"class_type": "MockV2"}}
    assert imported.json()["revisions"][0]["trusted"] is False

    restored = await client.post(
        f"/api/workflows/{workflow['id']}/revisions/{workflow['revisions'][0]['id']}/restore"
    )
    assert restored.status_code == 201
    assert restored.json()["version"] == 3
    assert restored.json()["api_graph_json"] == {"node": {"class_type": "Mock"}}

    missing_dependency = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "api_graph": {"node": {"class_type": "Mock"}},
            "dependencies": {"models": ["not-installed"]},
            "trusted": True,
        },
    )
    assert missing_dependency.status_code == 201
    validation = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert validation.json()["valid"] is False
    assert "missing model dependency" in validation.json()["errors"][0]


async def test_workflow_vram_requirement_uses_device_capacity(
    client: AsyncClient, monkeypatch
) -> None:
    memory = {
        "total": 16 * 1024**3,
        "available": 8 * 1024**3,
    }

    def system_info(_settings):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            devices=[
                SimpleNamespace(
                    kind="gpu",
                    total_memory_bytes=memory["total"],
                    available_memory_bytes=memory["available"],
                )
            ]
        )

    monkeypatch.setattr("local_lm.api.collect_system_info", system_info)
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "VRAM capacity contract",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": "Mock"}},
                "dependencies": {"minimum_vram_bytes": 12 * 1024**3},
                "trusted": True,
            },
        )
    ).json()

    under_pressure = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert under_pressure.status_code == 200
    assert under_pressure.json()["valid"] is True
    assert under_pressure.json()["errors"] == []
    assert under_pressure.json()["warnings"] == [
        "currently available accelerator memory is below the workflow requirement"
    ]

    memory["total"] = 8 * 1024**3
    insufficient = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert insufficient.status_code == 200
    assert insufficient.json()["valid"] is False
    assert insufficient.json()["warnings"] == []
    assert insufficient.json()["errors"] == [
        "accelerator memory capacity is below the workflow requirement"
    ]


async def test_workflow_validation_requires_trust_and_active_model_dependencies(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    inactive_path = tmp_path / "inactive-model.safetensors"
    inactive_path.write_bytes(b"inactive")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_inactive_workflow_dependency",
                name="Inactive workflow dependency",
                role="image",
                engine="comfyui",
                local_path=str(inactive_path),
                size_bytes=inactive_path.stat().st_size,
                compatibility="likely",
                active=False,
            )
        )
        session.commit()

    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Untrusted inactive dependency",
                "operation": "text_to_image",
                "engine": "comfyui",
                "api_graph": {"1": {"class_type": "SaveImage", "inputs": {}}},
                "dependencies": {"models": [{"id": "model_inactive_workflow_dependency"}]},
                "trusted": False,
            },
        )
    ).json()

    validation = await client.post(f"/api/workflows/{workflow['id']}/validate")

    assert validation.status_code == 200
    assert validation.json()["valid"] is False
    assert any("not trusted" in error for error in validation.json()["errors"])
    assert any("missing model dependency" in error for error in validation.json()["errors"])


async def test_project_pins_an_immutable_media_workflow_revision(client: AsyncClient) -> None:
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Project image recipe",
                "operation": "text_to_image",
                "api_graph": {"1": {"class_type": "SaveImage", "inputs": {}}},
                "trusted": True,
            },
        )
    ).json()
    revision_id = workflow["current_revision_id"]
    project_response = await client.post(
        "/api/projects",
        json={"name": "Pinned studio", "image_workflow_revision_id": revision_id},
    )
    assert project_response.status_code == 201
    project = project_response.json()
    assert project["image_workflow_revision_id"] == revision_id

    incompatible = await client.patch(
        f"/api/projects/{project['id']}",
        json={"video_workflow_revision_id": revision_id},
    )
    assert incompatible.status_code == 422

    chat = (
        await client.post("/api/chats", json={"title": "Pinned run", "project_id": project["id"]})
    ).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a blue cabin", "mode": "image"},
    )
    assert turn.status_code == 202
    assert turn.json()["run"]["workflow_revision_id"] == revision_id


async def test_media_workflow_follows_the_selected_model_dependency(
    client: AsyncClient, tmp_path: Path
) -> None:
    installs = []
    for name in ("first-image-model", "second-image-model"):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / f"{name}.safetensors").write_bytes(b"safe")
        installs.append(
            (
                await client.post(
                    "/api/models/import",
                    json={
                        "name": name,
                        "role": "image",
                        "engine": "mock",
                        "local_path": str(model_dir),
                    },
                )
            ).json()
        )
    profiles = (await client.get("/api/profiles")).json()
    selected_profile = next(
        profile for profile in profiles if profile["model_install_id"] == installs[0]["id"]
    )
    revisions = []
    for index, install in enumerate(installs, start=1):
        workflow = (
            await client.post(
                "/api/workflows",
                json={
                    "name": f"Model-specific image workflow {index}",
                    "operation": "text_to_image",
                    "engine": "mock",
                    "api_graph": {"node": {"class_type": f"MockImage{index}"}},
                    "dependencies": {"model_install_ids": [install["id"]]},
                    "trusted": True,
                },
            )
        ).json()
        revisions.append(workflow["current_revision_id"])

    chat = (await client.post("/api/chats", json={"title": "Dependency routing"})).json()
    updated = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": selected_profile["id"]},
    )
    assert updated.status_code == 200
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw with the first model", "mode": "image"},
    )

    assert turn.status_code == 202
    assert turn.json()["run"]["workflow_revision_id"] == revisions[0]


async def test_pinned_workflow_schema_drives_generation_settings(client: AsyncClient) -> None:
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Constrained video recipe",
                "operation": "text_to_video",
                "engine": "mock",
                "api_graph": {"node": {"class_type": "Mock"}},
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "const": 832},
                        "height": {"type": "integer", "const": 480},
                        "frames": {"type": "integer", "default": 81, "minimum": 1},
                        "fps": {"type": "number", "default": 16, "minimum": 1},
                        "steps": {
                            "type": "integer",
                            "default": 12,
                            "minimum": 1,
                            "maximum": 30,
                        },
                        "codec": {"type": "string", "const": "h264"},
                        "camera_strength": {
                            "type": "number",
                            "default": 0.5,
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
                "trusted": True,
            },
        )
    ).json()
    project = (
        await client.post(
            "/api/projects",
            json={
                "name": "Constrained video project",
                "video_workflow_revision_id": workflow["current_revision_id"],
            },
        )
    ).json()
    profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Reusable video profile",
                "role": "video",
                "engine": "mock",
                "request_settings": {"width": 768, "steps": 20},
            },
        )
    ).json()
    preset = await client.post(
        "/api/presets",
        json={
            "name": "Reusable video preset",
            "role": "video",
            "settings": {"frames": 49, "codec": "vp9"},
            "is_default": True,
        },
    )
    assert preset.status_code == 201
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Workflow settings", "project_id": project["id"]},
        )
    ).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_video_profile_id": profile["id"]},
    )
    assert selected.status_code == 200

    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a short camera move",
            "mode": "video",
            "settings": {"camera_strength": 0.75},
        },
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["workflow_revision_id"] == workflow["current_revision_id"]
    assert run["settings_json"]["width"] == 832
    assert run["settings_json"]["height"] == 480
    # Compatible stored profile/preset values still apply, while values that
    # conflict with fixed workflow controls fall back to the workflow defaults.
    assert run["settings_json"]["frames"] == 49
    assert run["settings_json"]["fps"] == 16
    assert run["settings_json"]["steps"] == 20
    assert run["settings_json"]["codec"] == "h264"
    assert run["settings_json"]["camera_strength"] == 0.75
    assert run["provenance_json"]["generation_estimate"]["width"] == 832
    assert run["provenance_json"]["generation_estimate"]["height"] == 480

    await wait_for_run(client, run["id"])
    regenerated = await client.post(
        f"/api/messages/{run['assistant_message_id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    assert regenerated.json()["run"]["settings_json"]["camera_strength"] == 0.75

    branched = await client.post(
        f"/api/messages/{run['user_message_id']}/branch",
        json={"text": "Create a different camera move", "settings": {}},
    )
    assert branched.status_code == 202
    assert branched.json()["run"]["settings_json"]["camera_strength"] == 0.75

    incompatible = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Try an unsupported codec",
            "mode": "video",
            "settings": {"codec": "vp9"},
        },
    )
    assert incompatible.status_code == 422
    assert "codec must be one of" in incompatible.json()["detail"]


async def test_adapter_capability_settings_cover_profile_preset_scopes_and_turn_reuse(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    adapter_cache = SettingField(
        key="adapter_cache",
        label="Adapter cache",
        type="boolean",
        default=False,
        scope="load",
        restart_required=True,
    )
    adapter_strength = SettingField(
        key="adapter_strength",
        label="Adapter strength",
        type="number",
        default=0.5,
        minimum=0,
        maximum=1,
        scope="request",
    )
    adapter = app.state.services.engines.chat
    original_capabilities = adapter.capabilities

    async def dynamic_capabilities() -> EngineCapabilities:
        capabilities = await original_capabilities()
        return extend_capability_role(
            capabilities,
            "chat",
            adapter_cache,
            adapter_strength,
        )

    monkeypatch.setattr(adapter, "capabilities", dynamic_capabilities)
    engines = (await client.get("/api/engines")).json()
    chat_schema = next(item for item in engines if "chat" in item["roles"])["settings_by_role"][
        "chat"
    ]
    assert {"context_length", "adapter_cache", "adapter_strength"} <= {
        field["key"] for field in chat_schema
    }

    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Dynamic chat",
            "role": "chat",
            "engine": "mock",
            "load_settings": {"adapter_cache": True},
            "request_settings": {"adapter_strength": 0.2},
            "is_default": True,
        },
    )
    assert profile.status_code == 201
    assert profile.json()["load_settings_json"]["adapter_cache"] is True

    preset = await client.post(
        "/api/presets",
        json={
            "name": "Dynamic preset",
            "role": "chat",
            "settings": {"adapter_strength": 0.3},
            "is_default": True,
        },
    )
    assert preset.status_code == 201
    project = await client.post(
        "/api/projects",
        json={
            "name": "Dynamic defaults",
            "generation_settings_json": {"chat": {"adapter_strength": 0.4}},
        },
    )
    assert project.status_code == 201
    chat = await client.post(
        "/api/chats",
        json={
            "title": "Dynamic turn",
            "project_id": project.json()["id"],
            "generation_settings_json": {"chat": {"adapter_strength": 0.6}},
        },
    )
    assert chat.status_code == 201

    turn = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={
            "text": "Use the adapter setting",
            "mode": "text",
            "settings": {"adapter_strength": 0.8},
        },
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["settings_json"]["adapter_cache"] is True
    assert run["settings_json"]["adapter_strength"] == 0.8

    await wait_for_run(client, run["id"])
    regenerated = await client.post(
        f"/api/messages/{run['assistant_message_id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    assert regenerated.json()["run"]["settings_json"]["adapter_strength"] == 0.8
    branched = await client.post(
        f"/api/messages/{run['user_message_id']}/branch",
        json={"text": "Use it on this branch", "settings": {}},
    )
    assert branched.status_code == 202
    assert branched.json()["run"]["settings_json"]["adapter_strength"] == 0.8

    invalid = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={
            "text": "Do not weaken validation",
            "mode": "text",
            "settings": {"adapter_strength": 2},
        },
    )
    assert invalid.status_code == 422
    assert "adapter_strength must be at most 1.0" in invalid.json()["detail"]


async def test_media_capability_settings_stay_isolated_by_role(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    image_detail = SettingField(
        key="image_detail",
        label="Image detail",
        type="integer",
        default=2,
        minimum=1,
        maximum=4,
        scope="workflow",
    )
    temporal_detail = SettingField(
        key="temporal_detail",
        label="Temporal detail",
        type="integer",
        default=3,
        minimum=1,
        maximum=8,
        scope="workflow",
    )
    adapter = app.state.services.engines.media
    original_capabilities = adapter.capabilities

    async def dynamic_capabilities() -> EngineCapabilities:
        capabilities = await original_capabilities()
        capabilities = extend_capability_role(capabilities, "image", image_detail)
        return extend_capability_role(capabilities, "video", temporal_detail)

    monkeypatch.setattr(adapter, "capabilities", dynamic_capabilities)
    image_profile = await client.post(
        "/api/profiles",
        json={
            "name": "Image schema",
            "role": "image",
            "engine": "mock",
            "request_settings": {"image_detail": 3},
        },
    )
    assert image_profile.status_code == 201
    video_profile = await client.post(
        "/api/profiles",
        json={
            "name": "Video schema",
            "role": "video",
            "engine": "mock",
            "request_settings": {"temporal_detail": 5},
        },
    )
    assert video_profile.status_code == 201

    wrong_image = await client.post(
        "/api/presets",
        json={
            "name": "Wrong image field",
            "role": "image",
            "settings": {"temporal_detail": 2},
        },
    )
    wrong_video = await client.post(
        "/api/presets",
        json={
            "name": "Wrong video field",
            "role": "video",
            "settings": {"image_detail": 2},
        },
    )
    assert wrong_image.status_code == 422
    assert "unsupported settings: temporal_detail" in wrong_image.json()["detail"]
    assert wrong_video.status_code == 422
    assert "unsupported settings: image_detail" in wrong_video.json()["detail"]


async def test_pinned_workflow_revision_keeps_dynamic_capability_constraints(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    adapter_strength = SettingField(
        key="adapter_strength",
        label="Adapter strength",
        type="number",
        default=0.5,
        minimum=0,
        maximum=1,
        scope="workflow",
    )
    adapter = app.state.services.engines.media
    original_capabilities = adapter.capabilities

    async def dynamic_capabilities() -> EngineCapabilities:
        return extend_capability_role(
            await original_capabilities(),
            "image",
            adapter_strength,
        )

    monkeypatch.setattr(adapter, "capabilities", dynamic_capabilities)
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Dynamic image recipe",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": "Mock"}},
                "input_schema": {
                    "properties": {
                        "adapter_strength": {
                            "type": "number",
                            "default": 0.5,
                            "minimum": 0,
                            "maximum": 0.8,
                        }
                    }
                },
                "trusted": True,
            },
        )
    ).json()
    pinned_revision = workflow["current_revision_id"]
    project = (
        await client.post(
            "/api/projects",
            json={
                "name": "Pinned dynamic schema",
                "image_workflow_revision_id": pinned_revision,
            },
        )
    ).json()
    replacement = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "engine_version": "2",
            "api_graph": {"node": {"class_type": "MockV2"}},
            "input_schema": {
                "properties": {
                    "adapter_strength": {
                        "type": "number",
                        "default": 0.3,
                        "minimum": 0,
                        "maximum": 0.4,
                    }
                }
            },
            "trusted": True,
        },
    )
    assert replacement.status_code == 201
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Pinned dynamic turn", "project_id": project["id"]},
        )
    ).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image with the pinned schema",
            "mode": "image",
            "settings": {"adapter_strength": 0.7},
        },
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["workflow_revision_id"] == pinned_revision
    assert run["settings_json"]["adapter_strength"] == 0.7

    await wait_for_run(client, run["id"])
    regenerated = await client.post(
        f"/api/messages/{run['assistant_message_id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    assert regenerated.json()["run"]["workflow_revision_id"] == pinned_revision
    assert regenerated.json()["run"]["settings_json"]["adapter_strength"] == 0.7


async def test_idempotent_replay_survives_capability_outage_without_new_state(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    chat = (await client.post("/api/chats", json={"title": "Capability replay"})).json()
    request_payload = {
        "text": "Create one durable turn",
        "mode": "text",
        "idempotency_key": "capability-replay",
    }
    first = await client.post(f"/api/chats/{chat['id']}/turns", json=request_payload)
    assert first.status_code == 202
    await wait_for_run(client, first.json()["run"]["id"])

    async def unavailable() -> EngineCapabilities:
        raise RuntimeError("private adapter failure")

    monkeypatch.setattr(app.state.services.engines.chat, "capabilities", unavailable)
    replay = await client.post(f"/api/chats/{chat['id']}/turns", json=request_payload)
    assert replay.status_code == 202
    assert replay.json()["run"]["id"] == first.json()["run"]["id"]

    failed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "This must not create a partial turn",
            "mode": "text",
            "idempotency_key": "capability-outage",
        },
    )
    assert failed.status_code == 503
    assert "private adapter failure" not in failed.text
    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert len([message for message in detail["messages"] if message["role"] == "user"]) == 1


async def test_profile_settings_report_an_inactive_engine(
    client: AsyncClient,
) -> None:
    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Inactive engine",
            "role": "chat",
            "engine": "llama.cpp",
            "request_settings": {"temperature": 0.4},
        },
    )
    assert profile.status_code == 201
    unsupported = await client.patch(
        f"/api/profiles/{profile.json()['id']}",
        json={"request_settings": {"external_only": True}},
    )
    assert unsupported.status_code == 422
    assert "unsupported settings: external_only" in unsupported.json()["detail"]

    chat = (await client.post("/api/chats", json={"title": "Inactive engine"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_chat_profile_id": profile.json()["id"]},
    )
    assert selected.status_code == 200
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Do not dispatch to the wrong engine", "mode": "text"},
    )
    assert turn.status_code == 409
    assert "not configured" in turn.json()["detail"]


async def test_profile_edit_clone_reset_and_portable_bundle(client: AsyncClient) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    source = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{source['id']}",
        json={
            "name": "Focused chat",
            "use_case": "Programming, code review, and technical explanations",
            "load_settings": {"context_length": 16_384},
            "request_settings": {"temperature": 0.25},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["use_case"] == "Programming, code review, and technical explanations"
    assert updated.json()["load_settings_json"] == {"context_length": 16_384}

    cloned = await client.post(
        f"/api/profiles/{source['id']}/clone", json={"name": "Focused chat copy"}
    )
    assert cloned.status_code == 201
    assert cloned.json()["use_case"] == updated.json()["use_case"]
    assert cloned.json()["request_settings_json"] == {"temperature": 0.25}
    assert cloned.json()["is_default"] is False

    exported = await client.get(f"/api/profiles/{source['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-profile"
    assert bundle["version"] == 1
    assert bundle["use_case"] == updated.json()["use_case"]

    bundle["name"] = "Imported portable chat"
    bundle["model_install_id"] = "missing-on-this-machine"
    missing_install = await client.post("/api/profiles/import", json=bundle)
    assert missing_install.status_code == 404

    bundle["model_install_id"] = None
    imported = await client.post("/api/profiles/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["model_install_id"] is None
    assert imported.json()["use_case"] == updated.json()["use_case"]
    assert imported.json()["load_settings_json"] == {"context_length": 16_384}

    reset = await client.post(f"/api/profiles/{source['id']}/reset")
    assert reset.status_code == 200
    assert reset.json()["load_settings_json"] == {}
    assert reset.json()["request_settings_json"] == {}

    invalid = await client.post(
        "/api/profiles/import",
        json={
            "format": "lm-atelier-profile",
            "version": 1,
            "name": "Invalid",
            "role": "chat",
            "engine": "llama.cpp",
            "load_settings": {"not_a_setting": True},
        },
    )
    assert invalid.status_code == 422


async def test_profiles_reject_inactive_missing_and_mismatched_installs(
    client: AsyncClient,
    settings: Settings,
) -> None:
    model_path = settings.model_dir / "profile-integrity.gguf"
    model_path.write_bytes(b"gguf")
    imported_model = await client.post(
        "/api/models/import",
        json={
            "name": "Profile integrity",
            "role": "chat",
            "engine": "llama.cpp",
            "local_path": str(model_path),
        },
    )
    assert imported_model.status_code == 201
    install_id = imported_model.json()["id"]
    bound_profile = next(
        profile
        for profile in (await client.get("/api/profiles?role=chat")).json()
        if profile["model_install_id"] == install_id
    )
    chat = (await client.post("/api/chats", json={"title": "Inactive selection"})).json()

    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        profile = session.get(ModelProfile, bound_profile["id"])
        assert install and profile
        install.active = False
        profile.is_default = True
        session.commit()

    listed = (await client.get("/api/profiles?role=chat")).json()
    assert not any(profile["id"] == bound_profile["id"] for profile in listed)

    inactive_create = await client.post(
        "/api/profiles",
        json={
            "name": "Inactive default",
            "role": "chat",
            "engine": "llama.cpp",
            "model_install_id": install_id,
            "is_default": True,
        },
    )
    assert inactive_create.status_code == 422
    assert "inactive" in inactive_create.json()["detail"]

    inactive_import = await client.post(
        "/api/profiles/import",
        json={
            "format": "lm-atelier-profile",
            "version": 1,
            "name": "Inactive import",
            "role": "chat",
            "engine": "llama.cpp",
            "model_install_id": install_id,
        },
    )
    assert inactive_import.status_code == 422
    assert (
        await client.patch(
            f"/api/profiles/{bound_profile['id']}",
            json={"name": "Still inactive"},
        )
    ).status_code == 422
    assert (await client.post(f"/api/workers/chat/load/{bound_profile['id']}")).status_code == 422
    assert (
        await client.patch(
            f"/api/chats/{chat['id']}",
            json={"active_chat_profile_id": bound_profile["id"]},
        )
    ).status_code == 422

    missing = await client.post(
        "/api/profiles",
        json={
            "name": "Missing",
            "role": "chat",
            "engine": "llama.cpp",
            "model_install_id": "model_missing",
        },
    )
    assert missing.status_code == 404

    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install
        install.active = True
        session.commit()

    wrong_role = await client.post(
        "/api/profiles",
        json={
            "name": "Wrong role",
            "role": "image",
            "engine": "llama.cpp",
            "model_install_id": install_id,
        },
    )
    assert wrong_role.status_code == 422
    assert "role is chat, not image" in wrong_role.json()["detail"]
    wrong_engine = await client.post(
        "/api/profiles",
        json={
            "name": "Wrong engine",
            "role": "chat",
            "engine": "mock",
            "model_install_id": install_id,
        },
    )
    assert wrong_engine.status_code == 422
    assert "engine is llama.cpp, not mock" in wrong_engine.json()["detail"]


async def test_superseded_media_profile_is_retired_from_defaults_and_chats(
    client: AsyncClient,
    settings: Settings,
) -> None:
    old_path = settings.model_dir / "old-media"
    current_path = settings.model_dir / "current-media"
    old_path.mkdir()
    current_path.mkdir()
    with SessionLocal() as session:
        seeded_default = session.scalar(
            select(ModelProfile).where(
                ModelProfile.role == "image",
                ModelProfile.is_default.is_(True),
            )
        )
        assert seeded_default
        seeded_default.is_default = False
        old_install = ModelInstall(
            id="model_old_media",
            name="Old media",
            role="image",
            engine="comfyui",
            local_path=str(old_path),
            manifest_json={"remote_id": "owner/base", "files": ["old.safetensors"]},
            active=True,
        )
        current_install = ModelInstall(
            id="model_current_media",
            name="Current media",
            role="image",
            engine="comfyui",
            local_path=str(current_path),
            manifest_json={
                "remote_id": "owner/variant",
                "source_remote_id": "owner/base",
                "files": ["current.safetensors"],
            },
            active=True,
        )
        old_profile = ModelProfile(
            id="profile_old_media",
            name="Old media",
            role="image",
            engine="comfyui",
            model_install_id=old_install.id,
            is_default=True,
        )
        current_profile = ModelProfile(
            id="profile_current_media",
            name="Current media",
            role="image",
            engine="comfyui",
            model_install_id=current_install.id,
        )
        chat = Chat(
            id="chat_old_media",
            title="Old media selection",
            active_image_profile_id=old_profile.id,
        )
        session.add_all([old_install, current_install, old_profile, current_profile, chat])
        session.flush()

        superseded = DownloadManager._deactivate_superseded_media_installs(
            session,
            current_install,
            "owner/base",
        )
        session.commit()

        assert superseded == [old_install.id]
        assert old_install.active is False
        assert old_profile.is_default is False
        assert seeded_default.is_default is True
        assert chat.active_image_profile_id == "__auto__"

    profiles = (await client.get("/api/profiles?role=image")).json()
    assert not any(profile["id"] == "profile_old_media" for profile in profiles)
    assert any(profile["id"] == "profile_current_media" for profile in profiles)


async def test_auto_model_selection_matches_profile_use_case(client: AsyncClient) -> None:
    programming = await client.post(
        "/api/profiles",
        json={
            "name": "Code specialist",
            "use_case": "Python programming, code review, debugging, and software architecture",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert programming.status_code == 201
    creative = await client.post(
        "/api/profiles",
        json={
            "name": "Creative writer",
            "use_case": "Fiction, poetry, characters, and narrative prose",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert creative.status_code == 201

    chat = (await client.post("/api/chats", json={"title": "Automatic model"})).json()
    assert chat["active_chat_profile_id"] == "__auto__"
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Review and debug this Python code", "mode": "text"},
    )
    assert response.status_code == 202
    selection = response.json()["run"]["provenance_json"]["model_selection"]
    assert selection["mode"] == "auto"
    assert selection["profile_id"] == programming.json()["id"]
    assert {"python", "code"}.issubset(selection["matched_terms"])


async def test_auto_model_selection_normalizes_common_intent_terms(client: AsyncClient) -> None:
    technical = await client.post(
        "/api/profiles",
        json={
            "name": "Technical specialist",
            "use_case": "Software development and application architecture",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert technical.status_code == 201
    visual = await client.post(
        "/api/profiles",
        json={
            "name": "Visual specialist",
            "use_case": "Illustration, photography, and artwork",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert visual.status_code == 201

    chat = (await client.post("/api/chats", json={"title": "Intent aliases"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Help me code a small command-line tool", "mode": "text"},
    )

    assert response.status_code == 202
    selection = response.json()["run"]["provenance_json"]["model_selection"]
    assert selection["profile_id"] == technical.json()["id"]
    assert "code" in selection["matched_terms"]


async def test_preset_lifecycle_and_portable_bundle(client: AsyncClient) -> None:
    created = await client.post(
        "/api/presets",
        json={"name": "Precise", "role": "chat", "settings": {"temperature": 0.1}},
    )
    assert created.status_code == 201
    preset = created.json()

    cloned = await client.post(f"/api/presets/{preset['id']}/clone", json={"name": "Precise copy"})
    assert cloned.status_code == 201
    assert cloned.json()["settings_json"] == {"temperature": 0.1}

    exported = await client.get(f"/api/presets/{preset['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-preset"

    bundle["name"] = "Imported precise"
    imported = await client.post("/api/presets/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["settings_json"] == {"temperature": 0.1}

    reset = await client.post(f"/api/presets/{preset['id']}/reset")
    assert reset.status_code == 200
    assert reset.json()["settings_json"] == {}

    deleted = await client.delete(f"/api/presets/{cloned.json()['id']}")
    assert deleted.status_code == 204


async def test_default_preset_is_resolved_between_profile_and_turn_settings(
    client: AsyncClient,
) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    profile = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{profile['id']}",
        json={"request_settings": {"temperature": 0.25, "max_tokens": 32}},
    )
    assert updated.status_code == 200
    preset = await client.post(
        "/api/presets",
        json={
            "name": "Default precise",
            "role": "chat",
            "settings": {"temperature": 0.1, "max_tokens": 48},
            "is_default": True,
        },
    )
    assert preset.status_code == 201
    chat = (await client.post("/api/chats", json={"title": "Preset resolution"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Apply all setting layers",
            "mode": "text",
            "settings": {"max_tokens": 16},
        },
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["settings_json"]["temperature"] == 0.1
    assert run["settings_json"]["max_tokens"] == 16
    assert run["provenance_json"]["preset"] == {
        "id": preset.json()["id"],
        "name": "Default precise",
        "role": "chat",
        "settings": {"temperature": 0.1, "max_tokens": 48},
    }


async def test_chat_generation_defaults_and_preset_binding_are_persisted(
    client: AsyncClient,
) -> None:
    profile = (await client.get("/api/profiles?role=chat")).json()[0]
    assert (
        await client.patch(
            f"/api/profiles/{profile['id']}",
            json={"request_settings": {"temperature": 0.7, "max_tokens": 700}},
        )
    ).status_code == 200

    presets = {}
    for scope, temperature, max_tokens, is_default in (
        ("Global", 0.6, 600, True),
        ("Project", 0.5, 500, False),
        ("Chat", 0.4, 400, False),
    ):
        response = await client.post(
            "/api/presets",
            json={
                "name": f"{scope} persisted defaults",
                "role": "chat",
                "settings": {"temperature": temperature, "max_tokens": max_tokens},
                "is_default": is_default,
            },
        )
        assert response.status_code == 201
        presets[scope.casefold()] = response.json()

    project_response = await client.post(
        "/api/projects",
        json={
            "name": "Persisted defaults",
            "generation_preset_ids_json": {"chat": presets["project"]["id"]},
            "generation_settings_json": {"chat": {"temperature": 0.35, "max_tokens": 350}},
        },
    )
    assert project_response.status_code == 201
    project = project_response.json()
    chat_response = await client.post(
        "/api/chats",
        json={
            "title": "Scoped defaults",
            "project_id": project["id"],
            "routing_mode": "image",
            "generation_preset_ids_json": {"chat": presets["chat"]["id"]},
            "generation_settings_json": {"chat": {"temperature": 0.25, "max_tokens": 250}},
        },
    )
    assert chat_response.status_code == 201
    chat = chat_response.json()
    assert chat["routing_mode"] == "image"

    updated = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"routing_mode": "text"},
    )
    assert updated.status_code == 200
    persisted = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert persisted["routing_mode"] == "text"
    assert persisted["generation_settings_json"]["chat"]["max_tokens"] == 250
    assert persisted["generation_preset_ids_json"]["chat"] == presets["chat"]["id"]

    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Resolve every settings scope", "settings": {"max_tokens": 150}},
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["settings_json"]["temperature"] == 0.25
    assert run["settings_json"]["max_tokens"] == 150
    assert [item["scope"] for item in run["provenance_json"]["preset_layers"]] == [
        "default",
        "project",
        "chat",
    ]
    assert run["provenance_json"]["preset"]["id"] == presets["chat"]["id"]

    deleted_preset = await client.delete(f"/api/presets/{presets['chat']['id']}")
    assert deleted_preset.status_code == 204
    after_delete = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert "chat" not in after_delete["generation_preset_ids_json"]
    assert after_delete["generation_settings_json"]["chat"] == {
        "temperature": 0.25,
        "max_tokens": 250,
    }

    project_only = await client.patch(
        f"/api/chats/{chat['id']}",
        json={
            "generation_preset_ids_json": {},
            "generation_settings_json": {},
        },
    )
    assert project_only.status_code == 200
    inherited = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Use project defaults"},
    )
    assert inherited.status_code == 202
    assert inherited.json()["run"]["settings_json"]["temperature"] == 0.35
    assert inherited.json()["run"]["settings_json"]["max_tokens"] == 350


async def test_generation_default_contract_rejects_invalid_bindings(
    client: AsyncClient,
) -> None:
    image_preset = (
        await client.post(
            "/api/presets",
            json={"name": "Image only", "role": "image", "settings": {"steps": 12}},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Validated defaults"})).json()

    wrong_role = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"generation_preset_ids_json": {"chat": image_preset["id"]}},
    )
    assert wrong_role.status_code == 422
    load_setting = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"generation_settings_json": {"chat": {"context_length": 16_384}}},
    )
    assert load_setting.status_code == 422
    unknown_role = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"generation_settings_json": {"music": {"temperature": 0.2}}},
    )
    assert unknown_role.status_code == 422


async def test_project_export_snapshots_local_preset_bindings(
    client: AsyncClient,
) -> None:
    preset = (
        await client.post(
            "/api/presets",
            json={
                "name": "Portable project defaults",
                "role": "chat",
                "settings": {"temperature": 0.2, "max_tokens": 222},
            },
        )
    ).json()
    project = (
        await client.post(
            "/api/projects",
            json={
                "name": "Portable settings",
                "generation_preset_ids_json": {"chat": preset["id"]},
                "generation_settings_json": {"chat": {"temperature": 0.1}},
            },
        )
    ).json()
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Portable chat settings",
                "project_id": project["id"],
                "generation_preset_ids_json": {"chat": preset["id"]},
                "generation_settings_json": {"chat": {"max_tokens": 111}},
            },
        )
    ).json()

    exported = (await client.post(f"/api/projects/{project['id']}/export")).json()
    archive = await client.get(exported["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["version"] == 6
    assert manifest["project"]["generation_preset_ids_json"] == {"chat": preset["id"]}
    assert manifest["project"]["generation_settings_json"]["chat"] == {
        "temperature": 0.1,
        "max_tokens": 222,
    }
    exported_chat = next(item for item in manifest["chats"] if item["id"] == chat["id"])
    assert exported_chat["generation_preset_ids_json"] == {"chat": preset["id"]}
    assert exported_chat["generation_settings_json"]["chat"] == {
        "temperature": 0.2,
        "max_tokens": 111,
    }

    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("portable.lm-atelier.zip", archive.content, "application/zip")},
    )
    assert imported.status_code == 201
    imported_chat = (
        await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    ).json()[0]
    imported_detail = (await client.get(f"/api/chats/{imported_chat['id']}")).json()
    assert imported.json()["generation_settings_json"]["chat"]["max_tokens"] == 222
    assert imported_detail["generation_settings_json"]["chat"]["max_tokens"] == 111
    assert imported_detail["generation_preset_ids_json"] == {"chat": preset["id"]}


async def test_chat_preset_rejects_load_only_settings(client: AsyncClient) -> None:
    preset = await client.post(
        "/api/presets",
        json={
            "name": "Invalid load preset",
            "role": "chat",
            "settings": {"context_length": 512},
        },
    )
    assert preset.status_code == 422
    assert "unsupported settings: context_length" in preset.json()["detail"]


async def test_model_storage_cleanup_and_shared_path_deletion(
    client: AsyncClient, settings: Settings
) -> None:
    shared = settings.model_dir / "shared-revision"
    shared.mkdir(parents=True)
    (shared / "model.gguf").write_bytes(b"shared-model")
    payload = {
        "name": "Shared model",
        "role": "chat",
        "engine": "llama.cpp",
        "local_path": str(shared),
    }
    first = await client.post("/api/models/import", json=payload)
    second = await client.post("/api/models/import", json={**payload, "name": "Second selection"})
    assert first.status_code == 201
    assert second.status_code == 201

    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Bound profile",
            "role": "chat",
            "engine": "llama.cpp",
            "model_install_id": first.json()["id"],
        },
    )
    assert profile.status_code == 201
    chat = (await client.post("/api/chats", json={"title": "Cascade model"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_chat_profile_id": profile.json()["id"]},
    )
    assert selected.status_code == 200
    assert (await client.delete(f"/api/models/{first.json()['id']}")).status_code == 409
    assert (
        await client.delete(f"/api/models/{first.json()['id']}", params={"delete_profiles": True})
    ).status_code == 204
    profiles = (await client.get("/api/profiles")).json()
    assert not any(item["model_install_id"] == first.json()["id"] for item in profiles)
    updated_chat = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert updated_chat["active_chat_profile_id"] == "__auto__"
    assert (shared / "model.gguf").is_file()
    assert (
        await client.delete(f"/api/models/{second.json()['id']}", params={"delete_profiles": True})
    ).status_code == 204
    assert not shared.exists()

    partial = settings.download_dir / "orphan.partial"
    partial.mkdir(parents=True)
    (partial / "chunk").write_bytes(b"incomplete")
    storage = await client.get("/api/models/storage")
    assert storage.status_code == 200
    assert storage.json()["partial_download_count"] == 1
    assert storage.json()["partial_download_bytes"] == len(b"incomplete")
    cleaned = await client.post("/api/downloads/cleanup")
    assert cleaned.status_code == 200
    assert cleaned.json() == {"removed_count": 1, "reclaimed_bytes": len(b"incomplete")}
    assert not partial.exists()

    unsafe = settings.data_dir / "unsafe-import"
    unsafe.mkdir()
    (unsafe / "weights.ckpt").write_bytes(b"pickle-compatible")
    blocked = await client.post(
        "/api/models/import",
        json={
            "name": "Unsafe directory",
            "role": "image",
            "engine": "comfyui",
            "local_path": str(unsafe),
        },
    )
    assert blocked.status_code == 422


async def test_model_delete_commit_failure_restores_files_and_database_state(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = settings.model_dir / "commit-rollback"
    managed.mkdir(parents=True)
    (managed / "model.gguf").write_bytes(b"transactional-model")
    model_id = "model_delete_commit_failure"
    create_managed_model(model_id=model_id, path=managed, files=["model.gguf"])
    profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Commit rollback profile",
                "role": "chat",
                "engine": "mock",
                "model_install_id": model_id,
            },
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Commit rollback"})).json()
    assert (
        await client.patch(
            f"/api/chats/{chat['id']}",
            json={"active_chat_profile_id": profile["id"]},
        )
    ).status_code == 200

    def fail_commit(_session: Session) -> None:
        raise RuntimeError("injected model deletion commit failure")

    with (
        monkeypatch.context() as patch,
        pytest.raises(RuntimeError, match="injected model deletion commit failure"),
    ):
        patch.setattr(Session, "commit", fail_commit)
        await client.delete(f"/api/models/{model_id}", params={"delete_profiles": True})

    assert (managed / "model.gguf").read_bytes() == b"transactional-model"
    assert not (settings.model_dir / ".delete-pending").exists()
    with SessionLocal() as session:
        assert session.get(ModelInstall, model_id)
        assert session.get(ModelProfile, profile["id"])
        persisted_chat = session.get(Chat, chat["id"])
        assert persisted_chat
        assert persisted_chat.active_chat_profile_id == profile["id"]


async def test_model_delete_post_commit_error_keeps_database_authoritative(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = settings.model_dir / "commit-completed"
    managed.mkdir(parents=True)
    (managed / "model.gguf").write_bytes(b"committed-model")
    model_id = "model_delete_commit_completed"
    create_managed_model(model_id=model_id, path=managed, files=["model.gguf"])
    original_commit = Session.commit

    def commit_then_fail(session: Session) -> None:
        original_commit(session)
        raise RuntimeError("injected post-commit reporting failure")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", commit_then_fail)
        response = await client.delete(
            f"/api/models/{model_id}",
            params={"delete_profiles": True},
        )

    assert response.status_code == 204
    assert not managed.exists()
    assert not (settings.model_dir / ".delete-pending").exists()
    with SessionLocal() as session:
        assert session.get(ModelInstall, model_id) is None


async def test_model_delete_finalization_runs_off_the_event_loop(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = settings.model_dir / "threaded-finalization"
    managed.mkdir(parents=True)
    (managed / "model.gguf").write_bytes(b"threaded")
    model_id = "model_delete_threaded_finalization"
    create_managed_model(model_id=model_id, path=managed, files=["model.gguf"])
    original_finalize = api_module._finalize_model_quarantine
    observed_threads: list[int] = []
    event_loop_thread = threading.get_ident()

    def record_finalize(quarantine: Path | None) -> None:
        observed_threads.append(threading.get_ident())
        original_finalize(quarantine)

    monkeypatch.setattr(api_module, "_finalize_model_quarantine", record_finalize)

    response = await client.delete(
        f"/api/models/{model_id}",
        params={"delete_profiles": True},
    )

    assert response.status_code == 204
    assert observed_threads
    assert all(thread_id != event_loop_thread for thread_id in observed_threads)


async def test_model_delete_flush_failures_never_lose_staged_files(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_stage = settings.model_dir / "flush-before-stage"
    before_stage.mkdir(parents=True)
    (before_stage / "model.gguf").write_bytes(b"before-stage")
    before_id = "model_delete_flush_before"
    create_managed_model(model_id=before_id, path=before_stage, files=["model.gguf"])

    def fail_first_flush(
        _session: Session,
        _objects: object | None = None,
    ) -> None:
        raise RuntimeError("injected pre-stage flush failure")

    with (
        monkeypatch.context() as patch,
        pytest.raises(RuntimeError, match="injected pre-stage flush failure"),
    ):
        patch.setattr(Session, "flush", fail_first_flush)
        await client.delete(f"/api/models/{before_id}", params={"delete_profiles": True})
    assert (before_stage / "model.gguf").read_bytes() == b"before-stage"

    after_stage = settings.model_dir / "flush-after-stage"
    after_stage.mkdir()
    (after_stage / "model.gguf").write_bytes(b"after-stage")
    after_id = "model_delete_flush_after"
    create_managed_model(model_id=after_id, path=after_stage, files=["model.gguf"])
    original_flush = Session.flush

    def fail_staged_flush(
        session: Session,
        objects: object | None = None,
    ) -> None:
        quarantine = settings.model_dir / ".delete-pending"
        staged = quarantine.exists() and any(
            (candidate / "payload").exists() for candidate in quarantine.iterdir()
        )
        if staged:
            raise RuntimeError("injected post-stage flush failure")
        original_flush(session, objects=objects)  # type: ignore[arg-type]

    with (
        monkeypatch.context() as patch,
        pytest.raises(RuntimeError, match="injected post-stage flush failure"),
    ):
        patch.setattr(Session, "flush", fail_staged_flush)
        await client.delete(f"/api/models/{after_id}", params={"delete_profiles": True})

    assert (after_stage / "model.gguf").read_bytes() == b"after-stage"
    with SessionLocal() as session:
        assert session.get(ModelInstall, before_id)
        assert session.get(ModelInstall, after_id)


async def test_model_delete_finalization_failure_is_recoverable_after_commit(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = settings.model_dir / "finalize-failure"
    managed.mkdir(parents=True)
    (managed / "model.gguf").write_bytes(b"finalize")
    model_id = "model_delete_finalize_failure"
    create_managed_model(model_id=model_id, path=managed, files=["model.gguf"])

    def fail_rmtree(_path: Path) -> None:
        raise OSError("injected finalization failure")

    with monkeypatch.context() as patch:
        patch.setattr(api_module.shutil, "rmtree", fail_rmtree)
        deleted = await client.delete(
            f"/api/models/{model_id}",
            params={"delete_profiles": True},
        )
    assert deleted.status_code == 204
    assert not managed.exists()
    with SessionLocal() as session:
        assert session.get(ModelInstall, model_id) is None

    quarantine_parent = settings.model_dir / ".delete-pending"
    quarantines = list(quarantine_parent.iterdir())
    assert len(quarantines) == 1
    assert quarantines[0].anchor == managed.anchor
    assert (quarantines[0] / ".model-id").read_text(encoding="utf-8") == model_id
    assert (quarantines[0] / "payload" / "model.gguf").read_bytes() == b"finalize"

    with SessionLocal() as session:
        api_module.recover_model_delete_quarantines(
            session,
            settings.model_dir.resolve(),
        )
    assert not quarantine_parent.exists()


def test_model_quarantine_rejects_nested_filesystem_links(
    settings: Settings,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-quarantine"
    outside.mkdir()
    sentinel = outside / "sentinel.gguf"
    sentinel.write_bytes(b"outside")
    quarantine = api_module._new_model_quarantine(
        settings.model_dir.resolve(),
        "model_nested_link",
    )
    files = quarantine / "files"
    files.mkdir()
    nested = files / "nested"
    try:
        nested.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem links are unavailable in this test environment")

    with pytest.raises(ValueError, match="filesystem link"):
        api_module._safe_quarantine_file_path(
            quarantine,
            PurePosixPath("nested/model.gguf"),
        )
    with pytest.raises(OSError, match="link"):
        api_module._finalize_model_quarantine(quarantine)

    assert sentinel.read_bytes() == b"outside"
    assert (quarantine / ".model-id").is_file()


async def test_model_delete_preserves_shared_files_and_restores_exclusive_files(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = settings.model_dir / "shared-manifest"
    shared.mkdir(parents=True)
    (shared / "first.gguf").write_bytes(b"first")
    (shared / "shared.gguf").write_bytes(b"shared")
    (shared / "second.gguf").write_bytes(b"second")
    first_id = "model_delete_shared_first"
    second_id = "model_delete_shared_second"
    create_managed_model(
        model_id=first_id,
        path=shared,
        files=["first.gguf", "shared.gguf"],
    )
    create_managed_model(
        model_id=second_id,
        path=shared,
        files=["second.gguf", "shared.gguf"],
    )

    def fail_commit(_session: Session) -> None:
        raise RuntimeError("injected shared deletion commit failure")

    with (
        monkeypatch.context() as patch,
        pytest.raises(RuntimeError, match="injected shared deletion commit failure"),
    ):
        patch.setattr(Session, "commit", fail_commit)
        await client.delete(f"/api/models/{first_id}", params={"delete_profiles": True})
    assert (shared / "first.gguf").read_bytes() == b"first"
    assert (shared / "shared.gguf").read_bytes() == b"shared"
    assert (shared / "second.gguf").read_bytes() == b"second"

    assert (
        await client.delete(f"/api/models/{first_id}", params={"delete_profiles": True})
    ).status_code == 204
    assert not (shared / "first.gguf").exists()
    assert (shared / "shared.gguf").read_bytes() == b"shared"
    assert (shared / "second.gguf").read_bytes() == b"second"
    assert (
        await client.delete(f"/api/models/{second_id}", params={"delete_profiles": True})
    ).status_code == 204
    assert not shared.exists()


async def test_model_delete_rejects_manifest_escape_and_linked_managed_paths(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
) -> None:
    shared = settings.model_dir / "unsafe-manifest"
    shared.mkdir(parents=True)
    (shared / "keep.gguf").write_bytes(b"keep")
    unsafe_id = "model_delete_unsafe_manifest"
    sibling_id = "model_delete_unsafe_sibling"
    create_managed_model(model_id=unsafe_id, path=shared, files=["../escape.gguf"])
    create_managed_model(model_id=sibling_id, path=shared, files=["keep.gguf"])

    unsafe = await client.delete(
        f"/api/models/{unsafe_id}",
        params={"delete_profiles": True},
    )
    assert unsafe.status_code == 422
    assert (shared / "keep.gguf").read_bytes() == b"keep"
    with SessionLocal() as session:
        assert session.get(ModelInstall, unsafe_id)

    outside = tmp_path / "outside-model"
    outside.mkdir()
    (outside / "outside.gguf").write_bytes(b"outside")
    linked = settings.model_dir / "linked-model"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem links are unavailable in this Windows test environment")
    linked_id = "model_delete_linked_path"
    create_managed_model(model_id=linked_id, path=linked, files=["outside.gguf"])

    response = await client.delete(
        f"/api/models/{linked_id}",
        params={"delete_profiles": True},
    )
    assert response.status_code == 422
    assert (outside / "outside.gguf").read_bytes() == b"outside"
    assert linked.is_symlink()
    with SessionLocal() as session:
        assert session.get(ModelInstall, linked_id)


async def test_download_pause_resume_and_cancel(client: AsyncClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gate = asyncio.Event()

    async def wait_for_release(_manager: DownloadManager, _job_id: str) -> None:
        await gate.wait()

    monkeypatch.setattr(DownloadManager, "_download", wait_for_release)
    created = await client.post(
        "/api/downloads",
        json={"remote_id": "owner/model", "role": "chat", "engine": "llama.cpp"},
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    paused = await client.post(f"/api/downloads/{job_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"

    resumed = await client.post(f"/api/downloads/{job_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "queued"

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    cannot_resume = await client.post(f"/api/downloads/{job_id}/resume")
    assert cannot_resume.status_code == 409

    failed = await client.post(
        "/api/downloads",
        json={"remote_id": "owner/retry", "role": "chat", "engine": "llama.cpp"},
    )
    failed_id = failed.json()["id"]
    with SessionLocal() as session:
        failed_job = session.get(Job, failed_id)
        assert failed_job
        failed_job.status = JobStatus.FAILED.value
        failed_job.phase = "failed"
        failed_job.error = "incomplete HTTP read"
        failed_job.completed_at = utcnow()
        session.commit()

    retried = await client.post(f"/api/downloads/{failed_id}/resume")
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert retried.json()["phase"] == "retry queued"
    assert retried.json()["error"] is None
    await client.post(f"/api/jobs/{failed_id}/cancel")


async def test_catalog_preflight_blocks_gated_unsafe_weights(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Unsafe model",
                "gated": True,
                "compatibility": "advanced_import",
                "compatibility_reasons": ["manual review required"],
            },
            "revision": revision,
            "files": [{"filename": "weights/model.ckpt", "size": 1024, "sha256": None}],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "main",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["weights/model.ckpt"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is False
    assert payload["install_plan"]["compatibility"] == "unsupported"
    assert payload["install_plan"]["failure_code"] is not None
    blocked = {check["id"] for check in payload["checks"] if check["status"] == "block"}
    assert {"weights", "access"}.issubset(blocked)


async def test_catalog_preflight_autoselects_smallest_gguf(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Safe model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["GGUF artifact detected"],
            },
            "revision": revision,
            "files": [
                {"filename": "large.gguf", "size": 2048, "sha256": "b" * 64},
                {"filename": "small.gguf", "size": 1024, "sha256": "a" * 64},
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "abc123",
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["can_install"] is True
    assert response.json()["selected_files"] == ["small.gguf"]
    assert response.json()["expected_sha256"] == {"small.gguf": "a" * 64}
    plan = response.json()["install_plan"]
    assert plan["compatibility"] == "supported"
    tampered = await client.post(
        "/api/downloads",
        json={
            "install_plan_id": plan["id"],
            "remote_id": "owner/model",
            "revision": "abc123",
            "role": "chat",
            "engine": "llama.cpp",
            "allow_patterns": ["large.gguf"],
        },
    )
    assert tampered.status_code == 422
    assert "immutable plan" in tampered.json()["detail"]
    checksum = next(check for check in response.json()["checks"] if check["id"] == "checksum")
    assert checksum["status"] == "pass"


async def test_catalog_preflight_prefers_balanced_gguf_quantization(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Quantized model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["GGUF artifact detected"],
            },
            "revision": revision,
            "files": [
                {"filename": "model-Q2_K.gguf", "size": 1024, "sha256": "a" * 64},
                {"filename": "model-Q4_K_M.gguf", "size": 2048, "sha256": "b" * 64},
                {"filename": "model-Q8_0.gguf", "size": 4096, "sha256": "c" * 64},
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "abc123",
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["selected_files"] == ["model-Q4_K_M.gguf"]


async def test_catalog_preflight_selects_a_complete_split_gguf_set(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        del requested_role
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Split model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["GGUF artifact detected"],
            },
            "revision": revision,
            "files": [
                {
                    "filename": "model-Q4_K_M-00002-of-00002.gguf",
                    "size": 2048,
                    "sha256": "b" * 64,
                },
                {
                    "filename": "model-Q4_K_M-00001-of-00002.gguf",
                    "size": 1024,
                    "sha256": "a" * 64,
                },
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "abc123",
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is True
    assert payload["selected_files"] == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
    ]
    assert payload["download_bytes"] == 3072
    assert payload["expected_sha256"] == {
        "model-Q4_K_M-00001-of-00002.gguf": "a" * 64,
        "model-Q4_K_M-00002-of-00002.gguf": "b" * 64,
    }
    assert payload["install_plan"]["compatibility"] == "supported"
    assert len(payload["install_plan"]["plan_hash"]) == 64

    captured: dict[str, DownloadRequest] = {}

    def create_from_plan(
        _manager: DownloadManager,
        session: Session,
        request: DownloadRequest,
    ) -> Job:
        captured["request"] = request
        job = Job(
            kind="download",
            status="queued",
            payload_json=request.model_dump(mode="json"),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

    monkeypatch.setattr(DownloadManager, "create", create_from_plan)
    accepted = await client.post(
        "/api/downloads",
        json={
            "install_plan_id": payload["install_plan"]["id"],
            "remote_id": payload["remote_id"],
            "source_remote_id": payload["source_remote_id"],
            "revision": payload["revision"],
            "role": "chat",
            "engine": "llama.cpp",
            "allow_patterns": payload["selected_files"],
            "expected_sha256": payload["expected_sha256"],
            "comfy_paths": payload["comfy_paths"],
            "workflow_template_id": payload["workflow_template_id"],
            "workflow_template_sha256": payload["workflow_template_sha256"],
        },
    )
    assert accepted.status_code == 202
    assert captured["request"].install_plan_id == payload["install_plan"]["id"]

    rejected = await client.post(
        "/api/downloads",
        json={
            **captured["request"].model_dump(mode="json"),
            "revision": "different-revision",
        },
    )
    assert rejected.status_code == 422
    assert "immutable plan" in rejected.json()["detail"]


async def test_catalog_preflight_explains_an_incomplete_split_gguf_set(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        del requested_role
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Incomplete split model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["GGUF artifact detected"],
            },
            "revision": revision,
            "files": [
                {
                    "filename": "model-Q4_K_M-00001-of-00002.gguf",
                    "size": 1024,
                    "sha256": "a" * 64,
                }
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "abc123",
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is False
    selection = next(check for check in payload["checks"] if check["id"] == "selection")
    assert selection["status"] == "block"
    assert "missing shard(s) 00002" in selection["detail"]


async def test_catalog_preflight_autoselects_safe_media_checkpoint(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Safe image model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["safetensors artifact detected"],
            },
            "revision": revision,
            "files": [
                {"filename": "model.safetensors", "size": 2048, "sha256": "a" * 64},
                {"filename": "vae.safetensors", "size": 1024, "sha256": "b" * 64},
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/image-model/preflight",
        json={
            "revision": "abc123",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["can_install"] is True
    assert response.json()["selected_files"] == ["model.safetensors"]


async def test_comfy_catalog_preflight_offers_a_provisional_adaptive_checkpoint(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.media_engine = "comfyui"
    external_runtime = tmp_path / "external-comfy"
    external_runtime.mkdir()
    settings.comfy_executable = external_runtime / "python.exe"
    settings.comfy_executable.write_bytes(b"external runtime")
    settings.comfy_directory = external_runtime / "ComfyUI"
    settings.comfy_directory.mkdir()
    (settings.comfy_directory / "main.py").write_text("", encoding="utf-8")

    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "New checkpoint",
                "license_id": "apache-2.0",
                "pipeline_tag": "text-to-image",
                "formats": ["safetensors"],
                "compatibility": "likely",
                "compatibility_reasons": ["image pipeline metadata detected"],
            },
            "revision": "a" * 40,
            "files": [
                {
                    "filename": "weights/model.safetensors",
                    "size": 2048,
                    "sha256": "b" * 64,
                }
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/new-checkpoint/preflight",
        json={
            "revision": "main",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is True
    assert payload["revision"] == "a" * 40
    assert payload["selected_files"] == ["weights/model.safetensors"]
    assert payload["comfy_paths"] == {"checkpoints": "weights"}
    assert payload["workflow_template_id"].startswith("lma_image_checkpoint_v1_")
    workflow_check = next(
        check for check in payload["checks"] if check["id"] == "workflow-template"
    )
    assert workflow_check["status"] == "warn"


async def test_comfy_catalog_preflight_blocks_an_unreviewed_managed_runtime(
    client: AsyncClient,
    settings: Settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.media_engine = "comfyui"

    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "New checkpoint",
                "license_id": "apache-2.0",
                "pipeline_tag": "text-to-image",
                "formats": ["safetensors"],
                "compatibility": "likely",
                "compatibility_reasons": ["image pipeline metadata detected"],
            },
            "revision": "a" * 40,
            "files": [
                {
                    "filename": "model.safetensors",
                    "size": 2048,
                    "sha256": "b" * 64,
                }
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    blocked_status = RuntimeStatus(
        engine="comfyui",
        release="v0.28.0",
        state="unsupported",
        supported=False,
        distribution="external-gpl-3.0",
        license="GPL-3.0-only",
        security_status="blocked",
        security_message="Automatic setup is paused pending security advisories.",
    )
    monkeypatch.setattr(
        RuntimeProvisioner,
        "status",
        lambda _self, _engine: blocked_status,
    )
    response = await client.post(
        "/api/catalog/owner/new-checkpoint/preflight",
        json={
            "revision": "main",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is False
    runtime_check = next(check for check in payload["checks"] if check["id"] == "runtime")
    assert runtime_check["status"] == "block"
    assert "security advisories" in runtime_check["detail"]
