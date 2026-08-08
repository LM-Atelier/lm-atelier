from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.exports import _CAS_IMPORT_SESSION_KEY, ProjectExporter
from local_lm.main import create_app
from local_lm.models import (
    Artifact,
    Chat,
    Message,
    ModelProfile,
    Run,
    WorkflowRevision,
)
from local_lm.profile_service import AUTO_PROFILE_ID
from local_lm.project_dependencies import DependencySourceIndex
from local_lm.project_portability import (
    LOCAL_PATH_REDACTION,
    has_local_path,
    redact_local_paths,
)
from local_lm.schemas import SettingField, VisionSettings

WINDOWS_MODEL_PATH = r"C:\Users\alice\AppData\Local\LM Atelier\models\private.gguf"
WINDOWS_LOG_PATH = r"\\workstation\private-share\lm-atelier\worker.log"
POSIX_MODEL_PATH = "/home/alice/.local/share/lm-atelier/models/private.safetensors"
POSIX_LOG_PATH = "/var/tmp/lm-atelier/private-worker.log"


def test_portability_scrubber_preserves_remote_ids_and_redacts_path_keys() -> None:
    value = {
        WINDOWS_MODEL_PATH: "path stored as a key",
        "remote": "https://huggingface.co/example/portable-model",
        "remote_file_uri": "file://workstation/private-share/model.gguf",
        "forward_slash_share": "//workstation/private-share/model.gguf",
        "home_relative": "~/.cache/lm-atelier/model.gguf",
        "traversal": "../../outside/private-cache",
    }

    redacted = redact_local_paths(value)

    assert redacted == {
        LOCAL_PATH_REDACTION: "path stored as a key",
        "remote": "https://huggingface.co/example/portable-model",
        "remote_file_uri": LOCAL_PATH_REDACTION,
        "forward_slash_share": LOCAL_PATH_REDACTION,
        "home_relative": LOCAL_PATH_REDACTION,
        "traversal": LOCAL_PATH_REDACTION,
    }
    assert not has_local_path(redacted)


async def _wait_for_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("run did not complete")


def _manifest(archive_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return json.loads(archive.read("manifest.json"))


def _rewrite_archive(archive_bytes: bytes, manifest: dict[str, Any]) -> bytes:
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
    return output.getvalue()


def _replace_json_value(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_json_value(child, source, destination) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_json_value(child, source, destination) for child in value]
    return destination if value == source else value


def _cas_path(settings: Settings, digest: str) -> Path:
    return settings.artifact_dir / digest[:2] / digest[2:4] / digest


async def _media_project_archive(client: AsyncClient) -> tuple[bytes, str]:
    project = (await client.post("/api/projects", json={"name": "CAS transaction"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "CAS media", "project_id": project["id"]},
        )
    ).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create content-addressed test media", "mode": "image"},
    )
    assert accepted.status_code == 202
    await _wait_for_run(client, accepted.json()["run"]["id"])
    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201
    archive = await client.get(exported.json()["url"])
    manifest = _manifest(archive.content)
    return archive.content, str(manifest["artifacts"][0]["sha256"])


async def test_project_vision_context_round_trip_and_legacy_defaults(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Portable vision profile",
                "role": "chat",
                "engine": "mock",
            },
        )
    ).json()
    project = (await client.post("/api/projects", json={"name": "Vision portable"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Vision context", "project_id": project["id"]},
        )
    ).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Keep the visual context portable", "mode": "text"},
    )
    source_run = await _wait_for_run(client, accepted.json()["run"]["id"])

    with SessionLocal() as session:
        stored_chat = session.get(Chat, chat["id"])
        stored_run = session.get(Run, source_run["id"])
        assert stored_chat and stored_run
        stored_chat.active_vision_profile_id = profile["id"]
        stored_chat.vision_settings_json = {
            "max_images": 2,
            "max_video_frames": 5,
            "include_prior_visual": False,
            "verify_image_edits": True,
        }
        stored_run.vision_profile_id = profile["id"]
        stored_run.provenance_json = {
            **stored_run.provenance_json,
            "context": {
                **stored_run.provenance_json.get("context", {}),
                "vision": {
                    "mode": "bridge",
                    "profile_id": profile["id"],
                    "profile": {
                        "profile_id": profile["id"],
                        "profile_name": profile["name"],
                        "install_id": "local-only-install",
                    },
                },
            },
        }
        session.commit()

    exported = await client.post(f"/api/projects/{project['id']}/export")
    archive = await client.get(exported.json()["url"])
    manifest = _manifest(archive.content)
    assert manifest["version"] == 6
    assert manifest["chats"][0]["active_vision_profile_id"] == profile["id"]
    assert manifest["chats"][0]["vision_settings_json"] == {
        "max_images": 2,
        "max_video_frames": 5,
        "include_prior_visual": False,
        "verify_image_edits": True,
        "compile_visual_prompts": True,
    }
    assert manifest["runs"][0]["vision_profile_id"] == profile["id"]
    portable_vision = manifest["runs"][0]["provenance_json"]["context"]["vision"]
    assert portable_vision["profile_id"] == profile["id"]
    assert portable_vision["profile"]["profile_id"] == profile["id"]
    assert "install_id" not in portable_vision["profile"]

    legacy = json.loads(json.dumps(manifest))
    legacy["version"] = 4
    for chat_record in legacy["chats"]:
        chat_record.pop("active_vision_profile_id", None)
        chat_record.pop("vision_settings_json", None)
    for run_record in legacy["runs"]:
        run_record.pop("vision_profile_id", None)
        context = run_record.get("provenance_json", {}).get("context")
        if isinstance(context, dict):
            context.pop("vision", None)
    legacy_archive = _rewrite_archive(archive.content, legacy)

    target_settings = Settings(
        data_dir=tmp_path / "vision-portable-target",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    target_app = create_app(target_settings)
    async with (
        target_app.router.lifespan_context(target_app),
        AsyncClient(
            transport=ASGITransport(app=target_app),
            base_url="http://testserver",
        ) as target,
    ):
        session_response = await target.post("/api/session")
        target.headers["x-local-lm-csrf"] = session_response.json()["csrf_token"]
        imported = await target.post(
            "/api/projects/import",
            files={
                "archive": (
                    "vision-portable.lm-atelier.zip",
                    archive.content,
                    "application/zip",
                )
            },
        )
        assert imported.status_code == 201, imported.text
        with SessionLocal() as session:
            imported_chat = session.scalar(
                select(Chat).where(Chat.project_id == imported.json()["id"])
            )
            assert imported_chat
            imported_run = session.scalar(select(Run).where(Run.chat_id == imported_chat.id))
            assert imported_run
            assert imported_chat.active_vision_profile_id not in {
                None,
                AUTO_PROFILE_ID,
                profile["id"],
            }
            assert imported_chat.vision_settings_json == {
                "max_images": 2,
                "max_video_frames": 5,
                "include_prior_visual": False,
                "verify_image_edits": True,
                "compile_visual_prompts": True,
            }
            assert imported_run.vision_profile_id == imported_chat.active_vision_profile_id
            imported_vision = imported_run.provenance_json["context"]["vision"]
            assert imported_vision["profile_id"] == imported_run.vision_profile_id
            assert imported_vision["profile"]["profile_id"] == imported_run.vision_profile_id

        imported_legacy = await target.post(
            "/api/projects/import",
            files={
                "archive": (
                    "legacy-vision-defaults.lm-atelier.zip",
                    legacy_archive,
                    "application/zip",
                )
            },
        )
        assert imported_legacy.status_code == 201, imported_legacy.text
        with SessionLocal() as session:
            legacy_chat = session.scalar(
                select(Chat).where(Chat.project_id == imported_legacy.json()["id"])
            )
            assert legacy_chat
            legacy_run = session.scalar(select(Run).where(Run.chat_id == legacy_chat.id))
            assert legacy_run
            assert legacy_chat.active_vision_profile_id == AUTO_PROFILE_ID
            assert legacy_chat.vision_settings_json == {
                "max_images": 4,
                "max_video_frames": 6,
                "include_prior_visual": True,
                "verify_image_edits": False,
                "compile_visual_prompts": True,
            }
            assert legacy_run.vision_profile_id is None


async def test_project_round_trip_redacts_paths_and_remaps_portable_identifiers(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Portable image profile",
                "role": "image",
                "engine": "mock",
                "request_settings": {"steps": 4},
            },
        )
    ).json()
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Portable image workflow",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"loader": {"class_type": "PortableLoader"}},
                "trusted": True,
            },
        )
    ).json()
    source_revision = workflow["revisions"][0]
    project = (
        await client.post(
            "/api/projects",
            json={
                "name": "Portable privacy",
                "image_workflow_revision_id": source_revision["id"],
            },
        )
    ).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Portable run", "project_id": project["id"]},
        )
    ).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": profile["id"]},
    )
    assert selected.status_code == 200
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a portable blue square", "mode": "image"},
    )
    assert accepted.status_code == 202
    source_run = await _wait_for_run(client, accepted.json()["run"]["id"])
    source_artifact_id = source_run["provenance_json"]["outputs"][0]["artifact_id"]

    with SessionLocal() as session:
        run = session.get(Run, source_run["id"])
        source_profile = session.get(ModelProfile, profile["id"])
        revision = session.get(WorkflowRevision, source_revision["id"])
        artifact = session.get(Artifact, source_artifact_id)
        message = session.get(Message, run.assistant_message_id if run else "")
        assert run and source_profile and revision and artifact and message

        source_profile.load_settings_json = {
            "context_length": 4096,
            "cache_directory": WINDOWS_MODEL_PATH,
            "relative_cache": r"..\..\outside\private-cache",
        }
        source_profile.request_settings_json = {
            **source_profile.request_settings_json,
            "output_directory": POSIX_MODEL_PATH,
        }
        revision.api_graph_json = {
            **revision.api_graph_json,
            "external_loader": {"path": POSIX_MODEL_PATH},
        }
        run.settings_json = {**run.settings_json, "cache_path": WINDOWS_MODEL_PATH}
        provenance = {
            **run.provenance_json,
            "input_artifact_ids": [source_artifact_id],
            "model_selection": {
                "mode": "explicit",
                "profile_id": profile["id"],
                "profile_name": profile["name"],
            },
            "model": {
                "profile_id": profile["id"],
                "profile_name": profile["name"],
                "install_id": "model_source_machine_only",
                "local_path": WINDOWS_MODEL_PATH,
                "manifest": {"cache_path": POSIX_MODEL_PATH},
                "source": {
                    "provider": "huggingface",
                    "remote_id": "example/portable-model",
                    "revision": "abc123",
                    "metadata": {"download_path": WINDOWS_MODEL_PATH},
                },
            },
            "worker": {
                "state": "ready",
                "pid": 4242,
                "command": [WINDOWS_MODEL_PATH, "--output", POSIX_MODEL_PATH],
                "log_path": WINDOWS_LOG_PATH,
                "stderr_tail": f"failed to read {POSIX_LOG_PATH}",
            },
            "diagnostic": (f"Windows model {WINDOWS_MODEL_PATH}; POSIX model {POSIX_MODEL_PATH}"),
        }
        run.provenance_json = provenance
        run.error = f"worker log: {WINDOWS_LOG_PATH}"
        artifact.metadata_json = {
            **artifact.metadata_json,
            "cache_path": POSIX_MODEL_PATH,
            "poster_for": source_artifact_id,
        }
        generation_part = next(part for part in message.parts if part.type == "generation_metadata")
        generation_part.metadata_json = {
            "run_id": run.id,
            "provenance": provenance,
            "debug_path": WINDOWS_LOG_PATH,
        }
        session.commit()

    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201
    archive = await client.get(exported.json()["url"])
    manifest = _manifest(archive.content)
    exported_run = manifest["runs"][0]
    exported_provenance = exported_run["provenance_json"]
    assert not has_local_path(exported_provenance)
    assert exported_provenance["model"]["profile_id"] == profile["id"]
    assert "install_id" not in exported_provenance["model"]
    assert "local_path" not in exported_provenance["model"]
    assert exported_provenance["worker"] == {"state": "ready"}
    serialized_manifest = json.dumps(manifest)
    assert "alice" not in serialized_manifest
    assert "private-worker.log" not in serialized_manifest
    assert (
        manifest["dependencies"]["profiles"][0]["load_settings"]["cache_directory"]
        == LOCAL_PATH_REDACTION
    )
    assert (
        manifest["dependencies"]["profiles"][0]["load_settings"]["relative_cache"]
        == LOCAL_PATH_REDACTION
    )
    assert (
        manifest["dependencies"]["workflows"][0]["revisions"][0]["api_graph"]["external_loader"][
            "path"
        ]
        == LOCAL_PATH_REDACTION
    )
    assert manifest["artifacts"][0]["metadata"]["cache_path"] == LOCAL_PATH_REDACTION

    portable_artifact_id = "portable-artifact-output"
    imported_manifest = _replace_json_value(
        manifest,
        source_artifact_id,
        portable_artifact_id,
    )
    imported_run_record = imported_manifest["runs"][0]
    imported_run_record["settings_json"]["cache_path"] = POSIX_MODEL_PATH
    imported_run_record["provenance_json"]["model"]["local_path"] = WINDOWS_MODEL_PATH
    imported_run_record["provenance_json"]["worker"] = {
        "command": [WINDOWS_MODEL_PATH],
        "log_path": POSIX_LOG_PATH,
    }
    imported_run_record["provenance_json"]["diagnostic"] = (
        f"{WINDOWS_MODEL_PATH} and {POSIX_MODEL_PATH}"
    )
    imported_manifest["dependencies"]["profiles"][0]["request_settings"]["output_directory"] = (
        POSIX_MODEL_PATH
    )
    imported_manifest["dependencies"]["profiles"][0]["request_settings"]["relative_cache"] = (
        "../../outside/private-cache"
    )
    imported_manifest["dependencies"]["workflows"][0]["revisions"][0]["api_graph"][
        "external_loader"
    ]["path"] = WINDOWS_MODEL_PATH
    imported_manifest["artifacts"][0]["metadata"]["cache_path"] = POSIX_MODEL_PATH
    portable_archive = _rewrite_archive(archive.content, imported_manifest)

    target_settings = Settings(
        data_dir=tmp_path / "portable-target",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    target_app = create_app(target_settings)
    async with (
        target_app.router.lifespan_context(target_app),
        AsyncClient(
            transport=ASGITransport(app=target_app),
            base_url="http://testserver",
        ) as target,
    ):
        session_response = await target.post("/api/session")
        target.headers["x-local-lm-csrf"] = session_response.json()["csrf_token"]
        imported = await target.post(
            "/api/projects/import",
            files={
                "archive": (
                    "portable.lm-atelier.zip",
                    portable_archive,
                    "application/zip",
                )
            },
        )
        assert imported.status_code == 201, imported.text

        with SessionLocal() as session:
            imported_chat = session.scalar(
                select(Chat).where(Chat.project_id == imported.json()["id"])
            )
            assert imported_chat
            imported_run = session.scalar(select(Run).where(Run.chat_id == imported_chat.id))
            assert imported_run
            assert imported_run.profile_id != profile["id"]
            assert imported_run.workflow_revision_id != source_revision["id"]
            assert not has_local_path(imported_run.settings_json)
            assert not has_local_path(imported_run.provenance_json)
            assert imported_run.provenance_json["model"]["profile_id"] == (imported_run.profile_id)
            assert "install_id" not in imported_run.provenance_json["model"]
            assert "local_path" not in imported_run.provenance_json["model"]
            imported_revision = session.get(
                WorkflowRevision,
                imported_run.workflow_revision_id,
            )
            assert imported_revision
            assert imported_run.provenance_json["workflow"]["revision_id"] == (imported_revision.id)
            assert imported_run.provenance_json["workflow"]["definition_id"] == (
                imported_revision.workflow_id
            )
            assert imported_run.provenance_json["workflow"]["trusted"] is False
            remapped_artifact_id = f"sha256:{manifest['artifacts'][0]['sha256']}"
            assert imported_run.provenance_json["input_artifact_ids"] == [remapped_artifact_id]
            assert imported_run.provenance_json["outputs"][0]["artifact_id"] == (
                remapped_artifact_id
            )

            imported_profile = session.get(ModelProfile, imported_run.profile_id)
            assert imported_profile
            assert imported_profile.model_install_id is None
            assert not has_local_path(imported_profile.load_settings_json)
            assert not has_local_path(imported_profile.request_settings_json)
            assert imported_profile.request_settings_json["output_directory"] == (
                LOCAL_PATH_REDACTION
            )
            assert imported_profile.request_settings_json["relative_cache"] == (
                LOCAL_PATH_REDACTION
            )
            assert imported_revision.api_graph_json["external_loader"]["path"] == (
                LOCAL_PATH_REDACTION
            )
            imported_artifact = session.get(Artifact, remapped_artifact_id)
            assert imported_artifact
            assert imported_artifact.metadata_json["poster_for"] == remapped_artifact_id
            assert imported_artifact.metadata_json["cache_path"] == LOCAL_PATH_REDACTION

            assistant = session.get(Message, imported_run.assistant_message_id)
            assert assistant
            generation_part = next(
                part for part in assistant.parts if part.type == "generation_metadata"
            )
            assert generation_part.metadata_json["run_id"] == imported_run.id
            assert generation_part.metadata_json["provenance"] == (imported_run.provenance_json)


async def test_project_import_rejects_unmapped_and_mismatched_provenance_references(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Reference source"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Reference chat", "project_id": project["id"]},
        )
    ).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Keep this portable", "mode": "text"},
    )
    assert accepted.status_code == 202
    await _wait_for_run(client, accepted.json()["run"]["id"])
    exported = await client.post(
        f"/api/projects/{project['id']}/export",
        params={"include_media": False},
    )
    archive = await client.get(exported.json()["url"])
    baseline = _manifest(archive.content)

    unknown_profile = json.loads(json.dumps(baseline))
    unknown_profile["runs"][0]["provenance_json"]["model_selection"]["profile_id"] = (
        "profile_not_in_dependencies"
    )

    unknown_artifact = json.loads(json.dumps(baseline))
    unknown_artifact["runs"][0]["provenance_json"]["input_artifact_ids"] = ["artifact_not_declared"]

    mismatched_workflow = json.loads(json.dumps(baseline))
    mismatched_workflow["runs"][0]["provenance_json"]["workflow"] = {
        "definition_id": "workflow_not_in_dependencies",
        "revision_id": "revision_not_in_dependencies",
    }

    cases = (
        (unknown_profile, "incompatible role"),
        (unknown_artifact, "undeclared artifact"),
        (mismatched_workflow, "incompatible operation"),
    )
    project_count = len((await client.get("/api/projects")).json())
    for manifest, expected in cases:
        response = await client.post(
            "/api/projects/import",
            files={
                "archive": (
                    "invalid-reference.lm-atelier.zip",
                    _rewrite_archive(archive.content, manifest),
                    "application/zip",
                )
            },
        )
        assert response.status_code == 422, response.text
        assert expected in response.json()["detail"]
        assert len((await client.get("/api/projects")).json()) == project_count


async def test_import_failure_removes_only_new_content_addressed_files(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, digest = await _media_project_archive(client)
    existing_path = _cas_path(settings, digest)
    assert existing_path.is_file()
    existing_content = existing_path.read_bytes()

    def reject_records(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("injected record failure")

    with monkeypatch.context() as patch:
        patch.setattr(ProjectExporter, "_import_records", reject_records)
        failed = await client.post(
            "/api/projects/import",
            files={
                "archive": (
                    "existing-cas.lm-atelier.zip",
                    archive,
                    "application/zip",
                )
            },
        )
    assert failed.status_code == 422
    assert existing_path.read_bytes() == existing_content

    target_settings = Settings(
        data_dir=tmp_path / "new-cas-target",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    target_path = _cas_path(target_settings, digest)
    target_app = create_app(target_settings)
    async with (
        target_app.router.lifespan_context(target_app),
        AsyncClient(
            transport=ASGITransport(app=target_app),
            base_url="http://testserver",
        ) as target,
    ):
        session_response = await target.post("/api/session")
        target.headers["x-local-lm-csrf"] = session_response.json()["csrf_token"]
        with monkeypatch.context() as patch:
            patch.setattr(ProjectExporter, "_import_records", reject_records)
            failed = await target.post(
                "/api/projects/import",
                files={
                    "archive": (
                        "new-cas.lm-atelier.zip",
                        archive,
                        "application/zip",
                    )
                },
            )
        assert failed.status_code == 422
        assert not target_path.exists()
        with SessionLocal() as session:
            assert not session.scalar(select(Artifact).where(Artifact.sha256 == digest))


async def test_commit_failure_rolls_back_new_content_addressed_files(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, digest = await _media_project_archive(client)
    target_settings = Settings(
        data_dir=tmp_path / "commit-failure-target",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    target_path = _cas_path(target_settings, digest)
    target_app = create_app(target_settings)
    original_commit = Session.commit

    def fail_import_commit(session: Session) -> None:
        if _CAS_IMPORT_SESSION_KEY in session.info:
            raise RuntimeError("injected commit failure")
        original_commit(session)

    async with (
        target_app.router.lifespan_context(target_app),
        AsyncClient(
            transport=ASGITransport(app=target_app),
            base_url="http://testserver",
        ) as target,
    ):
        session_response = await target.post("/api/session")
        target.headers["x-local-lm-csrf"] = session_response.json()["csrf_token"]
        with (
            monkeypatch.context() as patch,
            pytest.raises(RuntimeError, match="injected commit failure"),
        ):
            patch.setattr(Session, "commit", fail_import_commit)
            await target.post(
                "/api/projects/import",
                files={
                    "archive": (
                        "commit-failure.lm-atelier.zip",
                        archive,
                        "application/zip",
                    )
                },
            )
        assert not target_path.exists()
        with SessionLocal() as session:
            assert not session.scalar(select(Artifact).where(Artifact.sha256 == digest))


def test_import_strips_load_scope_settings_from_request_layers() -> None:
    """The API refuses load-scope keys on request layers; an archive could not."""
    from local_lm.project_dependencies import _without_load_scope_settings

    stripped = _without_load_scope_settings({"gpu_layers": 40, "temperature": 0.7}, "chat")

    assert "gpu_layers" not in stripped
    assert stripped["temperature"] == 0.7


def test_import_keeps_settings_the_registry_does_not_define() -> None:
    """A workflow can extend the field set, so unknown keys are not dropped here.

    Dropping them would silently discard legitimate workflow settings, which is
    the same class of failure this is meant to prevent.
    """
    from local_lm.project_dependencies import _without_load_scope_settings

    kept = _without_load_scope_settings({"a_workflow_parameter": 3}, "image")

    assert kept == {"a_workflow_parameter": 3}


def _exporter() -> ProjectExporter:
    return ProjectExporter(Settings(), ArtifactStore(Settings()))


def test_workflow_activation_provenance_exports_only_portable_digests() -> None:
    exporter = _exporter()
    dependencies = DependencySourceIndex(
        profile_roles={},
        preset_roles={},
        revision_operations={"revision-source": "text_to_image"},
        revision_workflow_ids={"revision-source": "workflow-source"},
    )
    provenance = {
        "workflow": {
            "family_id": "wffamily-local",
            "family_name": "Portable image family",
            "revision_id": "revision-source",
            "definition_id": "workflow-source",
            "trusted": True,
            "activation": {
                "id": "wfact-local",
                "resolver_version": "workflow-activation-v1",
                "dependency_contract_sha256": "a" * 64,
                "binding_sha256": "b" * 64,
                "launch_sha256": "c" * 64,
            },
        }
    }

    portable = exporter._portable_provenance(
        provenance,
        Operation.TEXT_TO_IMAGE,
        dependencies,
        set(),
    )

    assert portable["workflow"]["activation"] == {
        "resolver_version": "workflow-activation-v1",
        "dependency_contract_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
    }
    assert portable["workflow"]["family_id"] is None
    assert portable["workflow"]["family_name"] == "Portable image family"
    with pytest.raises(ValueError, match="local workflow family"):
        exporter._validate_portable_provenance(
            provenance,
            Operation.TEXT_TO_IMAGE,
            dependencies,
            set(),
        )
    provenance_without_family = json.loads(json.dumps(provenance))
    provenance_without_family["workflow"]["family_id"] = None
    with pytest.raises(ValueError, match="local workflow activation"):
        exporter._validate_portable_provenance(
            provenance_without_family,
            Operation.TEXT_TO_IMAGE,
            dependencies,
            set(),
        )


def _field(key: str, kind: str = "number", **extra: object) -> SettingField:
    return SettingField(
        key=key,
        label=key,
        type=kind,
        scope="request",
        visibility="basic",
        available=True,
        **extra,  # type: ignore[arg-type]
    )


def test_imported_settings_are_validated_against_the_live_engine_schema() -> None:
    """Imported settings used to be written verbatim.

    A value the REST API would reject could therefore arrive through an archive
    and fail later at generation time, where nothing connects the failure back to
    the import. A described field with a bad value is dropped; a field the role
    schema does not describe is kept, because a workflow contributes fields that
    only exist once that workflow is selected.
    """
    exporter = _exporter()
    exporter._known_fields = {"image": [_field("steps", "integer"), _field("cfg")]}

    result = exporter._generation_settings(
        {"image": {"steps": "not an integer", "cfg": 7.5, "a_workflow_parameter": 3}}
    )

    assert result["image"] == {"cfg": 7.5, "a_workflow_parameter": 3}


def test_one_bad_setting_costs_only_itself() -> None:
    """Rejecting the archive over a stale key would make old exports unimportable."""
    exporter = _exporter()
    exporter._known_fields = {"image": [_field("steps", "integer"), _field("cfg")]}

    result = exporter._generation_settings({"image": {"steps": "not an integer", "cfg": 7.5}})

    assert result["image"] == {"cfg": 7.5}


def test_a_role_with_no_resolvable_schema_imports_unvalidated() -> None:
    """An engine being down must not cost the user their settings."""
    exporter = _exporter()
    exporter._known_fields = {}

    result = exporter._generation_settings({"image": {"steps": 30, "anything": "at all"}})

    assert result["image"] == {"steps": 30, "anything": "at all"}


def test_validation_does_not_defeat_workflow_contributed_fields() -> None:
    """The regression this nearly reintroduced.

    `loras` is contributed by a workflow's input schema, so it is absent from the
    role schema while being entirely legitimate. An earlier attempt validated
    against `ROLE_SETTINGS` and dropped settings exactly like it; validating
    against the live role schema and dropping the undescribed would have done the
    same thing for the same reason.
    """
    exporter = _exporter()
    exporter._known_fields = {"image": [_field("steps", "integer")]}

    result = exporter._generation_settings({"image": {"steps": 30, "loras": [{"id": "x"}]}})

    assert result["image"] == {"steps": 30, "loras": [{"id": "x"}]}


def test_every_vision_setting_this_build_writes_can_be_imported_again() -> None:
    """A hand-written key list made each new vision setting reject our own archives.

    `_validate_vision_settings` refuses unknown keys, which is right - an archive
    must not smuggle settings this build cannot describe. But the known set was a
    literal copy of the model's fields, so adding one field to `VisionSettings`
    caused import to reject a manifest that export had just written, with
    "project manifest has invalid vision settings" and no indication of which key.
    """
    exported = VisionSettings().model_dump(mode="json")

    assert set(exported) == set(VisionSettings.model_fields)
    ProjectExporter._validate_vision_settings(exported)

    with pytest.raises(ValueError, match="invalid vision settings"):
        ProjectExporter._validate_vision_settings({**exported, "not_a_setting": True})
