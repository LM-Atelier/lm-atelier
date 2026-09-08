from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_project_work_plans import _archive

from local_lm.db import SessionLocal
from local_lm.models import Chat, Run
from local_lm.project_portability import has_local_path


@pytest.mark.parametrize("output_count", [1, 3], ids=["text", "media"])
async def test_imported_edited_version_keeps_accepted_context_and_can_be_edited_again(
    app: FastAPI, client: AsyncClient, output_count: int
) -> None:
    source, edited, content = await _archive(app, client, complete=True, output_count=output_count)
    from local_lm.accepted_turn_context import accepted_context

    with SessionLocal() as session:
        original = session.get(Run, edited["run"]["id"])
        assert original is not None
        original_context = accepted_context(session, original)
        assert original_context is not None
    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": ("accepted-context.lm-atelier.zip", content, "application/zip"),
        },
    )
    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        chat = session.scalar(select(Chat).where(Chat.project_id == imported.json()["id"]))
        assert chat is not None
        runs = list(session.scalars(select(Run).where(Run.chat_id == chat.id)))
        source_run = next(
            run
            for run in runs
            if run.provenance_json["imported_from_run_id"] == source["run"]["id"]
        )
        edited_run = next(
            run
            for run in runs
            if run.provenance_json["imported_from_run_id"] == edited["run"]["id"]
        )
        imported_context = accepted_context(session, edited_run)
        assert imported_context is not None
        assert imported_context.run_id == edited_run.id
        assert imported_context.chat_id == chat.id
        assert imported_context.source_message_id == source_run.user_message_id
        assert imported_context.source_run_id == source_run.id
        assert original_context.routing_mode is not None
        assert imported_context.routing_mode == original_context.routing_mode
        assert imported_context.settings == original_context.settings
        assert imported_context.context_limit == original_context.context_limit
        assert [message.content for message in imported_context.messages] == [
            message.content for message in original_context.messages
        ]
        assert [message.source_message_id for message in imported_context.messages] != [
            message.source_message_id for message in original_context.messages
        ]
        edited_message_id = edited_run.user_message_id
    loaded = await client.get(f"/api/messages/{edited_message_id}/edit-source")
    assert loaded.status_code == 200, loaded.text
    async with app.state.services.scheduler.lease("primary"):
        queued = await client.post(
            f"/api/messages/{edited_message_id}/edits",
            json={
                "text": "Describe an orange paper boat",
                "source_run_id": loaded.json()["source_run_id"],
                "source_snapshot_sha256": loaded.json()["source_snapshot_sha256"],
                "idempotency_key": "edit-an-imported-version",
            },
        )
        assert queued.status_code == 202, queued.text
    # Source snapshots must travel through the existing portability boundary.
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert not has_local_path(manifest)


def _rewrite(content: bytes, manifest: dict[str, object]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(output, "w") as target:
        for entry in source.infolist():
            target.writestr(
                entry,
                json.dumps(manifest) if entry.filename == "manifest.json" else source.read(entry),
            )
    return output.getvalue()


@pytest.mark.parametrize(
    "corruption", ["digest", "text_digest", "install", "trusted_workflow", "artifact", "source_run"]
)
async def test_import_refuses_tampered_accepted_context_before_creating_a_project(
    app: FastAPI, client: AsyncClient, corruption: str
) -> None:
    from sqlalchemy import func

    from local_lm.models import Project

    workflow_id = None
    if corruption == "trusted_workflow":
        workflow = await client.post(
            "/api/workflows",
            json={
                "name": "Portable scene graph",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"1": {"class_type": "MockScene", "inputs": {}}},
            },
        )
        assert workflow.status_code == 201, workflow.text
        workflow_id = workflow.json()["current_revision_id"]
        from local_lm.models import WorkflowRevision

        with SessionLocal() as session:
            revision = session.get(WorkflowRevision, workflow_id)
            assert revision is not None
            revision.trusted = True
            session.commit()
    source, edited, content = await _archive(
        app,
        client,
        complete=True,
        output_count=3,
        workflow_revision_id=workflow_id,
    )
    from local_lm.accepted_turn_context import _digest

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    record = next(
        record
        for record in manifest["accepted_contexts"]
        if record["payload_json"]["run_id"] == edited["run"]["id"]
    )
    payload = record["payload_json"]
    if corruption == "digest":
        payload["context_limit"] += 1
    elif corruption == "text_digest":
        payload["messages"][0]["content"] = "A different constructed scene"
    elif corruption == "install":
        profile = payload["profile"]
        assert profile is not None
        profile["install"] = {
            "id": "old-install",
            "name": "Archive model",
            "role": profile["role"],
            "engine": profile["engine"],
            "local_path": "portable-model.bin",
            "manifest_json": {},
            "source_id": None,
            "source_provenance": None,
            "shared_package_binding_id": None,
        }
    elif corruption == "trusted_workflow":
        assert payload["workflow"] is not None
        payload["workflow"]["trusted"] = True
    elif corruption == "artifact":
        payload["artifact_ids"].append("sha256:" + "f" * 64)
    else:
        payload["source_run_id"] = edited["run"]["id"]
    if corruption != "digest":
        record["sha256"] = _digest(payload)
        run_record = next(run for run in manifest["runs"] if run["id"] == edited["run"]["id"])
        run_record["provenance_json"]["accepted_context_sha256"] = record["sha256"]
    with SessionLocal() as session:
        before = session.scalar(select(func.count()).select_from(Project))
    response = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "tampered-context.lm-atelier.zip",
                _rewrite(content, manifest),
                "application/zip",
            ),
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "project-import-invalid"
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Project)) == before


@pytest.mark.parametrize("output_count", [1, 3], ids=["text", "media"])
async def test_project_round_trip_preserves_removed_source_and_redacts_its_snapshot_text(
    app: FastAPI, client: AsyncClient, output_count: int
) -> None:
    _source, source, _content = await _archive(
        app, client, complete=True, output_count=output_count
    )
    from local_lm.accepted_turn_context import accepted_context
    from local_lm.models import Message, RunContextSnapshot

    source_id = source["user_message"]["id"]
    preview = await client.get(f"/api/messages/{source_id}/removal-impact")
    assert preview.status_code == 200, preview.text
    removal = await client.post(
        f"/api/messages/{source_id}/remove-content",
        json={
            "expected_message_id": source_id,
            "expected_revision_id": preview.json()["message_revision_id"],
            "operation_key": "remove-portable-source",
        },
    )
    assert removal.status_code == 200, removal.text
    with SessionLocal() as session:
        chat = session.get(Chat, source["run"]["chat_id"])
        assert chat is not None
        project_id = chat.project_id
    exported = await client.post(f"/api/projects/{project_id}/export")
    assert exported.status_code == 201, exported.text
    content = (await client.get(exported.json()["url"])).content
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    record = next(
        record
        for record in manifest["accepted_contexts"]
        if record["payload_json"]["run_id"] == source["run"]["id"]
    )
    assert record["payload_json"]["unavailable_reason"] == "removed_context"
    assert record["payload_json"]["compiled_prompt"] is None
    assert record["payload_json"]["standalone_prompt"] == ""
    assert record["payload_json"]["media_prompt"] == ""
    assert all(
        not item["content"]
        for item in record["payload_json"]["messages"]
        if item["source_message_id"] == source_id
    )
    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": ("removed-context.lm-atelier.zip", content, "application/zip"),
        },
    )
    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        chat = session.scalar(select(Chat).where(Chat.project_id == imported.json()["id"]))
        assert chat is not None
        run = next(
            run
            for run in session.scalars(select(Run).where(Run.chat_id == chat.id))
            if run.provenance_json["imported_from_run_id"] == source["run"]["id"]
        )
        message = session.get(Message, run.user_message_id)
        assert message is not None and message.content_removed_at is not None and not message.parts
        assert session.get(RunContextSnapshot, run.id) is not None
        with pytest.raises(ValueError, match="Accepted conversation context is unavailable"):
            accepted_context(session, run)
        imported_source_id = message.id
    loaded = await client.get(f"/api/messages/{imported_source_id}/edit-source")
    assert loaded.status_code == 409, loaded.text


@pytest.mark.parametrize("include_media", [True, False], ids=["with-media", "without-media"])
async def test_portable_context_retains_input_media_or_stays_explicitly_unavailable(
    app: FastAPI, client: AsyncClient, tmp_path: Path, include_media: bool
) -> None:
    from httpx2 import ASGITransport

    from local_lm.config import Settings
    from local_lm.main import create_app

    _source, edited, _content = await _archive(app, client, complete=True, output_count=3)
    from local_lm.accepted_turn_context import accepted_context
    from local_lm.models import RunContextArtifact, RunContextSnapshot

    with SessionLocal() as session:
        source_run = session.get(Run, edited["run"]["id"])
        assert source_run is not None
        input_id = source_run.provenance_json["outputs"][0]["artifact_id"]
        chat = session.get(Chat, source_run.chat_id)
        assert chat is not None
        project_id = chat.project_id
    async with app.state.services.scheduler.lease("primary"):
        queued = await client.post(
            f"/api/messages/{edited['user_message']['id']}/edits",
            json={
                "text": "Recolor the boat orange",
                "mode": "image",
                "source_run_id": edited["run"]["id"],
                "input_artifact_ids": [input_id],
                "workflow_revision_id": None,
                "idempotency_key": "portable-context-input",
            },
        )
        assert queued.status_code == 202, queued.text
        exported = await client.post(
            f"/api/projects/{project_id}/export", params={"include_media": include_media}
        )
        assert exported.status_code == 201, exported.text
        content = (await client.get(exported.json()["url"])).content
    deadline = asyncio.get_running_loop().time() + 10
    while asyncio.get_running_loop().time() < deadline:
        finished = await client.get(f"/api/runs/{queued.json()['run']['id']}")
        if finished.json()["status"] in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.03)
    else:
        raise AssertionError("Constructed source work did not finish before opening the target")
    target_app = create_app(
        Settings(
            data_dir=tmp_path / "context-target", dev=True, chat_engine="mock", media_engine="mock"
        )
    )
    async with (
        target_app.router.lifespan_context(target_app),
        AsyncClient(
            transport=ASGITransport(app=target_app),
            base_url="http://testserver",
        ) as target,
    ):
        auth = await target.post("/api/session")
        target.headers["x-local-lm-csrf"] = auth.json()["csrf_token"]
        imported = await target.post(
            "/api/projects/import",
            files={
                "archive": ("context-media.lm-atelier.zip", content, "application/zip"),
            },
        )
        assert imported.status_code == 201, imported.text
        with SessionLocal() as session:
            chat = session.scalar(select(Chat).where(Chat.project_id == imported.json()["id"]))
            assert chat is not None
            run = next(
                run
                for run in session.scalars(select(Run).where(Run.chat_id == chat.id))
                if run.provenance_json["imported_from_run_id"] == queued.json()["run"]["id"]
            )
            row = session.get(RunContextSnapshot, run.id)
            assert row is not None
            retained = set(
                session.scalars(
                    select(RunContextArtifact.artifact_id).where(
                        RunContextArtifact.run_id == run.id
                    )
                )
            )
            if include_media:
                context = accepted_context(session, run)
                assert context is not None and context.input_artifact_ids
                assert set(context.input_artifact_ids) <= retained
            else:
                assert row.payload_json["unavailable_reason"] == "imported_media_unavailable"
                assert retained == set()
                with pytest.raises(
                    ValueError, match="Accepted conversation context is unavailable"
                ):
                    accepted_context(session, run)
