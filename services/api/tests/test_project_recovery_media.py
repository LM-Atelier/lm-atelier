"""Project recovery preserves accepted media work and independently referenced bytes."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_explicit_revision_model_binding import _png
from test_project_recovery_api import _project
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.artifact_library import ensure_library_entry, set_library_favorite
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Message,
    MessagePart,
    ReferenceAsset,
    ReferenceSubject,
    Run,
    RunContextSnapshot,
)


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_queued_media_keeps_settings_inputs_and_outputs_through_project_recovery(
    app: FastAPI,
    client: AsyncClient,
    mode: str,
    action: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await app.state.retention_sweep
    project_id, chat_id = await _project(client)
    assert (
        await client.patch(
            f"/api/projects/{project_id}",
            json={
                "instructions": "Use evenly spaced garden beds",
                "generation_settings_json": {mode: {"seed": 17}},
            },
        )
    ).status_code == 200
    assert (
        await client.patch(
            f"/api/chats/{chat_id}",
            json={
                "vision_settings_json": {
                    "compile_visual_prompts": False,
                    "verify_image_edits": False,
                }
            },
        )
    ).status_code == 200
    store = app.state.services.artifacts
    content = _png()
    with SessionLocal() as session:
        artifact = store.ingest_bytes(
            session,
            content,
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            original_name="Garden layout",
        )
        entry = ensure_library_entry(session, artifact)
        assert entry is not None
        set_library_favorite(session, artifact, True)
        other = Chat(title="Independent garden reference", archived=True)
        subject = ReferenceSubject(
            name="Garden structure", mention_slug="garden-structure", kind="object"
        )
        session.add_all([other, subject])
        session.flush()
        message = Message(chat_id=other.id, role="assistant", status="complete")
        session.add(message)
        session.flush()
        session.add_all(
            [
                MessagePart(
                    message_id=message.id, position=0, type="image", artifact_id=artifact.id
                ),
                ReferenceAsset(
                    reference_subject_id=subject.id, artifact_id=artifact.id, purpose="other"
                ),
            ]
        )
        session.commit()
        input_id, entry_id, other_id = artifact.id, entry.id, other.id
        input_path = store.resolve(artifact)
    seen: list[MediaRequest] = []
    original = app.state.services.engines.media.generate

    async def observe(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        async for event in original(request):
            yield event

    monkeypatch.setattr(app.state.services.engines.media, "generate", observe)
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "Keep the garden paths wide",
                "mode": mode,
                "input_artifact_ids": [input_id],
            },
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None and run.settings_json["seed"] == 17
            assert session.get(RunContextSnapshot, run_id) is None
        preview = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
        trashed = await client.post(
            f"/api/projects/{project_id}/trash",
            json=_command(preview, "trash-queued-media-project"),
        )
        assert trashed.status_code == 200, trashed.text
        deletion_id = trashed.json()["deletion_id"]
        with SessionLocal() as session:
            row = session.get(RunContextSnapshot, run_id)
            assert row is not None and input_id in row.payload_json["input_artifact_ids"]
            snapshot = dict(row.payload_json)
        preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        command = _command(preview, f"{action}-queued-media-project")
        if action == "purge":
            command["acknowledgement"] = "permanently-delete"
        result = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
        assert result.status_code == 200 and result.json()["reclaimed_bytes"] == 0
        assert input_path.read_bytes() == content and seen == []
        with SessionLocal() as session:
            retained_snapshot = session.get(RunContextSnapshot, run_id)
            assert retained_snapshot is not None
            assert retained_snapshot.payload_json == snapshot
            retained_entry = session.get(ArtifactLibraryEntry, entry_id)
            assert retained_entry is not None
            assert retained_entry.favorite
            retained_chat = session.get(Chat, other_id)
            assert retained_chat is not None
            assert retained_chat.archived

    async def read() -> dict[str, Any]:
        payload: dict[str, Any] = (await client.get(f"/api/runs/{run_id}")).json()
        return payload

    completed = await wait_for_terminal_status(read, what="accepted garden media")
    assert seen and seen[-1].parameters["seed"] == 17
    output_ids = [output["artifact_id"] for output in completed["provenance_json"]["outputs"]]
    assert output_ids
    with SessionLocal() as session:
        outputs = list(session.scalars(select(Artifact).where(Artifact.id.in_(output_ids))))
        assert any(artifact.kind == mode for artifact in outputs)
        paths = [store.resolve(artifact) for artifact in outputs]
        original_bytes = [path.read_bytes() for path in paths]
        protected = {input_id, *output_ids}
        protected.update(
            poster
            for artifact in outputs
            if isinstance((poster := artifact.metadata_json.get("poster_artifact_id")), str)
        )
        temporaries = {
            artifact.id: artifact.size_bytes
            for artifact in session.scalars(select(Artifact))
            if artifact.id not in protected
            and artifact.metadata_json.get("temporary_preview") is True
        }
        retained = store.cleanup_retention(
            session,
            retention_days=1,
            temporary_hours=1,
            dry_run=False,
            now=datetime.now(UTC) + timedelta(days=120),
        )
        session.commit()
        assert retained.removed_count == len(temporaries)
        assert retained.reclaimed_bytes == sum(temporaries.values())
        assert all(session.get(Artifact, identity) is None for identity in temporaries)
        assert all(session.get(Artifact, identity) is not None for identity in protected)
        retained_snapshot = session.get(RunContextSnapshot, run_id)
        assert retained_snapshot is not None
        assert retained_snapshot.payload_json == snapshot
    assert [path.read_bytes() for path in paths] == original_bytes
    assert input_path.read_bytes() == content
    response = await client.get(f"/api/artifacts/{input_id}/content")
    assert response.status_code == 200 and response.content == content
