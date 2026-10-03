"""A materialized recovery selection changes every member or none of them."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import delete, select, update
from test_chat_deletion import _text_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_media_recovery_api import _media
from test_project_recovery_api import _project
from test_recovery_expiry import manual_expiry as manual_expiry
from test_workflow_recovery_api import _seed as _workflow

from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Project,
    RecoveryItem,
    RecoveryOperation,
    WorkflowFamily,
)
from local_lm.recovery_previews import RecoveryPreviewConflict


async def _trash(client: AsyncClient, path: str, key: str) -> dict:
    preview = await _impact(client, path + "/deletion-impact")
    response = await client.post(path + "/trash", json=_command(preview, key))
    assert response.status_code == 200, response.text
    return response.json()


async def _batch(client: AsyncClient, items: list[dict], action: str, **options) -> dict:
    response = await client.post(
        "/api/recovery-items/batches",
        json={
            "deletion_ids": [item["deletion_id"] for item in items],
            "action": action,
            **options,
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    preview = response.json()
    assert preview["policy"] == "all-or-nothing"
    assert {member["deletion_id"] for member in preview["items"]} == {
        item["deletion_id"] for item in items
    }
    return preview


async def _apply(client: AsyncClient, preview: dict, key: str, **changes):
    command = _command(preview, key)
    if preview["action"] == "purge":
        command["acknowledgement"] = "permanently-delete"
    return await client.post(
        f"/api/recovery-items/batches/{preview['batch_id']}/apply", json=command | changes
    )


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_a_mixed_batch_keeps_shared_bytes_and_exact_replay_with_one_commit(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    project_id, chat_id = await _project(client)
    history = _history(chat_id)
    entry_id, artifact_id = _media(app)
    family_id, definition_id, revision_id, graph = _workflow()
    items = [
        await _trash(client, f"/api/projects/{project_id}", "trash-batch-project"),
        await _trash(client, f"/api/chats/{chat_id}", "trash-batch-chat"),
        await _trash(client, f"/api/artifact-library/{entry_id}", "trash-batch-media"),
        await _trash(client, f"/api/workflow-families/{family_id}", "trash-batch-workflow"),
    ]
    preview = await _batch(client, items, action)
    assert preview["available"] is True
    assert len(preview["items"]) == 4
    observed: list[str] = []
    publish = app.state.services.events.publish

    async def observe(event_type, entity_id=None, payload=None):
        if event_type == "recovery.updated":
            with SessionLocal() as session:
                states = [session.get(RecoveryItem, item["deletion_id"]) for item in items]
                assert (
                    all(row is None for row in states)
                    if action == "restore"
                    else all(row is not None and row.state == "purged" for row in states)
                )
            assert payload == {}
            observed.append(entity_id)
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observe)
    response = await _apply(client, preview, "apply-mixed-batch")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["policy"] == "all-or-nothing" and result["reclaimed_bytes"] == 0
    assert {member["deletion_id"] for member in result["results"]} == {
        item["deletion_id"] for item in items
    }
    assert observed
    replay = await _apply(client, preview, "apply-mixed-batch")
    assert replay.status_code == 200 and replay.json() == result
    assert (await _apply(client, preview, "different-batch-key")).status_code == 409
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        assert app.state.services.artifacts.verified_path(artifact).is_file()
        if action == "restore":
            assert session.get(ArtifactLibraryEntry, entry_id).favorite
            family = session.get(WorkflowFamily, family_id)
            assert family is not None and not family.enabled
    if action == "restore":
        assert _history(chat_id) == history
        detail = (await client.get(f"/api/workflows/{definition_id}")).json()
        assert (
            detail["current_revision_id"] == revision_id
            and detail["revisions"][0]["api_graph_json"] == graph
        )


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_a_changed_later_member_refuses_the_entire_batch(
    client: AsyncClient, action: str
) -> None:
    ids = [await _chat(client), await _chat(client)]
    items = [await _trash(client, f"/api/chats/{chat_id}", "trash-" + chat_id) for chat_id in ids]
    preview = await _batch(client, items, action)
    single = await _impact(client, f"/api/recovery-items/{items[1]['deletion_id']}/impact")
    restored = await client.post(
        f"/api/recovery-items/{items[1]['deletion_id']}/restore",
        json=_command(single, "restore-one-member"),
    )
    assert restored.status_code == 200
    response = await _apply(client, preview, "refuse-stale-batch")
    assert response.status_code == 409
    assert (await client.get(f"/api/chats/{ids[0]}")).status_code == 404
    with SessionLocal() as session:
        assert session.get(RecoveryItem, items[0]["deletion_id"]).state == "recoverable"
        assert not session.scalars(
            select(RecoveryOperation).where(RecoveryOperation.operation_key.startswith("batch_"))
        ).all()


async def test_expiry_refuses_every_restore_without_extending_the_original_clock(
    client: AsyncClient,
) -> None:
    ids = [await _chat(client), await _chat(client)]
    items = [await _trash(client, f"/api/chats/{chat_id}", "trash-" + chat_id) for chat_id in ids]
    preview = await _batch(client, items, "restore")
    expired = datetime.now(UTC) - timedelta(seconds=1)
    with SessionLocal() as session:
        session.execute(
            update(RecoveryItem)
            .where(RecoveryItem.deletion_id == items[1]["deletion_id"])
            .values(purge_after=expired)
        )
        session.commit()
    response = await _apply(client, preview, "restore-expired-batch")
    assert response.status_code == 409
    with SessionLocal() as session:
        assert all(session.get(RecoveryItem, item["deletion_id"]) is not None for item in items)
        assert (
            session.get(RecoveryItem, items[1]["deletion_id"]).purge_after.replace(tzinfo=UTC)
            == expired
        )


async def test_missing_project_requires_an_explicit_materialized_unfiled_choice(
    client: AsyncClient,
) -> None:
    project_id, chat_id = await _project(client)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-missing-project-chat")
    with SessionLocal() as session:
        session.execute(delete(Project).where(Project.id == project_id))
        session.commit()
    preview = await _batch(client, [item], "restore")
    assert preview["available"] is False
    assert (await _apply(client, preview, "restore-without-choice")).status_code == 409
    accepted = await _batch(client, [item], "restore", restore_unfiled=True)
    assert accepted["available"] is True
    response = await _apply(client, accepted, "restore-unfiled-batch")
    assert response.status_code == 200
    assert (await client.get(f"/api/chats/{chat_id}/metadata")).json()["project_id"] is None


@pytest.mark.parametrize("selection", [[], ["same", "same"], ["absent"] * 21])
async def test_batches_refuse_unbounded_empty_or_duplicate_selection(
    client: AsyncClient, selection: list[str]
) -> None:
    response = await client.post(
        "/api/recovery-items/batches", json={"deletion_ids": selection, "action": "purge"}
    )
    assert response.status_code == 422


async def test_purge_requires_exact_acknowledgement_and_does_not_accept_a_different_selection(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-acknowledged-batch")
    preview = await _batch(client, [item], "purge")
    response = await client.post(
        f"/api/recovery-items/batches/{preview['batch_id']}/apply",
        json=_command(preview, "unacknowledged-batch"),
    )
    assert response.status_code == 422
    response = await _apply(client, preview, "change-selection-batch", deletion_ids=[])
    assert response.status_code == 422
    with SessionLocal() as session:
        assert (
            session.get(Chat, chat_id) is not None
            and session.get(RecoveryItem, item["deletion_id"]).state == "recoverable"
        )


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_failure_after_real_member_transitions_rolls_back_every_row_and_receipt(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    ids = [await _chat(client), await _chat(client)]
    for chat_id in ids:
        await _text_exchange(client, chat_id, "Keep the garden spacing")
    items = [await _trash(client, f"/api/chats/{chat_id}", "trash-" + chat_id) for chat_id in ids]
    preview = await _batch(client, items, action)
    from local_lm import recovery_bulk

    tables = (
        "chats",
        "messages",
        "message_parts",
        "runs",
        "jobs",
        "work_plans",
        "work_steps",
        "recovery_items",
        "recovery_operations",
        "recovery_previews",
        "recovery_batches",
    )

    def rows():
        with SessionLocal() as session:
            return {
                table: sorted(
                    tuple(row)
                    for row in session.connection().exec_driver_sql(f"SELECT * FROM {table}")
                )
                for table in tables
            }

    before = rows()
    transition = recovery_bulk._transition
    calls = 0

    def fail_after_transition(*args, **kwargs):
        nonlocal calls
        value = transition(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RecoveryPreviewConflict("recovery-batch-unavailable")
        return value

    events = []
    publish = app.state.services.events.publish

    async def observe(*args, **kwargs):
        events.append(args)
        return await publish(*args, **kwargs)

    monkeypatch.setattr(recovery_bulk, "_transition", fail_after_transition)
    monkeypatch.setattr(app.state.services.events, "publish", observe)
    response = await _apply(client, preview, "fail-after-real-transitions")
    assert response.status_code == 409 and calls == 2
    assert rows() == before
    assert events == []


async def test_a_completed_batch_replays_after_preview_expiry_and_a_new_database_session(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-durable-batch")
    preview = await _batch(client, [item], "purge")
    response = await _apply(client, preview, "durable-batch-replay")
    assert response.status_code == 200
    from local_lm.models import RecoveryBatchRecord

    with SessionLocal() as session:
        session.execute(
            update(RecoveryBatchRecord)
            .where(RecoveryBatchRecord.id == preview["batch_id"])
            .values(expires_at=datetime.now(UTC) - timedelta(days=1))
        )
        session.commit()
    replay = await _apply(client, preview, "durable-batch-replay")
    assert replay.status_code == 200 and replay.json() == response.json()


@pytest.mark.parametrize("change", ["digest", "fingerprints", "expired"])
async def test_a_corrupt_or_expired_materialized_selection_is_inert(
    client: AsyncClient, change: str
) -> None:
    chat_id = await _chat(client)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-invalid-materialization")
    preview = await _batch(client, [item], "restore")
    from local_lm.models import RecoveryBatchRecord

    with SessionLocal() as session:
        batch = session.get(RecoveryBatchRecord, preview["batch_id"])
        assert batch is not None
        if change == "digest":
            batch.preview_json = batch.preview_json | {"restore_unfiled": True}
        elif change == "fingerprints":
            batch.fingerprints_json = {}
        else:
            batch.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()
    response = await _apply(client, preview, "invalid-materialization")
    assert response.status_code == 409
    with SessionLocal() as session:
        assert session.get(RecoveryItem, item["deletion_id"]).state == "recoverable"
        assert session.get(RecoveryBatchRecord, preview["batch_id"]).operation_key is None
