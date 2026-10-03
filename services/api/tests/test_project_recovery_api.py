"""Project recovery routes preserve conversations and commit before live updates."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import func, select
from test_chat_deletion import _text_exchange
from test_chat_recovery import _command, _history, _impact
from test_media_recovery_restart import _open
from test_recovery_expiry import REAL_MAINTENANCE
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import main, recovery_maintenance
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    Job,
    Project,
    RecoveryItem,
    RecoveryOperation,
    RecoveryPreviewRecord,
)
from local_lm.project_recovery import preview_project_trash, trash_project
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import RecoveryCommandV1


async def _project(client: AsyncClient, name: str = "Garden layout") -> tuple[str, str]:
    project = await client.post("/api/projects", json={"name": name})
    assert project.status_code == 201, project.text
    project_id = project.json()["id"]
    chat = await client.post("/api/chats", json={"title": "Garden notes", "project_id": project_id})
    assert chat.status_code == 201, chat.text
    return project_id, chat.json()["id"]


def _stored_trash(project_id: str, now: datetime):
    with SessionLocal() as session:
        preview = preview_project_trash(session, project_id, now)
        item = trash_project(
            session,
            project_id,
            RecoveryCommandV1(**_command(preview.model_dump(), f"trash-{project_id}")),
            now,
        )
        session.commit()
        return item


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_project_routes_replay_exactly_and_keep_child_history_after_later_moves(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    project_id, chat_id = await _project(client)
    await _text_exchange(client, chat_id, "Keep the spacing between garden beds")
    live_id, moved_id = await _project(client, "Garden paths")
    assert (
        await client.patch(f"/api/chats/{moved_id}", json={"project_id": project_id})
    ).status_code == 200
    history = _history(chat_id)
    observed = []
    publish = app.state.services.events.publish

    async def observe(event_type, entity_id=None, payload=None):
        if event_type in ("project.updated", "recovery.updated"):
            assert payload == {}
            with SessionLocal() as session:
                item = session.scalar(
                    select(RecoveryItem).where(
                        RecoveryItem.kind == "project", RecoveryItem.subject_id == project_id
                    )
                )
                observed.append((event_type, entity_id, item.state if item else None))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observe)
    preview = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
    assert preview["kind"] == "project" and preview["counts"]["chats"] == 2
    command = _command(preview, "trash-garden-project")
    response = await client.post(f"/api/projects/{project_id}/trash", json=command)
    assert response.status_code == 200, response.text
    item = response.json()
    deletion_id = item["deletion_id"]
    assert item["subject_id"] == project_id and item["delete_generated_media"] is False
    assert observed == [
        ("project.updated", project_id, "recoverable"),
        ("recovery.updated", deletion_id, "recoverable"),
    ]
    repeated = await client.post(f"/api/projects/{project_id}/trash", json=command)
    assert repeated.status_code == 200 and repeated.json() == item
    assert _history(chat_id) == history
    assert (await client.get(f"/api/chats/{chat_id}/messages")).status_code == 200
    assert (
        await client.patch(f"/api/chats/{moved_id}", json={"project_id": live_id})
    ).status_code == 200
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    assert preview["counts"]["chats"] == 1
    command = _command(preview, f"{action}-garden-project")
    if action == "purge":
        command["acknowledgement"] = "permanently-delete"
    response = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["kind"] == "project" and result["reclaimed_bytes"] == 0
    repeated = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert repeated.status_code == 200 and repeated.json() == result
    state = None if action == "restore" else "purged"
    assert observed[-2:] == [
        ("project.updated", project_id, state),
        ("recovery.updated", deletion_id, state),
    ]
    expected = dict(history)
    expected["chat"] = (
        *history["chat"][:2],
        project_id if action == "restore" else None,
        *history["chat"][3:],
    )
    assert _history(chat_id) == expected
    with SessionLocal() as session:
        assert (session.get(Project, project_id) is not None) == (action == "restore")
        assert session.scalar(select(Chat.project_id).where(Chat.id == moved_id)) == live_id


@pytest.mark.parametrize("change", ["configuration", "filing"])
async def test_project_route_refuses_stale_impact_without_partial_changes(
    app: FastAPI, client: AsyncClient, change: str
) -> None:
    project_id, chat_id = await _project(client)
    preview = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
    if change == "configuration":
        changed = await client.patch(
            f"/api/projects/{project_id}", json={"instructions": "Keep the garden paths wide"}
        )
    else:
        changed = await client.patch(f"/api/chats/{chat_id}", json={"project_id": None})
    assert changed.status_code == 200
    sequence = app.state.services.events.sequence
    before = _history(chat_id)
    refused = await client.post(
        f"/api/projects/{project_id}/trash", json=_command(preview, "stale-project")
    )
    assert refused.status_code == 409 and refused.json()["code"] == "recovery-impact-stale"
    assert _history(chat_id) == before
    assert app.state.services.events.sequence == sequence
    with SessionLocal() as session:
        assert session.scalar(select(RecoveryItem)) is None
        assert session.scalar(select(RecoveryOperation)) is None


async def test_project_recovery_paging_filters_before_limiting_and_never_issues_previews(
    client: AsyncClient,
) -> None:
    now = datetime.now(UTC)
    items = []
    for number in range(3):
        project_id, _chat_id = await _project(client, f"Garden section {number}")
        items.append(_stored_trash(project_id, now + timedelta(seconds=number)))
    with SessionLocal() as session:
        previews = set(session.scalars(select(RecoveryPreviewRecord.revision)))
        session.add(
            RecoveryItem(
                kind="project",
                subject_id="missing-project",
                display_label="Gone",
                deleted_at=now + timedelta(days=1),
                purge_after=now + timedelta(days=31),
                state="recoverable",
                subject_revision="e" * 64,
            )
        )
        session.commit()
    response = await client.get("/api/recovery-items", params={"kind": "project", "limit": 2})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    first = response.json()
    assert [row["deletion_id"] for row in first["items"]] == [
        items[2].deletion_id,
        items[1].deletion_id,
    ]
    assert all(row["counts"]["chats"] == 1 for row in first["items"])
    cursor = first["next_cursor"]
    assert cursor is not None
    assert (
        await client.get("/api/recovery-items", params={"cursor": cursor, "kind": "chat"})
    ).status_code == 400
    last = await client.get("/api/recovery-items", params={"cursor": cursor, "kind": "project"})
    assert last.status_code == 200
    assert [row["deletion_id"] for row in last.json()["items"]] == [items[0].deletion_id]
    assert last.json()["next_cursor"] is None
    with SessionLocal() as session:
        assert set(session.scalars(select(RecoveryPreviewRecord.revision))) == previews


@pytest.mark.parametrize("fail", [False, True])
async def test_project_expiry_keeps_child_history_and_rolls_back_failed_cascades(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    await app.state.retention_sweep
    project_id, chat_id = await _project(client)
    await _text_exchange(client, chat_id, "Keep the garden paths clear")
    history = _history(chat_id)
    item = _stored_trash(project_id, datetime.now(UTC))
    with SessionLocal() as session:
        job_count = session.scalar(select(func.count()).select_from(Job))
    if fail:
        original = recovery_maintenance.purge_project

        def refuse(*args, **kwargs):
            original(*args, **kwargs)
            raise RecoveryPreviewConflict("recovery-impact-stale")

        monkeypatch.setattr(recovery_maintenance, "purge_project", refuse)
    sequence = app.state.services.events.sequence
    early = await recovery_maintenance.expire_recovery_batch(
        app.state.services, item.purge_after - timedelta(microseconds=1)
    )
    assert early.examined == early.purged == 0
    result = await recovery_maintenance.expire_recovery_batch(app.state.services, item.purge_after)
    assert result.examined == 1 and result.purged == int(not fail) and result.deferred == int(fail)
    expected = dict(history)
    if not fail:
        expected["chat"] = (*history["chat"][:2], None, *history["chat"][3:])
    assert _history(chat_id) == expected
    with SessionLocal() as session:
        retained = session.get(RecoveryItem, item.deletion_id)
        assert retained is not None and retained.state == ("recoverable" if fail else "purged")
        assert retained.purge_after.replace(tzinfo=UTC) == item.purge_after
        assert (session.get(Project, project_id) is not None) == fail
        assert session.scalar(select(func.count()).select_from(Job)) == job_count
        assert session.scalar(select(func.count()).select_from(RecoveryOperation)) == (
            1 if fail else 2
        )
    events = app.state.services.events.since(sequence)
    assert [(event.type, event.entity_id, event.payload) for event in events] == (
        []
        if fail
        else [("project.updated", project_id, {}), ("recovery.updated", item.deletion_id, {})]
    )


@pytest.mark.parametrize("expired", [False, True])
async def test_project_recovery_restart_keeps_the_original_deadline_and_child_history(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    async with _open(settings) as (app, client):
        await app.state.retention_sweep
        project_id, chat_id = await _project(client)
        await _text_exchange(client, chat_id, "Keep the garden spacing")
        history = _history(chat_id)
        item = _stored_trash(project_id, datetime.now(UTC) - timedelta(days=31 if expired else 3))
        with SessionLocal() as session:
            job_count = session.scalar(select(func.count()).select_from(Job))
    if expired:
        monkeypatch.setattr(main, "maintain_recovery_expiry", REAL_MAINTENANCE)
    async with _open(settings) as (app, client):
        if expired:

            async def read_state():
                with SessionLocal() as session:
                    return session.get(RecoveryItem, item.deletion_id).state

            await wait_until(read_state, lambda state: state == "purged", what="project expiry")
            expected = dict(history)
            expected["chat"] = (*history["chat"][:2], None, *history["chat"][3:])
            assert _history(chat_id) == expected
        else:
            assert _history(chat_id) == history
            assert (await client.get(f"/api/projects/{project_id}")).status_code == 404
            assert (await client.get(f"/api/chats/{chat_id}/metadata")).json()["project_id"] is None
            preview = await _impact(client, f"/api/recovery-items/{item.deletion_id}/impact")
            restored = await client.post(
                f"/api/recovery-items/{item.deletion_id}/restore",
                json=_command(preview, "restore-project-after-restart"),
            )
            assert restored.status_code == 200
            assert _history(chat_id) == history
        with SessionLocal() as session:
            if expired:
                retained = session.get(RecoveryItem, item.deletion_id)
                assert (
                    retained is not None
                    and retained.purge_after.replace(tzinfo=UTC) == item.purge_after
                )
            assert session.scalar(select(func.count()).select_from(Job)) == job_count
