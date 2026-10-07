"""Public recovery excludes internal conversations even when a record names one."""

from datetime import UTC, datetime, timedelta

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_studio_uploads import _PNG, _upload

from local_lm.chat_recovery import (
    preview_chat_recovery,
    preview_chat_trash,
    purge_chat,
    restore_chat,
)
from local_lm.db import Base, SessionLocal
from local_lm.models import Chat, RecoveryItem
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE
from local_lm.recovery_operations import _remember
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryAction,
    RecoveryKind,
    RecoveryResultV1,
    RestoreRecoveryV1,
)


async def _internal_chat(client: AsyncClient, scope: str) -> str:
    if scope == "studio":
        uploaded = await _upload(client, "garden.png", "image/png", _PNG)
        opened = await client.post(
            "/api/studio/sessions", json={"source_artifact_id": uploaded["id"]}
        )
        assert opened.status_code == 200
        return str(opened.json()["id"])
    with SessionLocal() as session:
        chat = Chat(title="Garden helper", scope=scope)
        session.add(chat)
        session.commit()
        return chat.id


async def _internal(client: AsyncClient, scope: str) -> tuple[str, str]:
    now = datetime.now(UTC)
    chat_id = await _internal_chat(client, scope)
    with SessionLocal() as session:
        item = RecoveryItem(
            kind="chat",
            subject_id=chat_id,
            display_label="Garden helper",
            deleted_at=now,
            purge_after=now + timedelta(days=30),
            subject_revision="a" * 64,
        )
        session.add(item)
        session.commit()
        return chat_id, item.deletion_id


@pytest.mark.parametrize("scope", [PROMPT_HELPER_SCOPE, "studio"])
async def test_recovery_list_and_batch_preview_exclude_internal_chat_membership(
    client: AsyncClient,
    scope: str,
) -> None:
    created = await client.post("/api/chats", json={"title": "Garden notes"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    normal = await client.post(
        f"/api/chats/{chat_id}/trash",
        json=_command(impact, "trash-normal-namespace"),
    )
    assert normal.status_code == 200
    _chat_id, deletion_id = await _internal(client, scope)
    page = await client.get("/api/recovery-items", params={"kind": "chat", "limit": 1})
    assert page.status_code == 200 and page.json()["items"] == [normal.json()]
    response = await client.post(
        "/api/recovery-items/batches",
        json={
            "deletion_ids": [normal.json()["deletion_id"], deletion_id],
            "action": "restore",
            "restore_unfiled": False,
        },
    )
    assert response.status_code == 404
    assert response.json()["code"] == "recovery-item-not-found"
    assert deletion_id not in response.text
    with SessionLocal() as session:
        assert session.get(RecoveryItem, normal.json()["deletion_id"]) is not None
        assert session.get(RecoveryItem, deletion_id) is not None
        assert list(session.execute(select(Base.metadata.tables["recovery_batches"]))) == []


@pytest.mark.parametrize("scope", [PROMPT_HELPER_SCOPE, "studio"])
@pytest.mark.parametrize("action", ["impact", "restore", "purge"])
async def test_direct_recovery_identity_cannot_read_or_change_an_internal_chat(
    client: AsyncClient,
    scope: str,
    action: str,
) -> None:
    chat_id, deletion_id = await _internal(client, scope)
    with SessionLocal() as session:
        tables = ("chats", "recovery_items", "recovery_previews", "recovery_operations")
        before = {
            name: list(session.execute(select(Base.metadata.tables[name]))) for name in tables
        }
    path = f"/api/recovery-items/{deletion_id}/{action}"
    command = {
        "expected_revision": "a" * 64,
        "impact_sha256": "a" * 64,
        "operation_key": "change-internal-namespace",
    }
    if action == "impact":
        response = await client.get(path)
    else:
        if action == "purge":
            command["acknowledgement"] = "permanently-delete"
        response = await client.post(path, json=command)
    assert response.status_code == 404
    assert response.json()["code"] == "recovery-item-not-found"
    assert chat_id not in response.text and deletion_id not in response.text
    with SessionLocal() as session:
        for name in tables:
            assert list(session.execute(select(Base.metadata.tables[name]))) == before[name]


@pytest.mark.parametrize("scope", [PROMPT_HELPER_SCOPE, "studio"])
@pytest.mark.parametrize("action", ["deletion-impact", "trash"])
async def test_internal_chat_cannot_create_public_recovery_membership(
    client: AsyncClient,
    scope: str,
    action: str,
) -> None:
    chat_id = await _internal_chat(client, scope)
    path = f"/api/chats/{chat_id}/{action}"
    response = (
        await client.get(path)
        if action == "deletion-impact"
        else await client.post(
            path,
            json={
                "expected_revision": "a" * 64,
                "impact_sha256": "a" * 64,
                "operation_key": "trash-internal-namespace",
            },
        )
    )
    assert response.status_code == 404
    assert response.json()["code"] == "chat-not-found"
    with SessionLocal() as session:
        assert session.get(Chat, chat_id) is not None
        assert (
            session.scalar(select(RecoveryItem).where(RecoveryItem.subject_id == chat_id)) is None
        )


@pytest.mark.parametrize("scope", [PROMPT_HELPER_SCOPE, "studio"])
@pytest.mark.parametrize("action", ["trash-preview", "recovery-preview", "restore", "purge"])
async def test_internal_namespace_is_refused_inside_the_reserved_writer_and_cached_replay(
    client: AsyncClient, scope: str, action: str
) -> None:
    chat_id, deletion_id = await _internal(client, scope)
    now = datetime.now(UTC)
    with SessionLocal() as session:
        fields = {
            "expected_revision": "a" * 64,
            "impact_sha256": "a" * 64,
            "operation_key": "replay-internal-namespace",
        }
        if action in {"restore", "purge"}:
            recovery_action = (
                RecoveryAction.RESTORE if action == "restore" else RecoveryAction.PURGE
            )
            command = (
                RestoreRecoveryV1.model_validate(fields)
                if action == "restore"
                else PurgeRecoveryV1.model_validate(
                    {**fields, "acknowledgement": "permanently-delete"}
                )
            )
            _remember(
                session,
                recovery_action,
                command,
                RecoveryResultV1(
                    deletion_id=deletion_id,
                    kind=RecoveryKind.CHAT,
                    subject_id=chat_id,
                    action=recovery_action,
                    reclaimed_bytes=0,
                ),
            )
            item = session.get(RecoveryItem, deletion_id)
            assert item is not None
            session.delete(item)
            session.commit()
        tables = ("chats", "recovery_items", "recovery_previews", "recovery_operations")
        before = {
            name: list(session.execute(select(Base.metadata.tables[name]))) for name in tables
        }
        session.rollback()
        if action in {"restore", "purge"}:
            response = await client.post(
                f"/api/recovery-items/{deletion_id}/{action}",
                json=command.model_dump(mode="json"),
            )
            assert response.status_code == 404
            assert response.json()["code"] == "recovery-item-not-found"
            assert chat_id not in response.text and deletion_id not in response.text
        with pytest.raises(RecoveryPreviewConflict, match="^recovery-item-not-found$"):
            if action == "trash-preview":
                preview_chat_trash(session, chat_id, now)
            elif action == "recovery-preview":
                preview_chat_recovery(session, deletion_id, now)
            elif action == "restore":
                assert isinstance(command, RestoreRecoveryV1)
                restore_chat(session, deletion_id, command, now)
            else:
                assert isinstance(command, PurgeRecoveryV1)
                purge_chat(session, deletion_id, command, now)
        session.rollback()
        for name in tables:
            assert list(session.execute(select(Base.metadata.tables[name]))) == before[name]
