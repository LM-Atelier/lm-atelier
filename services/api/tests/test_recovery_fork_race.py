"""Fork admission and deletion serialize before source history and filing are copied."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_chat_deletion import _text_exchange
from test_chat_recovery import _command, _history, _impact
from test_media_recovery import CONTENT, _seed
from test_project_recovery_api import _project

from local_lm import api as api_module
from local_lm import chat_forking
from local_lm.chat_recovery import preview_chat_trash, trash_chat
from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Message, MessagePart
from local_lm.project_recovery import preview_project_trash, trash_project
from local_lm.recovery_previews import RecoveryPreviewConflict, reserve_recovery_write
from local_lm.recovery_v1 import RecoveryCommandV1, TrashChatV1


@pytest.mark.parametrize("deleted_kind", ["chat", "project"])
async def test_fork_does_not_copy_deleted_history_or_stale_project_filing(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, deleted_kind: str
) -> None:
    project_id, chat_id = await _project(client)
    exchange = await _text_exchange(client, chat_id, "Keep the garden paths clear")
    leaf_id = exchange["assistant_message"]["id"]
    with SessionLocal() as session:
        artifact, _entry, _collection, _tag = _seed(app.state.services.artifacts, session)
        artifact_id = artifact.id
        session.add(
            MessagePart(
                message_id=leaf_id,
                position=len(session.get(Message, leaf_id).parts),
                type="image",
                artifact_id=artifact_id,
            )
        )
        session.commit()
        chat_count = session.scalar(select(func.count()).select_from(Chat))
        job_count = session.scalar(select(func.count()).select_from(Job))
    before = _history(chat_id)
    original = chat_forking._fork_title
    outcomes: list[str] = []

    def delete_after_the_source_was_read(title: str) -> str:
        with SessionLocal() as contender:
            contender.connection().exec_driver_sql("PRAGMA busy_timeout=0")
            now = datetime.now(UTC)
            try:
                if deleted_kind == "chat":
                    preview = preview_chat_trash(contender, chat_id, now)
                    trash_chat(
                        contender,
                        chat_id,
                        TrashChatV1(**_command(preview.model_dump(), "trash-during-fork")),
                        now,
                    )
                else:
                    preview = preview_project_trash(contender, project_id, now)
                    trash_project(
                        contender,
                        project_id,
                        RecoveryCommandV1(**_command(preview.model_dump(), "trash-during-fork")),
                        now,
                    )
                contender.commit()
                outcomes.append("trashed")
            except RecoveryPreviewConflict as error:
                assert error.code == "recovery-database-busy"
                contender.rollback()
                outcomes.append("writer-reserved")
            finally:
                contender.connection().exec_driver_sql("PRAGMA busy_timeout=5000")
        return original(title)

    monkeypatch.setattr(chat_forking, "_fork_title", delete_after_the_source_was_read)
    response = await client.post(f"/api/messages/{leaf_id}/fork")
    monkeypatch.setattr(chat_forking, "_fork_title", original)
    assert len(outcomes) == 1
    if outcomes == ["trashed"] and deleted_kind == "chat":
        assert response.status_code == 404, response.text
        assert response.json()["code"] == "fork-source-not-found"
        with SessionLocal() as session:
            assert session.scalar(select(func.count()).select_from(Chat)) == chat_count
    else:
        assert response.status_code == 201, response.text
        fork_id = response.json()["id"]
        expected_project = None if outcomes == ["trashed"] else project_id
        with SessionLocal() as session:
            assert session.get(Chat, fork_id).project_id == expected_project
            copied = session.scalars(select(Message.id).where(Message.chat_id == fork_id)).all()
            assert len(copied) == 2
            assert (
                session.scalar(
                    select(MessagePart.artifact_id).where(
                        MessagePart.message_id.in_(copied), MessagePart.type == "image"
                    )
                )
                == artifact_id
            )
        if outcomes == ["writer-reserved"]:
            subject = chat_id if deleted_kind == "chat" else project_id
            path = f"/api/{'chats' if deleted_kind == 'chat' else 'projects'}/{subject}"
            preview = await _impact(client, f"{path}/deletion-impact")
            deleted = await client.post(f"{path}/trash", json=_command(preview, "trash-after-fork"))
            assert deleted.status_code == 200, deleted.text
        assert (await client.get(f"/api/chats/{fork_id}")).status_code == 200
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == job_count
    assert (await client.get(f"/api/artifacts/{artifact_id}/content")).content == CONTENT


async def test_fork_contention_refuses_without_copying_any_chat_history(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_id, chat_id = await _project(client)
    exchange = await _text_exchange(client, chat_id, "Keep the garden paths clear")
    leaf_id = exchange["assistant_message"]["id"]
    before = _history(chat_id)
    with SessionLocal() as session:
        chat_count = session.scalar(select(func.count()).select_from(Chat))
        message_count = session.scalar(select(func.count()).select_from(Message))
        job_count = session.scalar(select(func.count()).select_from(Job))

    async def zero_timeout_session() -> AsyncIterator[Session]:
        with SessionLocal() as session:
            session.connection().exec_driver_sql("PRAGMA busy_timeout=0")
            try:
                yield session
            finally:
                session.rollback()
                session.connection().exec_driver_sql("PRAGMA busy_timeout=5000")

    monkeypatch.setitem(
        app.dependency_overrides, api_module.get_conversation_session, zero_timeout_session
    )
    with SessionLocal() as writer:
        reserve_recovery_write(writer)
        refused = await client.post(f"/api/messages/{leaf_id}/fork")
        assert refused.status_code == 409, refused.text
        assert refused.json() == {
            "code": "fork-source-busy",
            "detail": "The conversation could not be forked safely. Try again.",
        }
        writer.rollback()
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Chat)) == chat_count
        assert session.scalar(select(func.count()).select_from(Message)) == message_count
        assert session.scalar(select(func.count()).select_from(Job)) == job_count
