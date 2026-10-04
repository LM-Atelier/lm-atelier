"""Studio admission and source Trash must serialize across database connections."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_chat_recovery import _chat, _command, _history, _impact
from test_studio_uploads import _PNG, _upload

from local_lm import api as api_module
from local_lm.chat_recovery import preview_chat_trash, trash_chat
from local_lm.db import SessionLocal
from local_lm.models import Chat, Job
from local_lm.recovery_previews import RecoveryPreviewConflict, reserve_recovery_write
from local_lm.recovery_v1 import TrashChatV1
from local_lm.studio_sessions import find_studio_session


@pytest.mark.parametrize("existing", [False, True])
async def test_studio_and_source_trash_cannot_both_admit_from_a_stale_read(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    chat_id = await _chat(client)
    assert (
        await client.patch(
            f"/api/chats/{chat_id}",
            json={
                "generation_settings_json": {"image": {"seed": 17}},
            },
        )
    ).status_code == 200
    uploaded = await _upload(client, "garden.png", "image/png", _PNG)
    payload = {"source_artifact_id": uploaded["id"], "source_chat_id": chat_id}
    studio_id = None
    if existing:
        opened = await client.post("/api/studio/sessions", json=payload)
        assert opened.status_code == 200
        studio_id = opened.json()["id"]
    before = _history(chat_id)
    with SessionLocal() as session:
        chat_count = session.scalar(select(func.count()).select_from(Chat))
        job_count = session.scalar(select(func.count()).select_from(Job))

    collaborator: object = getattr(api_module, "find_studio_session", None)
    assert collaborator is find_studio_session
    original = find_studio_session
    outcomes: list[str] = []

    def trash_from_another_connection(session: Session, artifact_id: str) -> Chat | None:
        with SessionLocal() as contender:
            contender.connection().exec_driver_sql("PRAGMA busy_timeout=0")
            now = datetime.now(UTC)
            try:
                impact = preview_chat_trash(contender, chat_id, now)
                trash_chat(
                    contender,
                    chat_id,
                    TrashChatV1.model_validate(
                        _command(impact.model_dump(), "trash-concurrent-studio-source")
                    ),
                    now,
                )
                contender.commit()
                outcomes.append("trashed")
            except RecoveryPreviewConflict as error:
                assert str(error) == "recovery-database-busy"
                contender.rollback()
                outcomes.append("writer-reserved")
            finally:
                contender.connection().exec_driver_sql("PRAGMA busy_timeout=5000")
        return original(session, artifact_id)

    monkeypatch.setattr(api_module, "find_studio_session", trash_from_another_connection)
    opened = await client.post("/api/studio/sessions", json=payload)
    monkeypatch.setattr(api_module, "find_studio_session", original)
    assert len(outcomes) == 1
    if outcomes == ["trashed"]:
        assert opened.status_code == 404, opened.text
        assert opened.json()["code"] == "chat-not-found"
        with SessionLocal() as session:
            assert session.scalar(select(func.count()).select_from(Chat)) == chat_count
    else:
        assert outcomes == ["writer-reserved"]
        assert opened.status_code == 200, opened.text
        assert opened.json()["generation_settings_json"]["image"] == {"seed": 17}
        if studio_id:
            assert opened.json()["id"] == studio_id
        preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
        trashed = await client.post(
            f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-after-studio-admission")
        )
        assert trashed.status_code == 200, trashed.text
        assert (await client.get(f"/api/studio/sessions/{opened.json()['id']}")).status_code == 200
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == job_count
    assert (await client.get(f"/api/artifacts/{uploaded['id']}/content")).content == _PNG


async def test_studio_writer_contention_refuses_without_partial_session_or_job(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client)
    uploaded = await _upload(client, "garden.png", "image/png", _PNG)
    before = _history(chat_id)
    with SessionLocal() as session:
        chat_count = session.scalar(select(func.count()).select_from(Chat))
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
        refused = await client.post(
            "/api/studio/sessions",
            json={
                "source_artifact_id": uploaded["id"],
                "source_chat_id": chat_id,
            },
        )
        assert refused.status_code == 409, refused.text
        assert refused.json() == {
            "code": "studio-session-busy",
            "detail": "The studio session could not be opened safely. Try again.",
        }
        writer.rollback()
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Chat)) == chat_count
        assert session.scalar(select(func.count()).select_from(Job)) == job_count
    assert (await client.get(f"/api/artifacts/{uploaded['id']}/content")).content == _PNG
