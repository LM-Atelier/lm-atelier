"""A create retried with the same key returns the chat the first one made.

A create whose answer was lost, retried by a script, used to make a second
chat. A key on the request now names the chat it made: a retry returns that
chat and says it is a replay, whatever else the retry asks, and only a chat
deleted for good gives the key back.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from httpx2 import AsyncClient
from recovery_requests import permanently_delete_chat
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_chat_recovery import _command, _impact

from local_lm import api
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import SessionLocal
from local_lm.models import Chat, ChatCreationKey


async def _create(client: AsyncClient, **body: Any) -> Any:
    return await client.post("/api/chats", json={"title": "Keyed chat", **body})


def _chats_titled(title: str) -> int:
    with SessionLocal() as session:
        return int(session.scalar(select(func.count()).where(Chat.title == title)) or 0)


async def test_a_retried_create_returns_the_chat_the_key_made(client: AsyncClient) -> None:
    first = await _create(client, idempotency_key="script-item-1")
    again = await _create(client, idempotency_key="script-item-1")

    assert first.status_code == 201, first.text
    assert first.json()["replayed"] is False
    assert again.status_code == 200, again.text
    assert again.json()["replayed"] is True
    assert again.json()["id"] == first.json()["id"]
    assert _chats_titled("Keyed chat") == 1


async def test_a_replay_answers_with_the_first_chat_whatever_the_retry_asks(
    client: AsyncClient,
) -> None:
    first = await _create(client, idempotency_key="script-item-2")
    again = await client.post(
        "/api/chats",
        json={
            "title": "Another title",
            "routing_mode": "image",
            "idempotency_key": "script-item-2",
        },
    )

    assert again.status_code == 200, again.text
    assert again.json()["replayed"] is True
    assert again.json()["id"] == first.json()["id"]
    assert again.json()["title"] == "Keyed chat"
    assert _chats_titled("Another title") == 0


async def test_other_keys_and_no_key_make_new_chats(client: AsyncClient) -> None:
    keyed = await _create(client, idempotency_key="script-item-3")
    other = await _create(client, idempotency_key="script-item-4")
    plain = [await _create(client), await _create(client)]

    ids = {keyed.json()["id"], other.json()["id"], *(item.json()["id"] for item in plain)}
    assert len(ids) == 4
    assert all(item.status_code == 201 and item.json()["replayed"] is False for item in plain)


async def _trash(client: AsyncClient, chat_id: str) -> str:
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(impact, f"trash-{chat_id}")
    )
    assert trashed.status_code == 200, trashed.text
    return str(trashed.json()["deletion_id"])


async def test_a_key_whose_chat_is_in_recently_deleted_says_so(client: AsyncClient) -> None:
    first = await _create(client, idempotency_key="script-item-5")
    deletion_id = await _trash(client, first.json()["id"])

    refused = await _create(client, idempotency_key="script-item-5")
    impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    restored = await client.post(
        f"/api/recovery-items/{deletion_id}/restore", json=_command(impact, "restore-keyed")
    )
    again = await _create(client, idempotency_key="script-item-5")

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "chat-creation-key-deleted"
    assert restored.status_code == 200, restored.text
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"], again.text
    assert _chats_titled("Keyed chat") == 1


async def test_a_key_whose_chat_was_deleted_for_good_makes_a_new_one(
    client: AsyncClient,
) -> None:
    first = await _create(client, idempotency_key="script-item-6")
    await permanently_delete_chat(client, first.json()["id"])

    remade = await _create(client, idempotency_key="script-item-6")
    again = await _create(client, idempotency_key="script-item-6")

    assert remade.status_code == 201 and remade.json()["replayed"] is False, remade.text
    assert remade.json()["id"] != first.json()["id"]
    assert again.status_code == 200 and again.json()["id"] == remade.json()["id"], again.text
    with SessionLocal() as session:
        made = session.get(ChatCreationKey, "script-item-6")
        assert made is not None and made.chat_id == remade.json()["id"]


@pytest.mark.parametrize("history", ["a new key", "a key given back"])
async def test_two_creates_racing_on_one_key_make_one_chat(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, history: str
) -> None:
    """The later create looks up its key, then waits while the first one finishes.

    Its own lookup runs for real, so what that read leaves behind in its
    session is still there when its write finds the key taken.
    """

    key = f"script-race-{history.replace(' ', '-')}"
    if history == "a key given back":
        gone = await _create(client, idempotency_key=key)
        await permanently_delete_chat(client, gone.json()["id"])
    look_up = api._chat_a_key_made
    found: list[Any] = []

    def looking_up(session: Session, value: str) -> Any:
        found.append(look_up(session, value))
        return found[-1]

    check = api._validate_generation_defaults
    waiting = asyncio.Event()
    go_on = asyncio.Event()

    async def the_first_check_waits(request: Any, session: Session, values: Any) -> None:
        if not waiting.is_set():
            waiting.set()
            await go_on.wait()
        await check(request, session, values)

    monkeypatch.setattr(api, "_chat_a_key_made", looking_up)
    monkeypatch.setattr(api, "_validate_generation_defaults", the_first_check_waits)
    before = _chats_titled("Keyed chat")

    later = asyncio.create_task(_create(client, idempotency_key=key))
    await asyncio.wait_for(waiting.wait(), 10)
    first = await _create(client, idempotency_key=key)
    go_on.set()
    answer = await asyncio.wait_for(later, 10)

    assert found[0] is None, "the later create looked before the first made its chat"
    assert first.status_code == 201, first.text
    assert answer.status_code == 200, answer.text
    assert answer.json()["replayed"] is True
    assert answer.json()["id"] == first.json()["id"]
    assert _chats_titled("Keyed chat") == before + 1


def _a_late_chat(session: Session) -> str:
    """The chat a create that lost the race had written before it reached the key."""

    late = Chat(title="Raced chat")
    session.add(late)
    session.flush()
    return late.id


async def test_a_create_that_lost_the_key_makes_no_chat_of_its_own(client: AsyncClient) -> None:
    winner = await _create(client, idempotency_key="script-race-written")

    with SessionLocal() as session:
        late = _a_late_chat(session)
        with pytest.raises(api._CreationKeyTaken):
            api._remember_creation_key(session, "script-race-written", late)
        session.rollback()

    assert _chats_titled("Raced chat") == 0
    with SessionLocal() as session:
        made = session.get(ChatCreationKey, "script-race-written")
        assert made is not None and made.chat_id == winner.json()["id"]


@pytest.mark.parametrize("read", ["read again", "read stale"])
async def test_a_key_given_back_goes_to_one_create_only(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, read: str
) -> None:
    """The later create read the key while it still named the chat deleted for good."""

    gone = await _create(client, idempotency_key="script-race-given-back")
    await permanently_delete_chat(client, gone.json()["id"])
    with SessionLocal() as session:
        seen = session.get(ChatCreationKey, "script-race-given-back")
        assert seen is not None and seen.chat_id == gone.json()["id"]
        winner = await _create(client, idempotency_key="script-race-given-back")
        assert winner.status_code == 201, winner.text
        late = _a_late_chat(session)
        if read == "read stale":
            # Even a read that missed the winner cannot take the key from it.
            monkeypatch.setattr(session, "scalar", lambda *_args, **_kwargs: gone.json()["id"])
        with pytest.raises(api._CreationKeyTaken):
            api._remember_creation_key(session, "script-race-given-back", late)
        session.rollback()

    with SessionLocal() as session:
        made = session.get(ChatCreationKey, "script-race-given-back")
        assert made is not None and made.chat_id == winner.json()["id"]
    assert _chats_titled("Raced chat") == 0


async def test_a_replay_answers_although_its_project_is_now_in_recently_deleted(
    client: AsyncClient,
) -> None:
    """The project went to Recently Deleted after the first create; its chat stays, unfiled."""

    project = await client.post("/api/projects", json={"name": "Keyed project"})
    assert project.status_code == 201, project.text
    project_id = project.json()["id"]
    first = await _create(client, idempotency_key="script-item-7", project_id=project_id)
    impact = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
    trashed = await client.post(
        f"/api/projects/{project_id}/trash", json=_command(impact, "trash-keyed-project")
    )
    assert trashed.status_code == 200, trashed.text

    plain = await _create(client, project_id=project_id)
    again = await _create(client, idempotency_key="script-item-7", project_id=project_id)
    opened = await client.get(f"/api/chats/{first.json()['id']}")

    assert first.status_code == 201 and first.json()["project_id"] == project_id, first.text
    assert plain.status_code == 404 and plain.json()["code"] == "project-not-found", plain.text
    assert again.status_code == 200 and again.json()["replayed"] is True, again.text
    assert again.json()["id"] == first.json()["id"]
    assert opened.status_code == 200, opened.text
    assert again.json()["project_id"] is None and opened.json()["project_id"] is None


@pytest.mark.parametrize(
    ("asks", "status", "code"),
    [
        (
            {"generation_settings_json": {"chat": {"_image_edit_strength_mode": "auto"}}},
            422,
            "strength-mode-invalid",
        ),
        (
            {"generation_preset_ids_json": {"chat": "no-such-preset"}},
            404,
            "generation-preset-not-found",
        ),
    ],
    ids=["settings a create refuses", "a preset that does not exist"],
)
async def test_a_replay_answers_before_the_settings_the_retry_asks_are_checked(
    client: AsyncClient, asks: dict[str, Any], status: int, code: str
) -> None:
    first = await _create(client, idempotency_key="script-item-8")
    plain = await _create(client, **asks)
    again = await _create(client, idempotency_key="script-item-8", **asks)

    assert plain.status_code == status and plain.json()["code"] == code, plain.text
    assert again.status_code == 200 and again.json()["replayed"] is True, again.text
    assert again.json()["id"] == first.json()["id"]
    assert _chats_titled("Keyed chat") == 1


@pytest.mark.parametrize("key", ["", "x" * 201, "has space", "-leading", "nul\u0000byte"])
async def test_a_key_must_be_a_short_plain_token(client: AsyncClient, key: str) -> None:
    refused = await _create(client, idempotency_key=key)

    assert refused.status_code == 422, refused.text
    assert _chats_titled("Keyed chat") == 0


def test_the_migration_adds_and_removes_only_the_key_table(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "b8d24f6a9c13")
    database = settings.state_dir / "local-lm.sqlite3"

    def schema() -> dict[str, str]:
        with sqlite3.connect(database) as connection:
            return {
                name: sql
                for name, sql in connection.execute(
                    "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"
                )
            }

    before = schema()
    command.upgrade(config, "d41c7a9e2b58")
    after = schema()
    command.downgrade(config, "b8d24f6a9c13")

    assert set(after) - set(before) == {"chat_creation_keys"}
    assert {name: after[name] for name in before} == before
    assert schema() == before
