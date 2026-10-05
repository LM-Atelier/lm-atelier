"""Picture purposes survive unsent drafts, accepted edits and database upgrades."""

import sqlite3
from typing import Any

import pytest
from alembic import command
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_chat_recovery import _command, _impact
from test_image_role_execution import selected_images

from local_lm.accepted_turn_context import accepted_context
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import SessionLocal
from local_lm.models import Run
from local_lm.prior_turn_edits import prepare_prior_turn_edit
from local_lm.schemas import PriorTurnEditRequest

ROLES = ["reference", "edit_source", "reference"]
PARENT = "9c4e7a2d1830"
REVISION = "e5a27b3d9061"


async def saved_draft(client: AsyncClient, roles: list[str] | None) -> tuple[str, list[str]]:
    ids = await selected_images(client)
    chat_id = (await client.post("/api/chats", json={"title": "Garden draft"})).json()["id"]
    response = await client.put(
        f"/api/chats/{chat_id}/composer-draft",
        json={
            "expected_revision": 0,
            "draft": {
                "text": "A garden",
                "mode": "image",
                "output_count": 1,
                "attachments": [
                    {
                        "artifact_id": artifact_id,
                        "kind": "image",
                        "origin": "uploaded",
                        **({"image_role": roles[index]} if roles else {}),
                    }
                    for index, artifact_id in enumerate(ids)
                ],
            },
        },
    )
    assert response.status_code == 200, response.text
    return chat_id, ids


async def test_draft_purposes_round_trip_in_order_and_can_return_to_automatic(
    client: AsyncClient,
) -> None:
    chat_id, ids = await saved_draft(client, ROLES)
    url = f"/api/chats/{chat_id}/composer-draft"
    read = (await client.get(url)).json()
    assert [item["artifact_id"] for item in read["attachments"]] == ids
    assert [item["image_role"] for item in read["attachments"]] == ROLES
    changed = await client.put(
        url,
        json={
            "expected_revision": 1,
            "draft": {
                "text": "A garden",
                "mode": "image",
                "attachments": [
                    {"artifact_id": artifact_id, "kind": "image", "origin": "uploaded"}
                    for artifact_id in ids
                ],
            },
        },
    )
    assert changed.status_code == 200, changed.text
    assert [item["image_role"] for item in (await client.get(url)).json()["attachments"]] == [
        None,
        None,
        None,
    ]


@pytest.mark.parametrize(
    "attachments",
    [
        [{"artifact_id": "clip", "kind": "video", "origin": "uploaded", "image_role": "reference"}],
        [
            {
                "artifact_id": value,
                "kind": "image",
                "origin": "uploaded",
                "image_role": "edit_source",
            }
            for value in ("first", "second")
        ],
    ],
)
async def test_invalid_draft_purposes_are_refused_before_replacing_a_saved_draft(
    client: AsyncClient, attachments: list[dict[str, str]]
) -> None:
    chat_id, _ = await saved_draft(client, ROLES)
    url = f"/api/chats/{chat_id}/composer-draft"
    before = (await client.get(url)).json()
    refused = await client.put(
        url,
        json={"expected_revision": 1, "draft": {"text": "A garden", "attachments": attachments}},
    )
    assert refused.status_code == 422
    assert (await client.get(url)).json() == before


async def test_recovery_preserves_purposes_and_refuses_bulk_changes_to_a_trashed_draft(
    client: AsyncClient, settings: Settings
) -> None:
    chat_id, _ = await saved_draft(client, ROLES)
    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-garden-purposes")
    )
    assert trashed.status_code == 200, trashed.text
    deletion_id = trashed.json()["deletion_id"]
    with (
        sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection,
        pytest.raises(sqlite3.IntegrityError, match="chat-recovery-write-refused"),
    ):
        connection.execute(
            "UPDATE chat_composer_draft_attachments SET image_role='reference' "
            "WHERE chat_id=? AND position=1",
            (chat_id,),
        )
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    restored = await client.post(
        f"/api/recovery-items/{deletion_id}/restore",
        json=_command(preview, "restore-garden-purposes"),
    )
    assert restored.status_code == 200, restored.text
    read = (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()
    assert [item["image_role"] for item in read["attachments"]] == ROLES


@pytest.mark.parametrize("change", ["inherit", "reorder", "roles_only", "automatic", "short"])
async def test_prior_edit_binds_purposes_to_the_resolved_inputs(
    app: FastAPI, client: AsyncClient, change: str
) -> None:
    ids = await selected_images(client)
    chat_id = (await client.post("/api/chats", json={"title": "Garden edit"})).json()["id"]
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "Describe a garden",
                "mode": "text",
                "input_artifact_ids": ids,
                "input_image_roles": ROLES,
            },
        )
        assert response.status_code == 202, response.text
        message_id = response.json()["user_message"]["id"]
        source = (await client.get(f"/api/messages/{message_id}/edit-source")).json()
        assert source["input_artifact_ids"] == ids
        assert source["input_image_roles"] == ROLES
        values: dict[str, Any] = {
            "text": "Describe the garden paths",
            "idempotency_key": "edited-garden",
        }
        expected_roles: list[str] | None = ROLES
        expected_ids = ids
        if change == "reorder":
            values["input_artifact_ids"] = expected_ids = list(reversed(ids))
            expected_roles = None
        elif change == "roles_only":
            values["input_image_roles"] = expected_roles = ["edit_source", "reference", "reference"]
        elif change == "automatic":
            values["input_image_roles"] = expected_roles = None
        elif change == "short":
            values["input_image_roles"] = ["reference"]
        with SessionLocal() as session, session.no_autoflush:
            if change == "short":
                with pytest.raises(ValueError, match="Image roles must match"):
                    await prepare_prior_turn_edit(
                        app.state.services.orchestrator,
                        session,
                        message_id,
                        PriorTurnEditRequest.model_validate(values),
                    )
                assert not session.new and not session.dirty
                return
            prepared = await prepare_prior_turn_edit(
                app.state.services.orchestrator,
                session,
                message_id,
                PriorTurnEditRequest.model_validate(values),
            )
            assert prepared.request.input_artifact_ids == expected_ids
            assert prepared.request.input_image_roles == expected_roles
            assert not session.new and not session.dirty
        edited = await client.post(f"/api/messages/{message_id}/edits", json=values)
        assert edited.status_code == 202, edited.text
        with SessionLocal() as session:
            run = session.get(Run, edited.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            assert snapshot.input_artifact_ids == expected_ids
            assert snapshot.input_image_roles == expected_roles
        replay = await client.post(f"/api/messages/{message_id}/edits", json=values)
        assert replay.status_code == 202, replay.text
        assert replay.json()["work_plan_id"] == edited.json()["work_plan_id"]


async def test_migration_preserves_draft_holds_order_and_indexes(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    chat_id, ids = await saved_draft(client, None)
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        triggers = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
    command.downgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(chat_composer_draft_attachments)")
        ]
        rows = connection.execute(
            "SELECT * FROM chat_composer_draft_attachments ORDER BY position"
        ).fetchall()
        indexes = connection.execute(
            "PRAGMA index_list(chat_composer_draft_attachments)"
        ).fetchall()
        holds = connection.execute(
            "PRAGMA foreign_key_list(chat_composer_draft_attachments)"
        ).fetchall()
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        assert (
            connection.execute(
                "SELECT "
                + ", ".join(columns)
                + " FROM chat_composer_draft_attachments ORDER BY position"
            ).fetchall()
            == rows
        )
        assert (
            connection.execute("SELECT image_role FROM chat_composer_draft_attachments").fetchall()
            == [(None,)] * 3
        )
        assert (
            connection.execute("PRAGMA index_list(chat_composer_draft_attachments)").fetchall()
            == indexes
        )
        assert (
            connection.execute(
                "PRAGMA foreign_key_list(chat_composer_draft_attachments)"
            ).fetchall()
            == holds
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
            ).fetchall()
            == triggers
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO chat_composer_draft_attachments "
                "(chat_id, position, artifact_id, kind, origin) "
                "VALUES (?, 99, ?, 'image', 'uploaded')",
                (chat_id, "missing-picture"),
            )
        connection.rollback()
    assert [
        item["artifact_id"]
        for item in (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()["attachments"]
    ] == ids
    command.downgrade(config, PARENT)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
            ).fetchall()
            == triggers
        )


async def test_downgrade_refuses_to_discard_saved_picture_purposes(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    chat_id, _ = await saved_draft(client, ROLES)
    with pytest.raises(RuntimeError, match="Picture purposes must be cleared"):
        command.downgrade(alembic_config(settings), PARENT)
    read = (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()
    assert [item["image_role"] for item in read["attachments"]] == ROLES
