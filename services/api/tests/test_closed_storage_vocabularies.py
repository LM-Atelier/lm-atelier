"""Storage refuses a value the server's vocabulary does not contain.

`Chat.routing_mode` and `EditTemplate.mask_mode` were closed above the database
and open inside it. A value neither the API nor the browser can produce could
still be written by a direct statement or a code path that forgot, and nothing
would notice until something downstream read it.

The parity test is the one that matters most over time. The migration holds the
values literally, because a migration must describe the schema at ITS revision
rather than at whatever the enum says later - so the two CAN drift, and this
asserts they have not.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base
from local_lm.domain import MaskMode, RoutingMode
from local_lm.migrations.versions import (
    c8e2f4a71d90_closed_storage_vocabularies as vocabulary_migration,
)
from local_lm.models import Chat, EditTemplate


@pytest.fixture
def sessions(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{tmp_path / 'vocab.sqlite3'}")
    Base.metadata.create_all(engine)
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()


def test_the_migration_and_the_enums_hold_the_same_values() -> None:
    """The two copies exist for a reason; this asserts they agree.

    Deriving the migration from the live enum would be worse, not better: a
    member added next year would silently change what a migration written today
    claims to have done. Two copies plus this test is the honest arrangement.
    """

    assert set(vocabulary_migration._ROUTING_MODES) == {m.value for m in RoutingMode}
    assert set(vocabulary_migration._MASK_MODES) == {m.value for m in MaskMode}


@pytest.mark.parametrize("value", [m.value for m in RoutingMode])
def test_every_declared_routing_mode_is_storable(
    sessions: sessionmaker[Session], value: str
) -> None:
    """A constraint that refuses a value the server produces is worse than none."""

    with sessions() as session:
        session.add(Chat(title="ok", routing_mode=value))
        session.commit()


@pytest.mark.parametrize("value", ["", "auto ", "AUTO", "chat", "text;--"])
def test_storage_refuses_a_routing_mode_outside_the_vocabulary(
    sessions: sessionmaker[Session], value: str
) -> None:
    """Including the near-misses: case, whitespace, and a plausible wrong word."""

    with sessions() as session:
        session.add(Chat(title="bad", routing_mode=value))
        with pytest.raises(IntegrityError) as caught:
            session.commit()
        assert "ck_chat_routing_mode" in str(caught.value)


@pytest.mark.parametrize("value", [m.value for m in MaskMode])
def test_every_declared_mask_mode_is_storable(sessions: sessionmaker[Session], value: str) -> None:
    with sessions() as session:
        session.add(EditTemplate(name=f"t-{value}", instruction="do a thing", mask_mode=value))
        session.commit()


@pytest.mark.parametrize("value", ["", "None", "invert", "mask"])
def test_storage_refuses_a_mask_mode_outside_the_vocabulary(
    sessions: sessionmaker[Session], value: str
) -> None:
    """ "invert" is the trap: the producer reads mask["invert"] but stores
    "inverse", so the wrong one is a plausible thing to write by hand."""

    with sessions() as session:
        session.add(EditTemplate(name=f"bad-{value}", instruction="x", mask_mode=value))
        with pytest.raises(IntegrityError) as caught:
            session.commit()
        assert "ck_edit_template_mask_mode" in str(caught.value)


def test_upgrading_refuses_a_legacy_row_rather_than_coercing_it(
    tmp_path: Path,
) -> None:
    """The case the migration exists to handle, driven end to end.

    A database is migrated to the revision BEFORE this one, given a row no
    vocabulary admits, and then upgraded. The upgrade must stop and say what it
    found - not rewrite the row to a default, which would destroy the evidence
    of how it got there and quietly change what the chat routes to.
    """

    settings = Settings(data_dir=tmp_path / "closed-vocabulary-refusal")
    settings.prepare()
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"

    command.upgrade(config, "e7b9c4d12f60")
    with sqlite3.connect(database) as raw:
        raw.execute(
            "INSERT INTO chats (id, title, archived, routing_mode,"
            " confirm_uncertain_media, vision_settings_json, created_at, updated_at)"
            " VALUES ('chat_legacy', 'legacy', 0, 'telepathy', 0, '{}',"
            " '2026-01-01', '2026-01-01')"
        )
        raw.commit()

    with pytest.raises(RuntimeError) as caught:
        command.upgrade(config, "c8e2f4a71d90")
    message = str(caught.value)
    assert "routing_mode" in message
    assert "telepathy" in message, message

    # The row is still there, untouched, which is the whole point.
    with sqlite3.connect(database) as raw:
        stored = raw.execute("SELECT routing_mode FROM chats WHERE id = 'chat_legacy'").fetchone()
    assert stored == ("telepathy",)


def test_the_rebuild_keeps_every_trigger_that_names_the_rebuilt_table(
    tmp_path: Path,
) -> None:
    """A CHECK arrives by table rebuild, and a rebuild loses triggers silently.

    Two guards are defined ON chats and one on artifacts names it. The first
    kind dies with the old table during the copy; the second makes the rename
    fail outright with "no such table: main.chats". Both happened here before
    the migration captured and restored them, and only the second was loud.

    Comparing the exact stored CREATE text, not just the names, so a trigger
    that comes back with a weakened body fails too.
    """

    settings = Settings(data_dir=tmp_path / "closed-vocabulary-triggers")
    settings.prepare()
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"

    command.upgrade(config, "e7b9c4d12f60")
    with sqlite3.connect(database) as raw:
        before = dict(
            raw.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND sql IS NOT NULL"
            ).fetchall()
        )
    naming_chats = {name for name, sql in before.items() if "chats" in sql}
    assert naming_chats, "expected the parent revision to carry chat triggers"

    command.upgrade(config, "c8e2f4a71d90")
    with sqlite3.connect(database) as raw:
        after = dict(
            raw.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND sql IS NOT NULL"
            ).fetchall()
        )
    assert set(after) == set(before)
    assert after == before


def test_a_clean_database_upgrades_and_carries_both_constraints(
    tmp_path: Path,
) -> None:
    """The ordinary path, and proof the constraints reach a migrated schema.

    Creating tables from the models would prove nothing about the MIGRATION -
    that is the gap this closes, since an installed instance only ever sees the
    migrated shape.
    """

    settings = Settings(data_dir=tmp_path / "closed-vocabulary-fresh")
    settings.prepare()
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    command.upgrade(config, "c8e2f4a71d90")

    engine = create_engine(f"sqlite:///{database}")
    try:
        with engine.connect() as connection:
            schema = (
                connection.execute(
                    text(
                        "SELECT sql FROM sqlite_master WHERE type='table'"
                        " AND name IN ('chats', 'edit_templates')"
                    )
                )
                .scalars()
                .all()
            )
        joined = "\n".join(schema)
        assert "ck_chat_routing_mode" in joined
        assert "ck_edit_template_mask_mode" in joined
        with engine.connect() as connection, pytest.raises(IntegrityError) as caught:
            connection.execute(
                text(
                    "INSERT INTO chats (id, title, archived, routing_mode,"
                    " confirm_uncertain_media, vision_settings_json,"
                    " created_at, updated_at)"
                    " VALUES ('c2', 't', 0, 'nonsense', 0, '{}',"
                    " '2026-01-01', '2026-01-01')"
                )
            )
            connection.commit()
        assert "ck_chat_routing_mode" in str(caught.value)
    finally:
        engine.dispose()
