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

import inspect
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import get_args

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base
from local_lm.domain import MaskMode, ResourceKind, RoutingMode
from local_lm.migrations.versions import (
    c8e2f4a71d90_closed_storage_vocabularies as vocabulary_migration,
)
from local_lm.models import Chat, EditTemplate, _closed_vocabulary_check
from local_lm.schemas import WorkflowDependencyResourceKind as SchemasResourceKind
from local_lm.workflow_dependencies import (
    WORKFLOW_DEPENDENCY_RESOURCE_KINDS,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyResourceKind as DependenciesResourceKind,
)


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


def test_the_resource_kind_check_still_matches_the_table_it_was_created_with() -> None:
    """Deriving a constraint must change where it comes from, not what it says.

    The resource_kind vocabulary used to be a SQL string written out inside
    models.py, with a second copy of the same string inside the migration that
    created the table. It is now generated from ResourceKind, so a seventh kind
    is added in one place instead of three.

    That is only safe if the generated text is what the table already has.
    SQLite fixes a CHECK at table creation, so a changed string would leave a
    running database and the models disagreeing about a constraint they both
    claim to hold, and nothing else in the suite would notice. This reads the
    literal out of the creating migration rather than retyping it here, because
    a retyped copy is the third copy this change exists to remove.
    """
    from local_lm.migrations.versions import (
        c8f2d7a91e64_workflow_dependency_bindings as creating_migration,
    )

    generated = _closed_vocabulary_check("resource_kind", ResourceKind)

    source = inspect.getsource(creating_migration)
    match = re.search(
        r'"(resource_kind IN \([^"]*)"\s*\n\s*"([^"]*\))"',
        source,
    )
    assert match is not None, "the creating migration no longer spells the check out"
    original = match.group(1) + match.group(2)

    assert generated == original, (
        f"the derived check no longer matches the one the table was created with."
        f"\n  generated: {generated}\n  migration: {original}"
    )


def test_every_resource_kind_spelling_agrees_with_the_enum() -> None:
    """Four places name this vocabulary. Only one of them can be the source.

    ResourceKind is that source: the database CHECK is generated from it, the
    frozenset in workflow_dependencies is derived from it, and the schemas alias
    IS it. Exactly one copy cannot be derived - the Literal in
    workflow_dependencies, consumed by a model that validates strictly, where
    the enum makes pydantic refuse the plain strings the wire carries ("Input
    should be an instance of ResourceKind") and changes the contract digest
    computed from those inputs.

    An earlier revision of this test asserted that BOTH aliases were unavoidable
    and bound both as ordered tuples. That was wrong: the schemas models use
    ApiModel, which is not strict, and a focused compatibility probe proved the
    enum works there. A seventh kind added to the enum and nowhere else fails here,
    naming the spelling that was missed - which is the whole point, because the
    alternative is discovering it when one API surface accepts a value another
    rejects.
    """
    expected = tuple(member.value for member in ResourceKind)

    assert SchemasResourceKind is ResourceKind, (
        "schemas.WorkflowDependencyResourceKind is no longer ResourceKind itself"
    )
    assert get_args(DependenciesResourceKind) == expected, (
        "workflow_dependencies.WorkflowDependencyResourceKind has drifted from ResourceKind"
    )
    assert frozenset(expected) == WORKFLOW_DEPENDENCY_RESOURCE_KINDS, (
        "the derived resource-kind set no longer matches ResourceKind"
    )


def test_the_resource_kind_order_is_load_bearing_not_cosmetic() -> None:
    """The generated database CHECK is compared character-for-character.

    Reordering the enum changes the SQL the models declare while the table keeps
    the constraint it was created with, so the two would silently disagree about
    a rule they both claim to hold. The one remaining Literal alias is compared
    as an ordered tuple above for the same reason.
    """
    generated = _closed_vocabulary_check("resource_kind", ResourceKind)
    assert generated.startswith("resource_kind IN ('model_profile', 'model_install'"), (
        "the enum order changed; the generated CHECK no longer leads with the "
        "order the table was created with"
    )
