"""Constrain chat routing mode and edit template mask mode in storage.

Both fields already had a closed vocabulary above the database: `RoutingMode`
governs one and `MaskMode` the other, and the API and browser both declare
them. The column accepted anything, so a value neither layer can produce could
still be written by a migration, a direct statement, or a future code path that
forgets.

REFUSES RATHER THAN COERCES. A row outside the vocabulary is not silently
rewritten to a default: coercion would destroy the evidence of how it got there
and quietly change what a chat routes to. The upgrade stops and names the
column, the count, and the offending values, which is recoverable by hand.

SQLite cannot add a CHECK to an existing table, so both constraints arrive by
table rebuild. A rebuild is where triggers get lost: the guards defined ON a
rebuilt table die with the old table, and a guard on ANOTHER table that names
this one makes the final rename fail outright. Both were observed here - the
rename raised "no such table: main.chats" from the artifacts delete guard, and
the two chat guards would have been dropped without a word. So every trigger
naming the table is captured with the exact text that created it, dropped, and
restored afterwards.

Revision ID: c8e2f4a71d90
Revises: e7b9c4d12f60
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8e2f4a71d90"
down_revision: str | None = "e7b9c4d12f60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Held literally here, and asserted against the enums by a parity test. A
# migration must describe the schema at ITS revision, so importing the live
# enum would let a later member silently rewrite what this migration did.
_ROUTING_MODES = ("auto", "text", "image", "video")
_MASK_MODES = ("none", "selection", "inverse")


def _refuse_invalid(table: str, column: str, allowed: tuple[str, ...]) -> None:
    """Stop the upgrade rather than rewrite a row nobody can explain."""

    values = ", ".join(f"'{value}'" for value in allowed)
    connection = op.get_bind()
    offending = (
        connection.execute(
            sa.text(
                f"SELECT DISTINCT {column} FROM {table} "  # noqa: S608 - fixed identifiers
                f"WHERE {column} NOT IN ({values})"
            )
        )
        .scalars()
        .all()
    )
    if offending:
        raise RuntimeError(
            f"{table}.{column} holds {len(offending)} value(s) outside its "
            f"vocabulary: {sorted(str(value) for value in offending)}. "
            f"Allowed: {sorted(allowed)}. Correct these rows before upgrading; "
            "they are not rewritten automatically because the correct value "
            "cannot be inferred from a wrong one."
        )


def _triggers_naming(table: str) -> list[tuple[str, str]]:
    """Every trigger defined ON `table` or naming it, as (name, create text).

    The stored text is what SQLite itself recorded, so restoring it reproduces
    the trigger exactly rather than whatever the current source happens to say.
    """

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            "AND sql IS NOT NULL AND (tbl_name = :table OR sql LIKE :pattern)"
        ),
        {"table": table, "pattern": f"%{table}%"},
    ).all()
    return [(str(name), str(sql)) for name, sql in rows]


def _add_vocabulary_check(table: str, name: str, column: str, allowed: tuple[str, ...]) -> None:
    """Rebuild `table` with a membership CHECK, keeping its triggers."""

    saved = _triggers_naming(table)
    for trigger, _ in saved:
        op.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
    values = ", ".join(f"'{value}'" for value in allowed)
    with op.batch_alter_table(table) as batch_op:
        batch_op.create_check_constraint(name, f"{column} IN ({values})")
    for _, statement in saved:
        op.execute(statement)


def _drop_vocabulary_check(table: str, name: str) -> None:
    saved = _triggers_naming(table)
    for trigger, _ in saved:
        op.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
    with op.batch_alter_table(table) as batch_op:
        batch_op.drop_constraint(name, type_="check")
    for _, statement in saved:
        op.execute(statement)


def upgrade() -> None:
    _refuse_invalid("chats", "routing_mode", _ROUTING_MODES)
    _refuse_invalid("edit_templates", "mask_mode", _MASK_MODES)
    _add_vocabulary_check("chats", "ck_chat_routing_mode", "routing_mode", _ROUTING_MODES)
    _add_vocabulary_check("edit_templates", "ck_edit_template_mask_mode", "mask_mode", _MASK_MODES)


def downgrade() -> None:
    _drop_vocabulary_check("edit_templates", "ck_edit_template_mask_mode")
    _drop_vocabulary_check("chats", "ck_chat_routing_mode")
