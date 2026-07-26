"""scope turn idempotency keys to their chat

Revision ID: b8e4f2c17a90
Revises: a4d7c2e91b63
Create Date: 2026-07-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8e4f2c17a90"
down_revision: str | None = "a4d7c2e91b63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAMING_CONVENTION = {
    "uq": "uq_%(table_name)s_%(column_0_name)s",
}


def upgrade() -> None:
    with op.batch_alter_table(
        "runs",
        recreate="always",
        naming_convention=_NAMING_CONVENTION,
    ) as batch:
        batch.drop_constraint("uq_runs_idempotency_key", type_="unique")
        batch.create_unique_constraint(
            "uq_runs_chat_id_idempotency_key",
            ["chat_id", "idempotency_key"],
        )

    op.create_table(
        "turn_creation_claims",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("chat_id", sa.String(length=40), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("owner_token", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_turn_creation_claim_chat_id_idempotency_key",
        ),
    )
    op.create_index(
        op.f("ix_turn_creation_claims_chat_id"),
        "turn_creation_claims",
        ["chat_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_turn_creation_claims_chat_id"),
        table_name="turn_creation_claims",
    )
    op.drop_table("turn_creation_claims")

    # The older schema permits only one occurrence of a key globally. Preserve
    # every run when rolling back a database that reused the same client key in
    # separate chats by deterministically namespacing later duplicates.
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            """
            SELECT id, idempotency_key
            FROM runs
            WHERE idempotency_key IS NOT NULL
            ORDER BY idempotency_key, created_at, id
            """
        )
    ).mappings()
    used: set[str] = set()
    for row in rows:
        key = str(row["idempotency_key"])
        if key not in used:
            used.add(key)
            continue
        run_id = str(row["id"])
        suffix = 0
        while True:
            marker = f":{run_id}" if suffix == 0 else f":{run_id}:{suffix}"
            candidate = f"{key[: 200 - len(marker)]}{marker}"
            if candidate not in used:
                break
            suffix += 1
        connection.execute(
            sa.text("UPDATE runs SET idempotency_key = :key WHERE id = :run_id"),
            {"key": candidate, "run_id": run_id},
        )
        used.add(candidate)

    with op.batch_alter_table(
        "runs",
        recreate="always",
        naming_convention=_NAMING_CONVENTION,
    ) as batch:
        batch.drop_constraint(
            "uq_runs_chat_id_idempotency_key",
            type_="unique",
        )
        batch.create_unique_constraint(
            "uq_runs_idempotency_key",
            ["idempotency_key"],
        )
