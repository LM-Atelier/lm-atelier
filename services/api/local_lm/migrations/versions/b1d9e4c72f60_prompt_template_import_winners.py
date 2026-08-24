"""add immutable Prompt Template import winners

Revision ID: b1d9e4c72f60
Revises: a6e2c9f31b47
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b1d9e4c72f60"
down_revision: str | None = "a6e2c9f31b47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CREATE_TRIGGER_SQL = (
    """
CREATE TRIGGER prompt_template_import_winner_insert_guard
BEFORE INSERT ON prompt_template_import_winners
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1
    FROM prompt_template_definitions AS definition
    JOIN prompt_template_revisions AS revision
      ON revision.id = NEW.prompt_template_revision_id
     AND revision.prompt_template_id = definition.id
    WHERE definition.id = NEW.prompt_template_id
      AND definition.current_revision_id = NEW.prompt_template_revision_id
      AND revision.version = 1
      AND revision.contract_sha256 = NEW.contract_sha256
  ) THEN RAISE(ABORT, 'prompt template import winner is invalid') END;
END
""",
    """
CREATE TRIGGER prompt_template_import_winner_update_guard
BEFORE UPDATE ON prompt_template_import_winners
BEGIN
  SELECT RAISE(ABORT, 'prompt template import winners are immutable');
END
""",
    """
CREATE TRIGGER prompt_template_import_winner_delete_guard
BEFORE DELETE ON prompt_template_import_winners
BEGIN
  SELECT RAISE(ABORT, 'prompt template import winners are immutable');
END
""",
)

_DROP_TRIGGER_SQL = tuple(
    f"DROP TRIGGER {name}"
    for name in (
        "prompt_template_import_winner_delete_guard",
        "prompt_template_import_winner_update_guard",
        "prompt_template_import_winner_insert_guard",
    )
)


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "prompt_template_import_winners",
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("authority_rule", sa.String(length=64), nullable=False),
        sa.Column("prompt_template_id", sa.String(length=40), nullable=False),
        sa.Column("prompt_template_revision_id", sa.String(length=40), nullable=False),
        sa.Column("contract_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("idempotency_key"),
        sa.ForeignKeyConstraint(
            ["prompt_template_id"], ["prompt_template_definitions.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["prompt_template_revision_id"], ["prompt_template_revisions.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "prompt_template_id", name="uq_prompt_template_import_winner_template_id"
        ),
        sa.UniqueConstraint(
            "prompt_template_revision_id", name="uq_prompt_template_import_winner_revision_id"
        ),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 200 AND instr(idempotency_key, char(0)) = 0",
            name="ck_prompt_template_import_winner_idempotency_key",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("request_sha256"),
            name="ck_prompt_template_import_winner_request_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("bundle_sha256"),
            name="ck_prompt_template_import_winner_bundle_sha256",
        ),
        sa.CheckConstraint(
            "authority_rule = 'prompt-template-import-authority-v1'",
            name="ck_prompt_template_import_winner_authority_rule",
        ),
        sa.CheckConstraint(
            "length(prompt_template_id) BETWEEN 1 AND 40",
            name="ck_prompt_template_import_winner_template_id",
        ),
        sa.CheckConstraint(
            "length(prompt_template_revision_id) BETWEEN 1 AND 40",
            name="ck_prompt_template_import_winner_revision_id",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("contract_sha256"),
            name="ck_prompt_template_import_winner_contract_sha256",
        ),
    )
    for statement in _CREATE_TRIGGER_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DROP_TRIGGER_SQL:
        op.execute(statement)
    op.drop_table("prompt_template_import_winners")
