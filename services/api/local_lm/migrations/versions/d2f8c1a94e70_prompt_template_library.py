"""add immutable Prompt Library definitions and revisions

Revision ID: d2f8c1a94e70
Revises: a9c4e7d21b60
Create Date: 2026-08-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d2f8c1a94e70"
down_revision: str | None = "a9c4e7d21b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Historical migrations own their SQL. Importing the application's current
# trigger tuple made this revision change whenever the live guard changed.
_DEFINITION_TRIGGER_SQL = (
    """
CREATE TRIGGER prompt_template_definition_insert_guard
BEFORE INSERT ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL
    THEN RAISE(ABORT, 'prompt template must begin without a current revision') END;
END
""",
    """
CREATE TRIGGER prompt_template_definition_update_guard
BEFORE UPDATE ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id
    THEN RAISE(ABORT, 'prompt template identity is immutable') END;
  SELECT CASE WHEN OLD.current_revision_id IS NOT NULL
                        AND NEW.current_revision_id IS NULL
    THEN RAISE(ABORT, 'prompt template current revision cannot be cleared') END;
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
    FROM prompt_template_revisions AS revision
    WHERE revision.id = NEW.current_revision_id
      AND revision.prompt_template_id = NEW.id
      AND revision.version = (
        SELECT max(candidate.version)
        FROM prompt_template_revisions AS candidate
        WHERE candidate.prompt_template_id = NEW.id
      )
  ) THEN RAISE(ABORT, 'prompt template current revision is invalid') END;
END
""",
    """
CREATE TRIGGER prompt_template_definition_delete_guard
BEFORE DELETE ON prompt_template_definitions
BEGIN
  SELECT RAISE(ABORT, 'prompt template definitions are archived, not deleted');
END
""",
)

_REVISION_TRIGGER_SQL = (
    """
CREATE TRIGGER prompt_template_revision_insert_guard
BEFORE INSERT ON prompt_template_revisions
BEGIN
  SELECT CASE WHEN NEW.version != COALESCE((
    SELECT max(existing.version) + 1
    FROM prompt_template_revisions AS existing
    WHERE existing.prompt_template_id = NEW.prompt_template_id
  ), 1)
    THEN RAISE(ABORT, 'prompt template revision version is not append-only') END;
  SELECT CASE WHEN NOT json_valid(NEW.contract_json)
                        OR json_type(NEW.contract_json) != 'object'
                        OR json_extract(NEW.contract_json, '$.schema_version')
                           != NEW.schema_version
    THEN RAISE(ABORT, 'prompt template revision contract is invalid') END;
END
""",
    """
CREATE TRIGGER prompt_template_revision_update_guard
BEFORE UPDATE ON prompt_template_revisions
BEGIN
  SELECT RAISE(ABORT, 'prompt template revisions are immutable');
END
""",
    """
CREATE TRIGGER prompt_template_revision_delete_guard
BEFORE DELETE ON prompt_template_revisions
BEGIN
  SELECT RAISE(ABORT, 'prompt template revisions are immutable');
END
""",
)

_CREATE_PROMPT_TEMPLATE_TRIGGER_SQL = _DEFINITION_TRIGGER_SQL + _REVISION_TRIGGER_SQL
_DROP_PROMPT_TEMPLATE_TRIGGER_SQL = (
    "DROP TRIGGER prompt_template_revision_delete_guard",
    "DROP TRIGGER prompt_template_revision_update_guard",
    "DROP TRIGGER prompt_template_revision_insert_guard",
    "DROP TRIGGER prompt_template_definition_delete_guard",
    "DROP TRIGGER prompt_template_definition_update_guard",
    "DROP TRIGGER prompt_template_definition_insert_guard",
)


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "prompt_template_definitions",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("current_revision_id", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_prompt_template_definition_name"),
        sa.CheckConstraint(
            "length(name) BETWEEN 1 AND 200 AND name = trim(name) AND instr(name, char(0)) = 0",
            name="ck_prompt_template_definition_name",
        ),
        sa.CheckConstraint(
            "length(description) <= 4000 AND instr(description, char(0)) = 0",
            name="ck_prompt_template_definition_description",
        ),
        sa.CheckConstraint(
            "current_revision_id IS NULL OR length(current_revision_id) BETWEEN 1 AND 40",
            name="ck_prompt_template_definition_current_revision",
        ),
    )
    op.create_index(
        "ix_prompt_template_definitions_archived_name",
        "prompt_template_definitions",
        ["archived", "name", "id"],
    )
    op.create_table(
        "prompt_template_revisions",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("prompt_template_id", sa.String(length=40), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("contract_json", sa.JSON(), nullable=False),
        sa.Column("contract_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["prompt_template_id"],
            ["prompt_template_definitions.id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "prompt_template_id",
            "version",
            name="uq_prompt_template_revision_version",
        ),
        sa.CheckConstraint(
            "version > 0",
            name="ck_prompt_template_revision_version_positive",
        ),
        sa.CheckConstraint(
            "schema_version = 1",
            name="ck_prompt_template_revision_schema_version",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("contract_sha256"),
            name="ck_prompt_template_revision_contract_sha256",
        ),
    )
    op.create_index(
        "ix_prompt_template_revisions_definition_created",
        "prompt_template_revisions",
        ["prompt_template_id", "version", "id"],
    )
    op.create_index(
        "ix_prompt_template_revisions_contract_sha256",
        "prompt_template_revisions",
        ["contract_sha256"],
    )
    for statement in _CREATE_PROMPT_TEMPLATE_TRIGGER_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DROP_PROMPT_TEMPLATE_TRIGGER_SQL:
        op.execute(statement)
    op.drop_index(
        "ix_prompt_template_revisions_contract_sha256",
        table_name="prompt_template_revisions",
    )
    op.drop_index(
        "ix_prompt_template_revisions_definition_created",
        table_name="prompt_template_revisions",
    )
    op.drop_table("prompt_template_revisions")
    op.drop_index(
        "ix_prompt_template_definitions_archived_name",
        table_name="prompt_template_definitions",
    )
    op.drop_table("prompt_template_definitions")
