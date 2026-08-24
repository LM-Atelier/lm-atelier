"""close the Prompt Library revision schema-version null guard

Revision ID: b7c1e4a90f26
Revises: d2f8c1a94e70
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c1e4a90f26"
down_revision: str | None = "d2f8c1a94e70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_FIXED_REVISION_INSERT_TRIGGER = """
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
                        OR json_type(NEW.contract_json, '$.schema_version')
                           != 'integer'
                        OR json_extract(NEW.contract_json, '$.schema_version')
                           IS NOT NEW.schema_version
    THEN RAISE(ABORT, 'prompt template revision contract is invalid') END;
END
"""


_LEGACY_REVISION_INSERT_TRIGGER = """
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
"""


def upgrade() -> None:
    invalid_count = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT count(*)
            FROM prompt_template_revisions
            WHERE CASE
              WHEN NOT json_valid(contract_json) THEN 1
              WHEN json_type(contract_json) != 'object' THEN 1
              WHEN json_type(contract_json, '$.schema_version') != 'integer' THEN 1
              WHEN json_extract(contract_json, '$.schema_version') IS NOT schema_version THEN 1
              ELSE 0
            END
            """
            )
        )
        .scalar_one()
    )
    if invalid_count:
        raise RuntimeError(
            "Cannot upgrade Prompt Library revision contract guard: "
            "stored revision contract is invalid."
        )
    op.execute("DROP TRIGGER prompt_template_revision_insert_guard")
    op.execute(_FIXED_REVISION_INSERT_TRIGGER)


def downgrade() -> None:
    op.execute("DROP TRIGGER prompt_template_revision_insert_guard")
    op.execute(_LEGACY_REVISION_INSERT_TRIGGER)
