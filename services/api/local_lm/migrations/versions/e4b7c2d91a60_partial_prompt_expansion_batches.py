"""Preserve complete model-slot results when a batch is only partly filled."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "e4b7c2d91a60"
down_revision: str | None = "d7a91c4e2b60"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

# Frozen trigger statements keep this revision independent of live model code.
_PREVIOUS_ITEM_INSERT = """
CREATE TRIGGER prompt_expansion_item_insert_guard
BEFORE INSERT ON prompt_expansion_items
BEGIN
  SELECT CASE WHEN NEW.ordinal IS NOT COALESCE((
    SELECT max(existing.ordinal) + 1
    FROM prompt_expansion_items AS existing
    WHERE existing.batch_id = NEW.batch_id
  ), 1)
    THEN RAISE(ABORT, 'prompt expansion item ordinals are not contiguous') END;
  SELECT CASE WHEN (
    SELECT count(*) FROM prompt_expansion_items AS existing
    WHERE existing.batch_id = NEW.batch_id
  ) >= COALESCE((
    SELECT json_extract(batch.request_json, '$.item_count')
    FROM prompt_expansion_batches AS batch
    WHERE batch.id = NEW.batch_id
  ), 0)
    THEN RAISE(ABORT, 'prompt expansion batch already has all items') END;
  SELECT CASE WHEN NEW.review_version != 1
                        OR NEW.reroll_count != 0
                        OR typeof(NEW.selected) != 'integer'
                        OR NEW.selected NOT IN (0, 1)
                        OR NEW.selected != 1
                        OR NEW.original_evidence_json IS NOT NEW.current_evidence_json
                        OR NEW.original_rendered_prompt IS NOT NEW.reviewed_prompt
                        OR NEW.original_rendered_sha256 IS NOT NEW.reviewed_sha256
                        OR NOT json_valid(NEW.original_evidence_json)
                        OR json_type(NEW.original_evidence_json) != 'array'
                        OR length(NEW.original_evidence_json)
                           > 262144
                        OR NEW.work_step_id IS NOT NULL
                        OR NEW.run_id IS NOT NULL
                        OR NEW.media_seed IS NOT NULL
    THEN RAISE(ABORT, 'prompt expansion item initial state is invalid') END;
END
"""

_PARTIAL_ITEM_INSERT = """
CREATE TRIGGER prompt_expansion_item_insert_guard
BEFORE INSERT ON prompt_expansion_items
BEGIN
  SELECT CASE WHEN (
    SELECT batch.codec_version FROM prompt_expansion_batches AS batch
    WHERE batch.id = NEW.batch_id
  ) = 2 AND NEW.ordinal IS NOT COALESCE((
    SELECT max(existing.ordinal) + 1
    FROM prompt_expansion_items AS existing
    WHERE existing.batch_id = NEW.batch_id
  ), 1)
    THEN RAISE(ABORT, 'prompt expansion item ordinals are not contiguous') END;
  SELECT CASE WHEN (
    SELECT count(*) FROM prompt_expansion_items AS existing
    WHERE existing.batch_id = NEW.batch_id
  ) >= COALESCE((
    SELECT json_extract(batch.request_json, '$.item_count')
           - CASE WHEN batch.codec_version = 3 THEN 1 ELSE 0 END
    FROM prompt_expansion_batches AS batch
    WHERE batch.id = NEW.batch_id
  ), 0)
    THEN RAISE(ABORT, 'prompt expansion batch already has all items') END;
  SELECT CASE WHEN typeof(NEW.ordinal) != 'integer'
                        OR NEW.ordinal <= COALESCE((
                          SELECT max(existing.ordinal) FROM prompt_expansion_items AS existing
                          WHERE existing.batch_id = NEW.batch_id
                        ), 0)
                        OR NEW.ordinal > COALESCE((
                          SELECT json_extract(batch.request_json, '$.item_count')
                          FROM prompt_expansion_batches AS batch WHERE batch.id = NEW.batch_id
                        ), 0)
    THEN RAISE(ABORT, 'prompt expansion item ordinal is invalid') END;
  SELECT CASE WHEN NEW.review_version != 1
                        OR NEW.reroll_count != 0
                        OR typeof(NEW.selected) != 'integer'
                        OR NEW.selected NOT IN (0, 1)
                        OR NEW.selected != 1
                        OR NEW.original_evidence_json IS NOT NEW.current_evidence_json
                        OR NEW.original_rendered_prompt IS NOT NEW.reviewed_prompt
                        OR NEW.original_rendered_sha256 IS NOT NEW.reviewed_sha256
                        OR NOT json_valid(NEW.original_evidence_json)
                        OR json_type(NEW.original_evidence_json) != 'array'
                        OR length(NEW.original_evidence_json)
                           > 262144
                        OR NEW.work_step_id IS NOT NULL
                        OR NEW.run_id IS NOT NULL
                        OR NEW.media_seed IS NOT NULL
    THEN RAISE(ABORT, 'prompt expansion item initial state is invalid') END;
END
"""


_TRIGGER_NAMES = (
    "prompt_expansion_batch_insert_guard",
    "prompt_expansion_batch_update_guard",
    "prompt_expansion_item_insert_guard",
    "prompt_expansion_item_update_guard",
)


def _change_codec_constraint(*, partial: bool) -> None:
    connection = op.get_bind()
    sqlite = connection.dialect.name == "sqlite"
    triggers: dict[str, str] = {}
    if sqlite:
        # Alembic uses a dedicated connection without referential deletion.
        # Refuse a differently configured connection before dropping the parent.
        if connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() != 0:
            raise RuntimeError("Prompt batch migration requires the dedicated migration connection")
        for name in _TRIGGER_NAMES:
            statement = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (name,),
            ).scalar_one_or_none()
            if not isinstance(statement, str):
                raise RuntimeError("Prompt batch migration requires all existing guards")
            triggers[name] = statement
        expected = _PREVIOUS_ITEM_INSERT if partial else _PARTIAL_ITEM_INSERT
        if triggers["prompt_expansion_item_insert_guard"].strip() != expected.strip():
            raise RuntimeError("Prompt batch insert guard does not match the preceding schema")
        for name in _TRIGGER_NAMES:
            connection.exec_driver_sql("DROP TRIGGER " + name)
    with op.batch_alter_table("prompt_expansion_batches") as batch:
        batch.drop_constraint("ck_prompt_expansion_batch_codec_version", type_="check")
        batch.create_check_constraint(
            "ck_prompt_expansion_batch_codec_version",
            "codec_version IN (2, 3)" if partial else "codec_version = 2",
        )
    if sqlite:
        # Recreate the three unchanged guards byte-for-byte after table recreation.
        triggers["prompt_expansion_item_insert_guard"] = (
            _PARTIAL_ITEM_INSERT if partial else _PREVIOUS_ITEM_INSERT
        )
        for name in _TRIGGER_NAMES:
            connection.exec_driver_sql(triggers[name])


def upgrade() -> None:
    _change_codec_constraint(partial=True)


def downgrade() -> None:
    partial_count = (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM prompt_expansion_batches WHERE codec_version = 3"))
        .scalar_one()
    )
    if partial_count:
        raise RuntimeError("Partial prompt batches cannot be represented by the previous schema")
    _change_codec_constraint(partial=False)
