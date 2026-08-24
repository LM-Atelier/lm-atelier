"""add chat-scoped Prompt Library expansion persistence

Revision ID: c1e7a4b92d60
Revises: b7c1e4a90f26
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c1e7a4b92d60"
down_revision: str | None = "b7c1e4a90f26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MAX_REQUEST_JSON_CHARS = 1_200_000
_MAX_MODEL_SNAPSHOT_JSON_CHARS = 1_024
_MAX_EVIDENCE_JSON_CHARS = 262_144


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


_TRIGGERS = (
    f"""
CREATE TRIGGER prompt_expansion_batch_insert_guard
BEFORE INSERT ON prompt_expansion_batches
BEGIN
  SELECT CASE WHEN NOT json_valid(NEW.request_json)
                        OR json_type(NEW.request_json) != 'object'
                        OR json_type(NEW.request_json, '$.item_count') != 'integer'
                        OR json_extract(NEW.request_json, '$.item_count') NOT BETWEEN 1 AND 16
                        OR length(NEW.request_json) > {_MAX_REQUEST_JSON_CHARS}
                        OR NOT json_valid(NEW.model_snapshot_json)
                        OR json_type(NEW.model_snapshot_json) != 'object'
                        OR json_type(NEW.model_snapshot_json, '$.version') != 'integer'
                        OR json_extract(NEW.model_snapshot_json, '$.version') != 1
                        OR length(NEW.model_snapshot_json)
                           > {_MAX_MODEL_SNAPSHOT_JSON_CHARS}
                        OR NEW.plan_version != 1
                        OR NEW.state != 'draft'
                        OR NEW.original_plan_sha256 IS NOT NEW.plan_sha256
    THEN RAISE(ABORT, 'prompt expansion JSON is invalid') END;
END
""",
    """
CREATE TRIGGER prompt_expansion_batch_update_guard
BEFORE UPDATE ON prompt_expansion_batches
BEGIN
  SELECT CASE WHEN OLD.state != 'draft'
    THEN RAISE(ABORT, 'queued prompt expansion batches are immutable') END;
  SELECT CASE WHEN NEW.id IS NOT OLD.id
                        OR NEW.chat_id IS NOT OLD.chat_id
                        OR NEW.idempotency_key IS NOT OLD.idempotency_key
                        OR NEW.prompt_template_id IS NOT OLD.prompt_template_id
                        OR NEW.prompt_template_revision_id
                           IS NOT OLD.prompt_template_revision_id
                        OR NEW.schema_version IS NOT OLD.schema_version
                        OR NEW.contract_sha256 IS NOT OLD.contract_sha256
                        OR NEW.codec_version IS NOT OLD.codec_version
                        OR NEW.request_json IS NOT OLD.request_json
                        OR NEW.model_snapshot_json IS NOT OLD.model_snapshot_json
                        OR NEW.original_plan_sha256 IS NOT OLD.original_plan_sha256
                        OR NEW.created_at IS NOT OLD.created_at
    THEN RAISE(ABORT, 'prompt expansion batch identity is immutable') END;
  SELECT CASE WHEN NEW.state NOT IN ('draft', 'queued')
                        OR NEW.plan_version IS NOT OLD.plan_version + 1
    THEN RAISE(ABORT, 'prompt expansion batch update is invalid') END;
END
""",
    f"""
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
                           > {_MAX_EVIDENCE_JSON_CHARS}
    THEN RAISE(ABORT, 'prompt expansion item initial state is invalid') END;
END
""",
    f"""
CREATE TRIGGER prompt_expansion_item_update_guard
BEFORE UPDATE ON prompt_expansion_items
BEGIN
  SELECT CASE WHEN COALESCE((
    SELECT batch.state FROM prompt_expansion_batches AS batch
    WHERE batch.id = OLD.batch_id
  ), 'missing') != 'draft'
    THEN RAISE(ABORT, 'queued prompt expansion items are immutable') END;
  SELECT CASE WHEN NEW.id IS NOT OLD.id
                        OR NEW.batch_id IS NOT OLD.batch_id
                        OR NEW.ordinal IS NOT OLD.ordinal
                        OR NEW.original_evidence_json IS NOT OLD.original_evidence_json
                        OR NEW.original_rendered_prompt IS NOT OLD.original_rendered_prompt
                        OR NEW.original_rendered_sha256 IS NOT OLD.original_rendered_sha256
                        OR NEW.created_at IS NOT OLD.created_at
    THEN RAISE(ABORT, 'prompt expansion item origin is immutable') END;
  SELECT CASE WHEN NEW.review_version IS NOT OLD.review_version + 1
                        OR NEW.reroll_count NOT IN (OLD.reroll_count, OLD.reroll_count + 1)
                        OR typeof(NEW.selected) != 'integer'
                        OR NEW.selected NOT IN (0, 1)
                        OR (NEW.reroll_count = OLD.reroll_count
                            AND NEW.current_evidence_json IS NOT OLD.current_evidence_json)
                        OR NOT json_valid(NEW.current_evidence_json)
                        OR json_type(NEW.current_evidence_json) != 'array'
                        OR length(NEW.current_evidence_json)
                           > {_MAX_EVIDENCE_JSON_CHARS}
    THEN RAISE(ABORT, 'prompt expansion item update is invalid') END;
END
""",
)

_DROP_TRIGGERS = (
    "DROP TRIGGER prompt_expansion_item_update_guard",
    "DROP TRIGGER prompt_expansion_item_insert_guard",
    "DROP TRIGGER prompt_expansion_batch_update_guard",
    "DROP TRIGGER prompt_expansion_batch_insert_guard",
)


def upgrade() -> None:
    op.create_table(
        "prompt_expansion_batches",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("chat_id", sa.String(length=40), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("prompt_template_id", sa.String(length=40), nullable=False),
        sa.Column("prompt_template_revision_id", sa.String(length=40), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("contract_sha256", sa.String(length=64), nullable=False),
        sa.Column("codec_version", sa.Integer(), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("model_snapshot_json", sa.Text(), nullable=False),
        sa.Column("original_plan_sha256", sa.String(length=64), nullable=False),
        sa.Column("plan_sha256", sa.String(length=64), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["prompt_template_id"], ["prompt_template_definitions.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["prompt_template_revision_id"],
            ["prompt_template_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "chat_id", "idempotency_key", name="uq_prompt_expansion_batch_chat_idempotency"
        ),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 200 AND instr(idempotency_key, char(0)) = 0",
            name="ck_prompt_expansion_batch_idempotency_key",
        ),
        sa.CheckConstraint("schema_version = 1", name="ck_prompt_expansion_batch_schema_version"),
        sa.CheckConstraint("codec_version = 2", name="ck_prompt_expansion_batch_codec_version"),
        sa.CheckConstraint(
            _lowercase_sha256_check("contract_sha256"),
            name="ck_prompt_expansion_batch_contract_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("original_plan_sha256"),
            name="ck_prompt_expansion_batch_original_plan_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("plan_sha256"),
            name="ck_prompt_expansion_batch_plan_sha256",
        ),
        sa.CheckConstraint("plan_version > 0", name="ck_prompt_expansion_batch_plan_version"),
        sa.CheckConstraint("state IN ('draft', 'queued')", name="ck_prompt_expansion_batch_state"),
    )
    op.create_index("ix_prompt_expansion_batches_chat_id", "prompt_expansion_batches", ["chat_id"])
    op.create_index(
        "ix_prompt_expansion_batches_prompt_template_id",
        "prompt_expansion_batches",
        ["prompt_template_id"],
    )
    op.create_index(
        "ix_prompt_expansion_batches_prompt_template_revision_id",
        "prompt_expansion_batches",
        ["prompt_template_revision_id"],
    )
    op.create_index(
        "ix_prompt_expansion_batches_chat_created",
        "prompt_expansion_batches",
        ["chat_id", "created_at", "id"],
    )

    op.create_table(
        "prompt_expansion_items",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("batch_id", sa.String(length=40), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("original_evidence_json", sa.Text(), nullable=False),
        sa.Column("current_evidence_json", sa.Text(), nullable=False),
        sa.Column("original_rendered_prompt", sa.Text(), nullable=False),
        sa.Column("original_rendered_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewed_prompt", sa.Text(), nullable=False),
        sa.Column("reviewed_sha256", sa.String(length=64), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("review_version", sa.Integer(), nullable=False),
        sa.Column("reroll_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["batch_id"], ["prompt_expansion_batches.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("batch_id", "ordinal", name="uq_prompt_expansion_item_ordinal"),
        sa.CheckConstraint("ordinal > 0", name="ck_prompt_expansion_item_ordinal"),
        sa.CheckConstraint(
            _lowercase_sha256_check("original_rendered_sha256"),
            name="ck_prompt_expansion_item_original_rendered_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("reviewed_sha256"),
            name="ck_prompt_expansion_item_reviewed_sha256",
        ),
        sa.CheckConstraint(
            "length(original_rendered_prompt) BETWEEN 1 AND 32000 "
            "AND instr(original_rendered_prompt, char(0)) = 0",
            name="ck_prompt_expansion_item_original_prompt",
        ),
        sa.CheckConstraint(
            "length(reviewed_prompt) BETWEEN 1 AND 32000 AND instr(reviewed_prompt, char(0)) = 0",
            name="ck_prompt_expansion_item_reviewed_prompt",
        ),
        sa.CheckConstraint("review_version > 0", name="ck_prompt_expansion_item_review_version"),
        sa.CheckConstraint("reroll_count >= 0", name="ck_prompt_expansion_item_reroll_count"),
    )
    op.create_index("ix_prompt_expansion_items_batch_id", "prompt_expansion_items", ["batch_id"])
    op.create_index(
        "ix_prompt_expansion_items_batch_ordinal",
        "prompt_expansion_items",
        ["batch_id", "ordinal"],
    )
    for statement in _TRIGGERS:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DROP_TRIGGERS:
        op.execute(statement)
    op.drop_index("ix_prompt_expansion_items_batch_ordinal", table_name="prompt_expansion_items")
    op.drop_index("ix_prompt_expansion_items_batch_id", table_name="prompt_expansion_items")
    op.drop_table("prompt_expansion_items")
    op.drop_index("ix_prompt_expansion_batches_chat_created", table_name="prompt_expansion_batches")
    op.drop_index(
        "ix_prompt_expansion_batches_prompt_template_revision_id",
        table_name="prompt_expansion_batches",
    )
    op.drop_index(
        "ix_prompt_expansion_batches_prompt_template_id",
        table_name="prompt_expansion_batches",
    )
    op.drop_index("ix_prompt_expansion_batches_chat_id", table_name="prompt_expansion_batches")
    op.drop_table("prompt_expansion_batches")
