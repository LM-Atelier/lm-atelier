"""SQLite guards for durable Prompt Library expansion drafts.

Alembic revisions embed their own copies of these statements.  This module is
only the live ``create_all`` authority; keeping migrations self-contained means
a future edit here cannot silently rewrite historical schema.
"""

from __future__ import annotations

MAX_EXPANSION_REQUEST_JSON_CHARS = 1_200_000
MAX_EXPANSION_MODEL_SNAPSHOT_JSON_CHARS = 1_024
MAX_EXPANSION_EVIDENCE_JSON_CHARS = 262_144
MEDIA_SEED_SPACE = 2_147_483_648

BATCH_INSERT_TRIGGER = f"""
CREATE TRIGGER prompt_expansion_batch_insert_guard
BEFORE INSERT ON prompt_expansion_batches
BEGIN
  SELECT CASE WHEN NOT json_valid(NEW.request_json)
                        OR json_type(NEW.request_json) != 'object'
                        OR json_type(NEW.request_json, '$.item_count') != 'integer'
                        OR json_extract(NEW.request_json, '$.item_count') NOT BETWEEN 1 AND 16
                        OR length(NEW.request_json) > {MAX_EXPANSION_REQUEST_JSON_CHARS}
                        OR NOT json_valid(NEW.model_snapshot_json)
                        OR json_type(NEW.model_snapshot_json) != 'object'
                        OR json_type(NEW.model_snapshot_json, '$.version') != 'integer'
                        OR json_extract(NEW.model_snapshot_json, '$.version') != 1
                        OR length(NEW.model_snapshot_json) > 1024
                        OR NEW.plan_version != 1
                        OR NEW.state != 'draft'
                        OR NEW.original_plan_sha256 IS NOT NEW.plan_sha256
                        OR NEW.queue_idempotency_key IS NOT NULL
                        OR NEW.work_plan_id IS NOT NULL
                        OR NEW.queued_at IS NOT NULL
    THEN RAISE(ABORT, 'prompt expansion JSON is invalid') END;
END
"""

BATCH_UPDATE_TRIGGER = """
CREATE TRIGGER prompt_expansion_batch_update_guard
BEFORE UPDATE ON prompt_expansion_batches
BEGIN
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
  SELECT CASE WHEN OLD.state = 'draft' AND NEW.state = 'draft'
                        AND (NEW.plan_version IS NOT OLD.plan_version + 1
                             OR NEW.queue_idempotency_key IS NOT OLD.queue_idempotency_key
                             OR NEW.work_plan_id IS NOT OLD.work_plan_id
                             OR NEW.queued_at IS NOT OLD.queued_at)
    THEN RAISE(ABORT, 'prompt expansion draft update is invalid') END;
  SELECT CASE WHEN OLD.state = 'draft' AND NEW.state = 'queued'
                        AND (NEW.plan_version IS NOT OLD.plan_version + 1
                             OR (NEW.queue_idempotency_key IS NOT NULL
                                 AND (typeof(NEW.queue_idempotency_key) != 'text'
                                      OR length(NEW.queue_idempotency_key)
                                         NOT BETWEEN 1 AND 200
                                      OR instr(NEW.queue_idempotency_key, char(0)) != 0))
                             OR NEW.work_plan_id IS NOT NULL
                             OR NEW.queued_at IS NOT NULL)
    THEN RAISE(ABORT, 'prompt expansion queue claim is invalid') END;
  SELECT CASE WHEN OLD.state = 'queued'
                        AND NOT (
                          NEW.state IS OLD.state
                          AND NEW.plan_version IS OLD.plan_version
                          AND NEW.plan_sha256 IS OLD.plan_sha256
                          AND NEW.queue_idempotency_key IS OLD.queue_idempotency_key
                          AND OLD.queue_idempotency_key IS NOT NULL
                          AND OLD.work_plan_id IS NULL
                          AND NEW.work_plan_id IS NOT NULL
                          AND OLD.queued_at IS NULL
                          AND NEW.queued_at IS NOT NULL
                        )
    THEN RAISE(ABORT, 'queued prompt expansion batches are immutable') END;
END
"""

ITEM_INSERT_TRIGGER = f"""
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
                           > {MAX_EXPANSION_EVIDENCE_JSON_CHARS}
                        OR NEW.work_step_id IS NOT NULL
                        OR NEW.run_id IS NOT NULL
                        OR NEW.media_seed IS NOT NULL
    THEN RAISE(ABORT, 'prompt expansion item initial state is invalid') END;
END
"""

ITEM_UPDATE_TRIGGER = f"""
CREATE TRIGGER prompt_expansion_item_update_guard
BEFORE UPDATE ON prompt_expansion_items
BEGIN
  SELECT CASE WHEN NEW.id IS NOT OLD.id
                        OR NEW.batch_id IS NOT OLD.batch_id
                        OR NEW.ordinal IS NOT OLD.ordinal
                        OR NEW.original_evidence_json IS NOT OLD.original_evidence_json
                        OR NEW.original_rendered_prompt IS NOT OLD.original_rendered_prompt
                        OR NEW.original_rendered_sha256 IS NOT OLD.original_rendered_sha256
                        OR NEW.created_at IS NOT OLD.created_at
    THEN RAISE(ABORT, 'prompt expansion item origin is immutable') END;
  SELECT CASE WHEN COALESCE((
    SELECT batch.state FROM prompt_expansion_batches AS batch
    WHERE batch.id = OLD.batch_id
  ), 'missing') = 'draft'
                        AND (NEW.review_version IS NOT OLD.review_version + 1
                             OR NEW.reroll_count NOT IN (
                               OLD.reroll_count, OLD.reroll_count + 1
                             )
                             OR typeof(NEW.selected) != 'integer'
                             OR NEW.selected NOT IN (0, 1)
                             OR (NEW.reroll_count = OLD.reroll_count
                                 AND NEW.current_evidence_json
                                     IS NOT OLD.current_evidence_json)
                             OR NOT json_valid(NEW.current_evidence_json)
                             OR json_type(NEW.current_evidence_json) != 'array'
                             OR length(NEW.current_evidence_json)
                                > {MAX_EXPANSION_EVIDENCE_JSON_CHARS}
                             OR NEW.work_step_id IS NOT OLD.work_step_id
                             OR NEW.run_id IS NOT OLD.run_id
                             OR NEW.media_seed IS NOT OLD.media_seed)
    THEN RAISE(ABORT, 'prompt expansion item update is invalid') END;
  SELECT CASE WHEN COALESCE((
    SELECT batch.state FROM prompt_expansion_batches AS batch
    WHERE batch.id = OLD.batch_id
  ), 'missing') = 'queued'
                        AND NOT (
                          OLD.selected = 1
                          AND NEW.current_evidence_json IS OLD.current_evidence_json
                          AND NEW.reviewed_prompt IS OLD.reviewed_prompt
                          AND NEW.reviewed_sha256 IS OLD.reviewed_sha256
                          AND NEW.selected IS OLD.selected
                          AND NEW.review_version IS OLD.review_version
                          AND NEW.reroll_count IS OLD.reroll_count
                          AND OLD.work_step_id IS NULL
                          AND NEW.work_step_id IS NOT NULL
                          AND OLD.run_id IS NULL
                          AND NEW.run_id IS NOT NULL
                          AND OLD.media_seed IS NULL
                          AND typeof(NEW.media_seed) = 'integer'
                          AND NEW.media_seed >= 0
                          AND NEW.media_seed < {MEDIA_SEED_SPACE}
                          AND (SELECT step.plan_id FROM work_steps AS step
                               WHERE step.id = NEW.work_step_id) = (
                            SELECT batch.work_plan_id
                            FROM prompt_expansion_batches AS batch
                            WHERE batch.id = OLD.batch_id
                          )
                          AND (SELECT run.work_step_id FROM runs AS run
                               WHERE run.id = NEW.run_id) = NEW.work_step_id
                        )
    THEN RAISE(ABORT, 'prompt expansion execution link is invalid') END;
  SELECT CASE WHEN COALESCE((
    SELECT batch.state FROM prompt_expansion_batches AS batch
    WHERE batch.id = OLD.batch_id
  ), 'missing') NOT IN ('draft', 'queued')
    THEN RAISE(ABORT, 'prompt expansion item parent is invalid') END;
END
"""

CREATE_PROMPT_EXPANSION_TRIGGER_SQL = (
    BATCH_INSERT_TRIGGER,
    BATCH_UPDATE_TRIGGER,
    ITEM_INSERT_TRIGGER,
    ITEM_UPDATE_TRIGGER,
)

DROP_PROMPT_EXPANSION_TRIGGER_SQL = (
    "DROP TRIGGER prompt_expansion_item_update_guard",
    "DROP TRIGGER prompt_expansion_item_insert_guard",
    "DROP TRIGGER prompt_expansion_batch_update_guard",
    "DROP TRIGGER prompt_expansion_batch_insert_guard",
)
