"""allow queued prompt execution links to follow referential deletion

Revision ID: f2a7c9d41e63
Revises: b1d9e4c72f60
Create Date: 2026-08-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f2a7c9d41e63"
down_revision: str | None = "b1d9e4c72f60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _batch_update_trigger(*, allow_deleted_work_plan_unlink: bool) -> str:
    deleted_work_plan_unlink = (
        """
                          OR (
                            NEW.state IS OLD.state
                            AND NEW.plan_version IS OLD.plan_version
                            AND NEW.plan_sha256 IS OLD.plan_sha256
                            AND NEW.queue_idempotency_key IS OLD.queue_idempotency_key
                            AND OLD.work_plan_id IS NOT NULL
                            AND NEW.work_plan_id IS NULL
                            AND NEW.queued_at IS OLD.queued_at
                            AND NOT EXISTS (
                              SELECT 1 FROM work_plans AS plan
                              WHERE plan.id = OLD.work_plan_id
                            )
                          )"""
        if allow_deleted_work_plan_unlink
        else ""
    )
    return f"""
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
                          (
                            NEW.state IS OLD.state
                            AND NEW.plan_version IS OLD.plan_version
                            AND NEW.plan_sha256 IS OLD.plan_sha256
                            AND NEW.queue_idempotency_key IS OLD.queue_idempotency_key
                            AND OLD.queue_idempotency_key IS NOT NULL
                            AND OLD.work_plan_id IS NULL
                            AND NEW.work_plan_id IS NOT NULL
                            AND OLD.queued_at IS NULL
                            AND NEW.queued_at IS NOT NULL
                          ){deleted_work_plan_unlink}
                        )
    THEN RAISE(ABORT, 'queued prompt expansion batches are immutable') END;
END
"""


def _item_update_trigger(*, allow_deleted_run_unlink: bool) -> str:
    deleted_run_unlink = (
        """
                          OR (
                            NEW.current_evidence_json IS OLD.current_evidence_json
                            AND NEW.reviewed_prompt IS OLD.reviewed_prompt
                            AND NEW.reviewed_sha256 IS OLD.reviewed_sha256
                            AND NEW.selected IS OLD.selected
                            AND NEW.review_version IS OLD.review_version
                            AND NEW.reroll_count IS OLD.reroll_count
                            AND NEW.work_step_id IS OLD.work_step_id
                            AND OLD.run_id IS NOT NULL
                            AND NEW.run_id IS NULL
                            AND NEW.media_seed IS OLD.media_seed
                            AND NOT EXISTS (
                              SELECT 1 FROM runs AS run
                              WHERE run.id = OLD.run_id
                            )
                          )"""
        if allow_deleted_run_unlink
        else ""
    )
    deleted_work_step_unlink = (
        """
                          OR (
                            NEW.current_evidence_json IS OLD.current_evidence_json
                            AND NEW.reviewed_prompt IS OLD.reviewed_prompt
                            AND NEW.reviewed_sha256 IS OLD.reviewed_sha256
                            AND NEW.selected IS OLD.selected
                            AND NEW.review_version IS OLD.review_version
                            AND NEW.reroll_count IS OLD.reroll_count
                            AND OLD.work_step_id IS NOT NULL
                            AND NEW.work_step_id IS NULL
                            AND NEW.run_id IS OLD.run_id
                            AND NEW.media_seed IS OLD.media_seed
                            AND NOT EXISTS (
                              SELECT 1 FROM work_steps AS step
                              WHERE step.id = OLD.work_step_id
                            )
                          )"""
        if allow_deleted_run_unlink
        else ""
    )
    return f"""
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
                                > 262144
                             OR NEW.work_step_id IS NOT OLD.work_step_id
                             OR NEW.run_id IS NOT OLD.run_id
                             OR NEW.media_seed IS NOT OLD.media_seed)
    THEN RAISE(ABORT, 'prompt expansion item update is invalid') END;
  SELECT CASE WHEN COALESCE((
    SELECT batch.state FROM prompt_expansion_batches AS batch
    WHERE batch.id = OLD.batch_id
  ), 'missing') = 'queued'
                        AND NOT (
                          (
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
                            AND NEW.media_seed < 2147483648
                            AND (SELECT step.plan_id FROM work_steps AS step
                                 WHERE step.id = NEW.work_step_id) = (
                              SELECT batch.work_plan_id
                              FROM prompt_expansion_batches AS batch
                              WHERE batch.id = OLD.batch_id
                            )
                            AND (SELECT run.work_step_id FROM runs AS run
                                 WHERE run.id = NEW.run_id) = NEW.work_step_id
                          ){deleted_run_unlink}{deleted_work_step_unlink}
                        )
    THEN RAISE(ABORT, 'prompt expansion execution link is invalid') END;
  SELECT CASE WHEN COALESCE((
    SELECT batch.state FROM prompt_expansion_batches AS batch
    WHERE batch.id = OLD.batch_id
  ), 'missing') NOT IN ('draft', 'queued')
    THEN RAISE(ABORT, 'prompt expansion item parent is invalid') END;
END
"""


def _replace_update_triggers(*, allow_referential_unlink: bool) -> None:
    op.execute("DROP TRIGGER prompt_expansion_item_update_guard")
    op.execute("DROP TRIGGER prompt_expansion_batch_update_guard")
    op.execute(_batch_update_trigger(allow_deleted_work_plan_unlink=allow_referential_unlink))
    op.execute(_item_update_trigger(allow_deleted_run_unlink=allow_referential_unlink))


def upgrade() -> None:
    _replace_update_triggers(allow_referential_unlink=True)


def downgrade() -> None:
    _replace_update_triggers(allow_referential_unlink=False)
