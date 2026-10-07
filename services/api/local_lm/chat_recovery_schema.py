"""Keep a recoverable conversation immutable, including writers that use bulk SQL."""

from __future__ import annotations

from collections.abc import Mapping


def _via(column: str, table: str, key: str = "id") -> str:
    return f"{{row}}.{column} IN (SELECT {key} FROM {table} WHERE {table}.chat_id = {{chat}})"


def _step(column: str) -> str:
    return (
        f"{{row}}.{column} IN (SELECT s.id FROM work_steps s JOIN work_plans p "
        "ON p.id = s.plan_id WHERE p.chat_id = {chat})"
    )


def _revision(column: str) -> str:
    return (
        f"{{row}}.{column} IN (SELECT v.id FROM response_revisions v JOIN messages m "
        "ON m.id = v.message_id WHERE m.chat_id = {chat})"
    )


_JOB_OWNER = " OR ".join(
    (
        _via("run_id", "runs"),
        _via("work_plan_id", "work_plans"),
        _step("work_step_id"),
        "json_extract({row}.payload_json, '$.chat_id') = {chat}",
        "json_extract({row}.payload_json, '$.source_run_id') "
        "IN (SELECT id FROM runs WHERE runs.chat_id = {chat})",
    )
)


def _job(column: str) -> str:
    owner = _JOB_OWNER.format(row="j", chat="{chat}")
    return f"{{row}}.{column} IN (SELECT j.id FROM jobs j WHERE ({owner}))"


# Each path is an attachment to a conversation, even when the row has another
# parent too. Both the old and new attachment are checked on a move.
CHAT_OWNER_SQL: Mapping[str, str] = {
    "chats": "{row}.id = {chat}",
    "chat_composer_drafts": "{row}.chat_id = {chat}",
    "chat_composer_draft_attachments": "{row}.chat_id = {chat}",
    "chat_workflow_selections": "{row}.chat_id = {chat}",
    "chat_workflow_use_case_selections": "{row}.chat_id = {chat}",
    "messages": "{row}.chat_id = {chat} OR " + _via("parent_id", "messages"),
    "message_parts": _via("message_id", "messages"),
    "message_references": _via("message_id", "messages"),
    "chat_item_removal_receipts": "{row}.chat_id = {chat} OR " + _via("message_id", "messages"),
    "runs": " OR ".join(
        (
            "{row}.chat_id = {chat}",
            _via("user_message_id", "messages"),
            _via("assistant_message_id", "messages"),
            _via("work_plan_id", "work_plans"),
            _step("work_step_id"),
        )
    ),
    "run_context_snapshots": _via("run_id", "runs"),
    "run_context_artifacts": "{row}.run_id IN (SELECT run_id FROM run_context_snapshots "
    "WHERE run_id IN (SELECT id FROM runs WHERE runs.chat_id = {chat}))",
    "response_revisions": _via("message_id", "messages") + " OR " + _via("run_id", "runs"),
    "response_revision_parts": _revision("response_revision_id"),
    "response_feedback": " OR ".join(
        (_via("message_id", "messages"), _via("run_id", "runs"), _revision("response_revision_id"))
    ),
    "chat_activity_events": " OR ".join(
        (
            "{row}.chat_id = {chat}",
            _via("message_id", "messages"),
            _revision("response_revision_id"),
        )
    ),
    "work_plans": "{row}.chat_id = {chat} OR " + _via("context_head_message_id", "messages"),
    "work_steps": _via("plan_id", "work_plans") + " OR " + _via("run_id", "runs"),
    "work_step_dependencies": _step("step_id") + " OR " + _step("depends_on_step_id"),
    "work_plan_controls": _via("plan_id", "work_plans"),
    "work_plan_control_receipts": _via("plan_id", "work_plans"),
    "turn_creation_claims": "{row}.chat_id = {chat}",
    "prompt_expansion_batches": "{row}.chat_id = {chat} OR " + _via("work_plan_id", "work_plans"),
    "prompt_expansion_items": " OR ".join(
        (
            _via("batch_id", "prompt_expansion_batches"),
            _via("run_id", "runs"),
            _step("work_step_id"),
        )
    ),
    "jobs": _JOB_OWNER,
    "web_search_proposals": _via("run_id", "runs") + " OR " + _job("job_id"),
    "setup_verifications": " OR ".join(
        ("{row}.chat_id = {chat}", _via("run_id", "runs"), _job("job_id"))
    ),
}

# These resources retain an independent lifecycle. Their foreign keys are
# historical links, rather than ownership of the conversation's mutable graph.
INDEPENDENT_REFERENCES = {
    "generation_experiments",
    "generation_experiment_trials",
    "workflow_install_offers",
    "workflow_install_offer_downloads",
    "workflow_install_offer_packages",
}


# Pin the compared columns so a new field cannot bypass a cascade-only update.
ROW_COLUMNS = {
    "chats": (
        "id",
        "project_id",
        "title",
        "archived",
        "pinned",
        "scope",
        "draft_prompt",
        "routing_mode",
        "confirm_uncertain_media",
        "active_chat_profile_id",
        "active_vision_profile_id",
        "active_image_profile_id",
        "active_video_profile_id",
        "active_head_message_id",
        "generation_settings_json",
        "generation_preset_ids_json",
        "vision_settings_json",
        "web_settings_json",
        "origin_json",
        "created_at",
        "updated_at",
    ),
    "chat_composer_drafts": (
        "chat_id",
        "revision",
        "text",
        "prompt_source_json",
        "mode",
        "output_count",
        "mentions_json",
        "template_settings_json",
        "created_at",
        "updated_at",
    ),
    "chat_composer_draft_attachments": (
        "chat_id",
        "position",
        "artifact_id",
        "kind",
        "origin",
        "image_role",
    ),
    "chat_workflow_selections": (
        "id",
        "chat_id",
        "selector_capability",
        "mode",
        "workflow_family_id",
        "created_at",
        "updated_at",
    ),
    "chat_workflow_use_case_selections": (
        "chat_id",
        "use_case",
        "preset_id",
        "created_at",
        "updated_at",
    ),
    "messages": (
        "id",
        "chat_id",
        "parent_id",
        "role",
        "status",
        "transcript_visible",
        "content_removed_at",
        "active_response_revision_id",
        "created_at",
        "updated_at",
    ),
    "message_parts": (
        "id",
        "message_id",
        "position",
        "type",
        "text",
        "artifact_id",
        "metadata_json",
        "created_at",
        "updated_at",
    ),
    "message_references": (
        "id",
        "message_id",
        "position",
        "reference_subject_id",
        "mention_slug",
        "subject_name",
        "subject_kind",
        "role",
        "strength",
        "source",
        "reference_asset_ids_json",
        "artifact_ids_json",
        "created_at",
        "updated_at",
    ),
    "chat_item_removal_receipts": (
        "id",
        "chat_id",
        "operation_key",
        "message_id",
        "request_sha256",
        "message_revision_id",
        "content_removed_at",
        "created_at",
        "updated_at",
    ),
    "runs": (
        "id",
        "idempotency_key",
        "chat_id",
        "user_message_id",
        "assistant_message_id",
        "work_plan_id",
        "work_step_id",
        "operation",
        "status",
        "standalone_prompt",
        "profile_id",
        "vision_profile_id",
        "workflow_revision_id",
        "settings_json",
        "provenance_json",
        "error",
        "started_at",
        "completed_at",
        "duration_ms",
        "created_at",
        "updated_at",
    ),
    "run_context_snapshots": ("run_id", "payload_json", "sha256"),
    "run_context_artifacts": ("run_id", "artifact_id"),
    "response_revisions": (
        "id",
        "message_id",
        "run_id",
        "sequence",
        "status",
        "activity_json",
        "created_at",
        "updated_at",
    ),
    "response_revision_parts": (
        "id",
        "response_revision_id",
        "position",
        "type",
        "text",
        "artifact_id",
        "metadata_json",
        "created_at",
        "updated_at",
    ),
    "response_feedback": (
        "id",
        "message_id",
        "response_revision_id",
        "run_id",
        "rating",
        "created_at",
        "updated_at",
    ),
    "chat_activity_events": (
        "sequence",
        "id",
        "chat_id",
        "message_id",
        "response_revision_id",
        "job_id",
        "attempt",
        "kind",
        "occurred_at",
    ),
    "work_plans": (
        "id",
        "chat_id",
        "idempotency_key",
        "source_action",
        "persistence_scope",
        "status",
        "context_head_message_id",
        "transcript_sequence",
        "priority",
        "planner_version",
        "failure_policy",
        "summary_json",
        "created_at",
        "updated_at",
    ),
    "work_steps": (
        "id",
        "plan_id",
        "run_id",
        "ordinal",
        "display_group",
        "operation",
        "status",
        "prompt",
        "profile_id",
        "workflow_revision_id",
        "settings_json",
        "input_bindings_json",
        "output_contract_json",
        "queue_class",
        "error",
        "created_at",
        "updated_at",
    ),
    "work_step_dependencies": ("step_id", "depends_on_step_id"),
    "work_plan_controls": ("plan_id", "state", "revision", "eligible_since"),
    "work_plan_control_receipts": (
        "plan_id",
        "command_key",
        "action",
        "expected_revision",
        "response_json",
    ),
    "turn_creation_claims": ("id", "chat_id", "idempotency_key", "owner_token", "created_at"),
    "prompt_expansion_batches": (
        "id",
        "chat_id",
        "idempotency_key",
        "prompt_template_id",
        "prompt_template_revision_id",
        "schema_version",
        "contract_sha256",
        "codec_version",
        "request_json",
        "model_snapshot_json",
        "original_plan_sha256",
        "plan_sha256",
        "plan_version",
        "state",
        "queue_idempotency_key",
        "work_plan_id",
        "queued_at",
        "created_at",
        "updated_at",
    ),
    "prompt_expansion_items": (
        "id",
        "batch_id",
        "ordinal",
        "original_evidence_json",
        "current_evidence_json",
        "original_rendered_prompt",
        "original_rendered_sha256",
        "reviewed_prompt",
        "reviewed_sha256",
        "selected",
        "review_version",
        "reroll_count",
        "work_step_id",
        "run_id",
        "media_seed",
        "created_at",
        "updated_at",
    ),
    "jobs": (
        "id",
        "kind",
        "status",
        "run_id",
        "work_plan_id",
        "work_step_id",
        "progress",
        "phase",
        "progress_json",
        "queue_resource",
        "queue_group",
        "queue_priority",
        "queue_ticket",
        "enqueued_at",
        "claim_owner",
        "claim_expires_at",
        "heartbeat_at",
        "payload_json",
        "result_json",
        "error",
        "attempt",
        "cancellable",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    ),
    "web_search_proposals": (
        "id",
        "job_id",
        "run_id",
        "revision",
        "state",
        "query",
        "provider_endpoint",
        "provider_revision",
        "approved_automatically",
        "dispatch_after",
        "dispatch_owner",
        "dispatch_attempt",
        "result_json",
        "error_code",
        "created_at",
        "updated_at",
    ),
    "setup_verifications": (
        "id",
        "role",
        "evidence_key",
        "state",
        "model_install_id",
        "profile_id",
        "workflow_revision_id",
        "chat_id",
        "run_id",
        "job_id",
        "input_artifact_id",
        "failure_code",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    ),
}

SET_NULL_LINKS = {
    "chats": (("project_id", "projects", "id"),),
    "messages": (("parent_id", "messages", "id"),),
    "message_parts": (("artifact_id", "artifacts", "id"),),
    "runs": (("work_plan_id", "work_plans", "id"), ("work_step_id", "work_steps", "id")),
    "response_revisions": (("run_id", "runs", "id"),),
    "response_revision_parts": (("artifact_id", "artifacts", "id"),),
    "response_feedback": (("run_id", "runs", "id"),),
    "work_plans": (("context_head_message_id", "messages", "id"),),
    "work_steps": (("run_id", "runs", "id"),),
    "prompt_expansion_batches": (("work_plan_id", "work_plans", "id"),),
    "prompt_expansion_items": (("run_id", "runs", "id"), ("work_step_id", "work_steps", "id")),
    "jobs": (
        ("run_id", "runs", "id"),
        ("work_plan_id", "work_plans", "id"),
        ("work_step_id", "work_steps", "id"),
    ),
    "setup_verifications": (("input_artifact_id", "artifacts", "id"),),
}


def owner_predicate(table: str, row: str, chat: str) -> str:
    """Resolve an audited SQL ownership expression using fixed application names."""
    return CHAT_OWNER_SQL[table].format(row=row, chat=chat)


def _blocked(table: str, row: str, *, deleting: bool = False) -> str:
    ownership = owner_predicate(table, row, "recovery.subject_id")
    state = "AND recovery.state != 'purging'" if deleting else ""
    return (
        "EXISTS (SELECT 1 FROM recovery_items recovery "
        f"WHERE recovery.kind = 'chat' AND ({ownership}) {state})"
    )


def _cascade_nullification(table: str) -> str:
    """Allow only a vanished parent's nullable link to be cleared by its cascade."""
    links = SET_NULL_LINKS.get(table, ())
    if not links:
        return "0"
    changed_columns = {column for column, _parent, _key in links}
    unchanged = [
        f"NEW.{column} IS OLD.{column}"
        for column in ROW_COLUMNS[table]
        if column not in changed_columns and column != "updated_at"
    ]
    gone = [
        f"(OLD.{column} IS NOT NULL AND NEW.{column} IS NULL "
        f"AND NOT EXISTS (SELECT 1 FROM {parent} WHERE {key} = OLD.{column}))"
        for column, parent, key in links
    ]
    permitted = [
        f"(NEW.{column} IS OLD.{column} OR {missing})"
        for (column, _parent, _key), missing in zip(links, gone, strict=True)
    ]
    return " AND ".join((*unchanged, *permitted, "(" + " OR ".join(gone) + ")"))


def _statements() -> tuple[str, ...]:
    statements: list[str] = []
    for table in CHAT_OWNER_SQL:
        for action in ("INSERT", "UPDATE", "DELETE"):
            if action == "INSERT":
                blocked = _blocked(table, "NEW")
            elif action == "DELETE":
                blocked = _blocked(table, "OLD", deleting=True)
            else:
                blocked = (
                    f"({_blocked(table, 'OLD')} OR {_blocked(table, 'NEW')}) "
                    f"AND NOT ({_cascade_nullification(table)})"
                )
            statements.append(
                f"CREATE TRIGGER chat_recovery_{table}_{action.lower()} "
                f"BEFORE {action} ON {table} WHEN {blocked} "
                "BEGIN SELECT RAISE(ABORT, 'chat-recovery-write-refused'); END"
            )
    return tuple(statements)


CREATE_CHAT_RECOVERY_TRIGGER_SQL = _statements()
DROP_CHAT_RECOVERY_TRIGGER_SQL = tuple(
    f"DROP TRIGGER {statement.split()[2]}" for statement in CREATE_CHAT_RECOVERY_TRIGGER_SQL
)
