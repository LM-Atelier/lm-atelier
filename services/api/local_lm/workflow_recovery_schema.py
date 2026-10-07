"""Fence deleted workflow identity against cached ORM and bulk SQL consumers."""

from __future__ import annotations

WORKFLOW_IDENTITY_KEYS = (
    "workflow_family_id",
    "family_id",
    "workflow_revision_id",
    "source_workflow_revision_id",
    "workflow_id",
    "revision_id",
    "id",
    "workflow_activation_id",
    "activation_id",
    "workflow_install_offer_id",
    "offer_id",
)


def _revision(column: str) -> str:
    return (
        f"{{row}}.{column} IN (SELECT r.id FROM workflow_revisions r "
        "JOIN workflow_definitions d ON d.id = r.workflow_id WHERE d.family_id = {family})"
    )


OWNED_WORKFLOW_SQL = {
    "workflow_families": "{row}.id = {family}",
    "workflow_definitions": "{row}.family_id = {family}",
    "workflow_revisions": (
        "{row}.workflow_id IN (SELECT id FROM workflow_definitions WHERE family_id = {family})"
    ),
    "workflow_preferences": "{row}.workflow_family_id = {family}",
    "workflow_profile_compatibility": "{row}.workflow_family_id = {family}",
    **{
        name: _revision("workflow_revision_id")
        for name in (
            "workflow_revision_reviews",
            "workflow_dependency_slots",
            "workflow_activations",
            "workflow_dependency_bindings",
            "workflow_trust_attestations",
        )
    },
}

DIRECT_WORKFLOW_COLUMNS = {
    "projects": ("image_workflow_revision_id", "video_workflow_revision_id"),
    "chat_workflow_selections": ("workflow_family_id",),
    "project_workflow_selections": ("workflow_family_id", "workflow_revision_id"),
    "runs": ("workflow_revision_id",),
    "work_steps": ("workflow_revision_id",),
    "setup_verifications": ("workflow_revision_id",),
    "edit_templates": ("workflow_revision_id",),
    "workflow_install_offers": ("workflow_revision_id",),
    "generation_experiment_arms": ("workflow_revision_id",),
}
JSON_WORKFLOW_COLUMNS = {
    "runs": ("settings_json", "provenance_json"),
    "work_steps": ("settings_json", "input_bindings_json"),
    "jobs": ("payload_json", "result_json"),
    "run_context_snapshots": ("payload_json",),
    "workflow_use_case_presets": ("settings_json",),
    "edit_templates": ("settings_json",),
    "prompt_template_revisions": ("contract_json",),
}
WORKFLOW_STATUS_COLUMNS = {
    "runs": "status",
    "work_steps": "status",
    "jobs": "status",
    "setup_verifications": "state",
    "workflow_install_offers": "status",
}


def workflow_consumer_sql(table: str, row: str, family: str) -> str:
    """Build a structural reference predicate from trusted table and SQL names."""
    targets = [
        (
            f"{row}.{column} = {family}"
            if column == "workflow_family_id"
            else _revision(column).format(row=row, family=family)
        )
        for column in DIRECT_WORKFLOW_COLUMNS.get(table, ())
    ]
    keys = ", ".join(f"'{key}'" for key in WORKFLOW_IDENTITY_KEYS)
    identities = (
        "SELECT recovery.subject_id UNION SELECT id FROM workflow_definitions "
        "WHERE family_id = recovery.subject_id UNION SELECT r.id FROM workflow_revisions r "
        "JOIN workflow_definitions d ON d.id = r.workflow_id "
        "WHERE d.family_id = recovery.subject_id "
        "UNION SELECT a.id FROM workflow_activations a "
        "JOIN workflow_revisions r ON r.id = a.workflow_revision_id "
        "JOIN workflow_definitions d ON d.id = r.workflow_id "
        "WHERE d.family_id = recovery.subject_id "
        "UNION SELECT o.id FROM workflow_install_offers o "
        "JOIN workflow_revisions r ON r.id = o.workflow_revision_id "
        "JOIN workflow_definitions d ON d.id = r.workflow_id "
        "WHERE d.family_id = recovery.subject_id"
    ).replace("recovery.subject_id", family)
    for column in JSON_WORKFLOW_COLUMNS.get(table, ()):
        targets.append(
            f"EXISTS (SELECT 1 FROM json_tree({row}.{column}) identity "
            f"WHERE identity.type = 'text' AND identity.key IN ({keys}) "
            f"AND identity.value IN ({identities}))"
        )
    return "(" + " OR ".join(targets) + ")"


def _consumer(table: str, row: str) -> str:
    target = workflow_consumer_sql(table, row, "recovery.subject_id")
    return (
        "EXISTS (SELECT 1 FROM recovery_items recovery WHERE recovery.kind = 'workflow_family' "
        f"AND {target})"
    )


def _owned(table: str, row: str, *, purging: bool = False) -> str:
    state = " AND recovery.state != 'purging'" if purging else ""
    owner = OWNED_WORKFLOW_SQL[table].format(row=row, family="recovery.subject_id")
    return (
        "EXISTS (SELECT 1 FROM recovery_items recovery WHERE recovery.kind = 'workflow_family' "
        f"AND ({owner}){state})"
    )


def _statements() -> tuple[str, ...]:
    statements: list[str] = []
    for table in OWNED_WORKFLOW_SQL:
        for action in ("INSERT", "UPDATE", "DELETE"):
            if action == "INSERT":
                blocked = _owned(table, "NEW")
            elif action == "DELETE":
                blocked = _owned(table, "OLD", purging=True)
            else:
                blocked = (
                    f"({_owned(table, 'OLD', purging=True)} "
                    f"OR {_owned(table, 'NEW', purging=True)})"
                )
            statements.append(
                f"CREATE TRIGGER workflow_recovery_{table}_{action.lower()} "
                f"BEFORE {action} ON {table} WHEN {blocked} "
                "BEGIN SELECT RAISE(ABORT, 'workflow-recovery-write-refused'); END"
            )
    for table in sorted(set(DIRECT_WORKFLOW_COLUMNS) | set(JSON_WORKFLOW_COLUMNS)):
        for action in ("INSERT", "UPDATE"):
            blocked = _consumer(table, "NEW")
            if table == "generation_experiment_arms":
                blocked = (
                    "NOT EXISTS (SELECT 1 FROM workflow_revisions r "
                    "JOIN workflow_definitions d ON d.id = r.workflow_id "
                    "JOIN workflow_families f ON f.id = d.family_id "
                    "WHERE r.id = NEW.workflow_revision_id AND NOT EXISTS "
                    "(SELECT 1 FROM recovery_items recovery "
                    "WHERE recovery.kind = 'workflow_family' "
                    "AND recovery.subject_id = f.id))"
                )
                if action == "UPDATE":
                    blocked += " AND NEW.workflow_revision_id IS NOT OLD.workflow_revision_id"
            elif action == "UPDATE" and table == "projects":
                changed = " OR ".join(
                    f"NEW.{column} IS NOT OLD.{column}" for column in DIRECT_WORKFLOW_COLUMNS[table]
                )
                blocked += f" AND ({changed})"
            elif action == "UPDATE" and table in WORKFLOW_STATUS_COLUMNS:
                status = WORKFLOW_STATUS_COLUMNS[table]
                columns = (
                    *DIRECT_WORKFLOW_COLUMNS.get(table, ()),
                    *JSON_WORKFLOW_COLUMNS.get(table, ()),
                )
                changed = " OR ".join(f"NEW.{column} IS NOT OLD.{column}" for column in columns)
                terminal = (
                    "'invalidated', 'completed', 'expired'"
                    if table == "workflow_install_offers"
                    else "'complete', 'failed', 'cancelled'"
                )
                blocked += f" AND (NEW.{status} NOT IN ({terminal}) OR {changed})"
            statements.append(
                f"CREATE TRIGGER workflow_recovery_{table}_{action.lower()} "
                f"BEFORE {action} ON {table} WHEN {blocked} "
                "BEGIN SELECT RAISE(ABORT, 'workflow-recovery-write-refused'); END"
            )
    return tuple(statements)


CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL = _statements()
DROP_WORKFLOW_RECOVERY_TRIGGER_SQL = tuple(
    f"DROP TRIGGER {statement.split()[2]}" for statement in CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL
)
