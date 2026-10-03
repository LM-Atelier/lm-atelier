"""Preserve deleted project configuration without freezing its retained conversations."""

PROJECT_COLUMNS = (
    "id",
    "name",
    "description",
    "instructions",
    "archived",
    "pinned",
    "image_workflow_revision_id",
    "video_workflow_revision_id",
    "generation_settings_json",
    "generation_preset_ids_json",
    "created_at",
    "updated_at",
)
PROJECT_OWNER_COLUMNS = {
    "projects": "id",
    "project_workflow_selections": "project_id",
    "project_workflow_use_case_selections": "project_id",
}


def _blocked(table: str, row: str, *, deleting: bool = False) -> str:
    state = "AND recovery.state != 'purging'" if deleting else ""
    return (
        "EXISTS (SELECT 1 FROM recovery_items recovery WHERE recovery.kind = 'project' "
        f"AND recovery.subject_id = {row}.{PROJECT_OWNER_COLUMNS[table]} {state})"
    )


def _vanished_workflow() -> str:
    links = ("image_workflow_revision_id", "video_workflow_revision_id")
    unchanged = [
        f"NEW.{column} IS OLD.{column}"
        for column in PROJECT_COLUMNS
        if column not in (*links, "updated_at")
    ]
    gone = [
        f"(OLD.{column} IS NOT NULL AND NEW.{column} IS NULL "
        f"AND NOT EXISTS (SELECT 1 FROM workflow_revisions WHERE id = OLD.{column}))"
        for column in links
    ]
    permitted = [
        f"(NEW.{column} IS OLD.{column} OR {missing})"
        for column, missing in zip(links, gone, strict=True)
    ]
    return " AND ".join((*unchanged, *permitted, "(" + " OR ".join(gone) + ")"))


def _statements() -> tuple[str, ...]:
    statements: list[str] = []
    for table in PROJECT_OWNER_COLUMNS:
        for action in ("INSERT", "UPDATE", "DELETE"):
            if action == "INSERT":
                blocked = _blocked(table, "NEW")
            elif action == "DELETE":
                blocked = _blocked(table, "OLD", deleting=True)
            else:
                blocked = f"({_blocked(table, 'OLD')} OR {_blocked(table, 'NEW')})"
                if table == "projects":
                    blocked += f" AND NOT ({_vanished_workflow()})"
            statements.append(
                f"CREATE TRIGGER project_recovery_{table}_{action.lower()} "
                f"BEFORE {action} ON {table} WHEN {blocked} "
                "BEGIN SELECT RAISE(ABORT, 'project-recovery-write-refused'); END"
            )
    deleted_target = (
        "EXISTS (SELECT 1 FROM recovery_items WHERE kind = 'project' "
        "AND subject_id = NEW.project_id)"
    )
    for action, predicate in (
        ("INSERT", deleted_target),
        ("UPDATE", f"NEW.project_id IS NOT OLD.project_id AND {deleted_target}"),
    ):
        statements.append(
            f"CREATE TRIGGER project_recovery_chats_{action.lower()} BEFORE {action} ON chats "
            f"WHEN {predicate} BEGIN SELECT RAISE(ABORT, 'project-recovery-write-refused'); END"
        )
    return tuple(statements)


CREATE_PROJECT_RECOVERY_TRIGGER_SQL = _statements()
DROP_PROJECT_RECOVERY_TRIGGER_SQL = tuple(
    f"DROP TRIGGER {statement.split()[2]}" for statement in CREATE_PROJECT_RECOVERY_TRIGGER_SQL
)
