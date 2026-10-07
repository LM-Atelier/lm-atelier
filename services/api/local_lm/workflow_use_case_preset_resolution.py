"""Resolve recipe inheritance before selecting or admitting a workflow revision."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .chat_recovery_visibility import visible_chat
from .models import (
    Chat,
    ChatWorkflowUseCaseSelection,
    Project,
    ProjectWorkflowUseCaseSelection,
    WorkflowUseCasePreset,
)
from .project_recovery_visibility import effective_project_id, visible_project
from .workflow_use_cases_v1 import WorkflowUseCase

PresetScope = Literal["chat", "project", "workspace"]
ResolutionRefusal = Literal[
    "workflow-use-case-chat-not-found",
    "workflow-use-case-project-not-found",
    "workflow-use-case-project-mismatch",
    "workflow-use-case-preset-not-found",
    "workflow-use-case-preset-disabled",
    "workflow-use-case-preset-mismatch",
    "workflow-use-case-preset-settings-invalid",
]


class WorkflowUseCasePresetResolutionError(ValueError):
    def __init__(self, code: ResolutionRefusal) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ResolvedWorkflowUseCasePreset:
    """Detach selected settings; exact-revision validation is still required."""

    use_case: WorkflowUseCase
    mode: Literal["unconfigured", "automatic", "preset"]
    scope: PresetScope | None = None
    preset_id: str | None = None
    preset_name: str | None = None
    settings_json: dict[str, Any] = field(default_factory=dict)


def _recipe(
    session: Session, use_case: WorkflowUseCase, preset_id: str, scope: PresetScope
) -> ResolvedWorkflowUseCasePreset:
    row = session.execute(
        select(
            WorkflowUseCasePreset.id,
            WorkflowUseCasePreset.name,
            WorkflowUseCasePreset.use_case,
            WorkflowUseCasePreset.enabled,
            WorkflowUseCasePreset.settings_json,
        ).where(WorkflowUseCasePreset.id == preset_id)
    ).one_or_none()
    if row is None:
        raise WorkflowUseCasePresetResolutionError("workflow-use-case-preset-not-found")
    if row.use_case != use_case:
        raise WorkflowUseCasePresetResolutionError("workflow-use-case-preset-mismatch")
    if not row.enabled:
        raise WorkflowUseCasePresetResolutionError("workflow-use-case-preset-disabled")
    if not isinstance(row.settings_json, dict):
        raise WorkflowUseCasePresetResolutionError("workflow-use-case-preset-settings-invalid")
    return ResolvedWorkflowUseCasePreset(
        use_case, "preset", scope, row.id, row.name, deepcopy(row.settings_json)
    )


def resolve_workflow_use_case_preset(
    session: Session,
    use_case: WorkflowUseCase,
    *,
    chat_id: str | None = None,
    project_id: str | None = None,
) -> ResolvedWorkflowUseCasePreset:
    """Resolve chat, then project, then workspace within the caller's transaction.

    Automatic stops inheritance and contributes no settings. Missing or invalid
    selected recipes refuse instead of silently choosing a lower scope. This
    function neither commits nor flushes pending edits.
    """
    connection = session.connection()
    if connection.dialect.name == "sqlite":
        driver = connection.connection.driver_connection
        if not bool(getattr(driver, "in_transaction", False)):
            connection.exec_driver_sql("BEGIN")
    with session.no_autoflush:
        if chat_id is not None:
            chat = session.execute(
                select(
                    Chat.project_id,
                    effective_project_id(Chat.project_id).label("effective_project_id"),
                ).where(Chat.id == chat_id, visible_chat(Chat.id))
            ).one_or_none()
            if chat is None:
                raise WorkflowUseCasePresetResolutionError("workflow-use-case-chat-not-found")
            if project_id is not None and project_id != chat.project_id:
                raise WorkflowUseCasePresetResolutionError("workflow-use-case-project-mismatch")
            if (
                project_id is not None
                and session.scalar(
                    select(Project.id).where(Project.id == project_id, visible_project(Project.id))
                )
                is None
            ):
                raise WorkflowUseCasePresetResolutionError("workflow-use-case-project-not-found")
            project_id = chat.effective_project_id
            choice = session.execute(
                select(ChatWorkflowUseCaseSelection.preset_id).where(
                    ChatWorkflowUseCaseSelection.chat_id == chat_id,
                    ChatWorkflowUseCaseSelection.use_case == use_case,
                )
            ).one_or_none()
            if choice is not None:
                if choice.preset_id is None:
                    return ResolvedWorkflowUseCasePreset(use_case, "automatic", "chat")
                return _recipe(session, use_case, choice.preset_id, "chat")
        if project_id is not None:
            if (
                session.scalar(
                    select(Project.id).where(Project.id == project_id, visible_project(Project.id))
                )
                is None
            ):
                raise WorkflowUseCasePresetResolutionError("workflow-use-case-project-not-found")
            choice = session.execute(
                select(ProjectWorkflowUseCaseSelection.preset_id).where(
                    ProjectWorkflowUseCaseSelection.project_id == project_id,
                    ProjectWorkflowUseCaseSelection.use_case == use_case,
                )
            ).one_or_none()
            if choice is not None:
                if choice.preset_id is None:
                    return ResolvedWorkflowUseCasePreset(use_case, "automatic", "project")
                return _recipe(session, use_case, choice.preset_id, "project")
        default_id = session.scalar(
            select(WorkflowUseCasePreset.id).where(
                WorkflowUseCasePreset.use_case == use_case,
                WorkflowUseCasePreset.is_default.is_(True),
            )
        )
        if default_id is not None:
            return _recipe(session, use_case, default_id, "workspace")
        return ResolvedWorkflowUseCasePreset(use_case, "unconfigured")
