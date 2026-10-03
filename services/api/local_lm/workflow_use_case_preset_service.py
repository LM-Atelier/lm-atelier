"""Manage recipes and scoped choices in atomic database transactions."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue, ValidationError
from sqlalchemy import delete, exists, or_, select, text, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from .chat_recovery_visibility import visible_chat
from .models import (
    Chat,
    ChatWorkflowUseCaseSelection,
    Project,
    ProjectWorkflowUseCaseSelection,
    WorkflowUseCasePreset,
)
from .project_recovery_visibility import visible_project
from .workflow_use_case_presets_v1 import WorkflowUseCaseChoice, WorkflowUseCasePresetCreate
from .workflow_use_cases_v1 import WorkflowUseCase

SelectionScope = Literal["chat", "project"]
ServiceRefusal = Literal[
    "workflow-use-case-preset-conflict",
    "workflow-use-case-preset-not-found",
    "workflow-use-case-preset-builtin",
    "workflow-use-case-preset-in-use",
    "workflow-use-case-preset-disabled",
    "workflow-use-case-preset-mismatch",
    "workflow-use-case-preset-invalid",
    "workflow-use-case-scope-not-found",
]


class WorkflowUseCasePresetServiceError(ValueError):
    def __init__(self, code: ServiceRefusal) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WorkflowUseCasePresetRecord:
    id: str
    name: str
    use_case: WorkflowUseCase
    settings_json: dict[str, JsonValue]
    enabled: bool
    builtin: bool
    is_default: bool


@contextmanager
def _write(session: Session) -> Iterator[None]:
    """Serialize selections with disable/delete and default transfers.

    A fresh session is required so this operation cannot commit or roll back
    another caller's transaction. A failed operation restores every changed row.
    """
    if session.in_transaction():
        raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-conflict")
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        yield
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-conflict") from exc
    except OperationalError as exc:
        session.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-conflict") from exc
        raise
    except BaseException:
        session.rollback()
        raise


def _record(row: WorkflowUseCasePreset) -> WorkflowUseCasePresetRecord:
    _explicit_choice(row.id)
    try:
        value = WorkflowUseCasePresetCreate.model_validate(
            dict(
                name=row.name,
                use_case=row.use_case,
                settings_json=row.settings_json,
                enabled=row.enabled,
                is_default=row.is_default,
            )
        )
    except ValidationError as exc:
        raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-invalid") from exc
    return WorkflowUseCasePresetRecord(
        row.id,
        value.name,
        value.use_case,
        deepcopy(value.settings_json),
        value.enabled,
        row.builtin,
        value.is_default,
    )


def _preset(session: Session, preset_id: str) -> WorkflowUseCasePreset:
    row = session.get(WorkflowUseCasePreset, preset_id)
    if row is None:
        raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-not-found")
    return row


def _selected(session: Session, preset_id: str) -> bool:
    return bool(
        session.scalar(
            select(
                or_(
                    exists(
                        select(ChatWorkflowUseCaseSelection.chat_id).where(
                            ChatWorkflowUseCaseSelection.preset_id == preset_id
                        )
                    ),
                    exists(
                        select(ProjectWorkflowUseCaseSelection.project_id).where(
                            ProjectWorkflowUseCaseSelection.preset_id == preset_id
                        )
                    ),
                )
            )
        )
    )


def _clear_default(session: Session, use_case: WorkflowUseCase) -> None:
    session.execute(
        update(WorkflowUseCasePreset)
        .where(
            WorkflowUseCasePreset.use_case == use_case,
            WorkflowUseCasePreset.is_default.is_(True),
        )
        .values(is_default=False)
    )


def list_presets(
    session: Session,
    *,
    use_case: WorkflowUseCase | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[WorkflowUseCasePresetRecord]:
    if not 1 <= limit <= 200 or not 0 <= offset <= 2**63 - 1:
        raise ValueError("Invalid use-case recipe page.")
    query = select(WorkflowUseCasePreset)
    if use_case is not None:
        query = query.where(WorkflowUseCasePreset.use_case == use_case)
    with session.no_autoflush:
        return [
            _record(row)
            for row in session.scalars(
                query.order_by(
                    WorkflowUseCasePreset.use_case,
                    WorkflowUseCasePreset.name,
                    WorkflowUseCasePreset.id,
                )
                .limit(limit)
                .offset(offset)
            )
        ]


def read_preset(session: Session, preset_id: str) -> WorkflowUseCasePresetRecord:
    with session.no_autoflush:
        return _record(_preset(session, preset_id))


def read_workspace_default(session: Session, use_case: WorkflowUseCase) -> str | None:
    with session.no_autoflush:
        preset_id = session.scalar(
            select(WorkflowUseCasePreset.id).where(
                WorkflowUseCasePreset.use_case == use_case,
                WorkflowUseCasePreset.is_default.is_(True),
            )
        )
        if preset_id is not None:
            _explicit_choice(preset_id)
        return preset_id


def create_preset(
    session: Session,
    payload: WorkflowUseCasePresetCreate,
) -> WorkflowUseCasePresetRecord:
    with _write(session):
        if payload.is_default:
            _clear_default(session, payload.use_case)
        row = WorkflowUseCasePreset(
            name=payload.name,
            use_case=payload.use_case,
            settings_json=deepcopy(payload.settings_json),
            enabled=payload.enabled,
            is_default=payload.is_default,
        )
        session.add(row)
        session.flush()
        result = _record(row)
    return result


def replace_preset(
    session: Session,
    preset_id: str,
    payload: WorkflowUseCasePresetCreate,
) -> WorkflowUseCasePresetRecord:
    with _write(session):
        row = _preset(session, preset_id)
        if row.builtin:
            raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-builtin")
        if row.use_case != payload.use_case:
            raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-mismatch")
        if not payload.enabled and _selected(session, preset_id):
            raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-in-use")
        if payload.is_default:
            _clear_default(session, payload.use_case)
        row.name = payload.name
        row.settings_json = deepcopy(payload.settings_json)
        row.enabled = payload.enabled
        row.is_default = payload.is_default
        session.flush()
        result = _record(row)
    return result


def set_workspace_default(
    session: Session, use_case: WorkflowUseCase, preset_id: str | None
) -> WorkflowUseCasePresetRecord | None:
    """Select or clear a default without changing a built-in recipe's contents."""
    with _write(session):
        row = _preset(session, preset_id) if preset_id is not None else None
        if row is not None:
            if row.use_case != use_case:
                raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-mismatch")
            if not row.enabled:
                raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-disabled")
        _clear_default(session, use_case)
        result = None
        if row is not None:
            row.is_default = True
            session.flush()
            result = _record(row)
    return result


def delete_preset(session: Session, preset_id: str) -> None:
    with _write(session):
        row = _preset(session, preset_id)
        if row.builtin:
            raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-builtin")
        if row.is_default or _selected(session, preset_id):
            raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-in-use")
        session.delete(row)


def _scope(session: Session, scope: SelectionScope, scope_id: str) -> None:
    model = Chat if scope == "chat" else Project
    visibility = visible_chat(Chat.id) if scope == "chat" else visible_project(Project.id)
    if session.scalar(select(model.id).where(model.id == scope_id, visibility)) is None:
        raise WorkflowUseCasePresetServiceError("workflow-use-case-scope-not-found")


def _explicit_choice(preset_id: str) -> WorkflowUseCaseChoice:
    try:
        choice = WorkflowUseCaseChoice.model_validate({"mode": "preset", "preset_id": preset_id})
    except ValidationError:
        raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-invalid") from None
    if choice.root.mode != "preset" or choice.root.preset_id != preset_id:
        raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-invalid")
    return choice


def read_choice(
    session: Session,
    scope: SelectionScope,
    scope_id: str,
    use_case: WorkflowUseCase,
) -> WorkflowUseCaseChoice:
    with session.no_autoflush:
        _scope(session, scope, scope_id)
        row = _choice_row(session, scope, scope_id, use_case)
        if row is None:
            return WorkflowUseCaseChoice.model_validate({"mode": "inherit"})
        if row.preset_id is None:
            return WorkflowUseCaseChoice.model_validate({"mode": "automatic"})
        return _explicit_choice(row.preset_id)


def _choice_row(
    session: Session,
    scope: SelectionScope,
    scope_id: str,
    use_case: WorkflowUseCase,
) -> ChatWorkflowUseCaseSelection | ProjectWorkflowUseCaseSelection | None:
    if scope == "chat":
        return session.get(ChatWorkflowUseCaseSelection, (scope_id, use_case.value))
    return session.get(ProjectWorkflowUseCaseSelection, (scope_id, use_case.value))


def write_choice(
    session: Session,
    scope: SelectionScope,
    scope_id: str,
    use_case: WorkflowUseCase,
    choice: WorkflowUseCaseChoice,
) -> WorkflowUseCaseChoice:
    model = ChatWorkflowUseCaseSelection if scope == "chat" else ProjectWorkflowUseCaseSelection
    key = (
        ChatWorkflowUseCaseSelection.chat_id
        if scope == "chat"
        else ProjectWorkflowUseCaseSelection.project_id
    )
    with _write(session):
        _scope(session, scope, scope_id)
        selected = choice.root
        if selected.mode == "inherit":
            session.execute(delete(model).where(key == scope_id, model.use_case == use_case))
        else:
            preset_id = None
            if selected.mode == "preset":
                preset_id = selected.preset_id
                preset = _preset(session, preset_id)
                if preset.use_case != use_case:
                    raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-mismatch")
                if not preset.enabled:
                    raise WorkflowUseCasePresetServiceError("workflow-use-case-preset-disabled")
            row = _choice_row(session, scope, scope_id, use_case)
            if row is None:
                row = (
                    ChatWorkflowUseCaseSelection(chat_id=scope_id, use_case=use_case.value)
                    if scope == "chat"
                    else ProjectWorkflowUseCaseSelection(
                        project_id=scope_id, use_case=use_case.value
                    )
                )
                session.add(row)
            row.preset_id = preset_id
        result = choice.model_copy(deep=True)
    return result
