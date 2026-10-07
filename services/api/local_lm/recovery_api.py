"""Preview and perform recovery with committed, content-free updates."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .api_errors import ApiError, api_error
from .chat_recovery import (
    preview_chat_recovery,
    preview_chat_trash,
    purge_chat,
    restore_chat,
    trash_chat,
)
from .chat_recovery_graph import ChatRecoveryGraphError
from .db import get_session
from .media_recovery import (
    preview_media_recovery,
    preview_media_trash,
    purge_media,
    restore_media,
    trash_media,
)
from .models import Chat, RecoveryItem, RecoveryOperation
from .project_recovery import (
    preview_project_recovery,
    preview_project_trash,
    purge_project,
    restore_project,
    trash_project,
)
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .recovery_bulk_api import router as bulk_router
from .recovery_listing import recovery_page
from .recovery_previews import RecoveryPreviewConflict
from .recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryCommandV1,
    RecoveryImpactV1,
    RecoveryItemV1,
    RecoveryKind,
    RecoveryPageV1,
    RecoveryResultV1,
    RecoveryState,
    RestoreRecoveryV1,
    TrashChatV1,
)
from .workflow_recovery import (
    preview_workflow_recovery,
    preview_workflow_trash,
    purge_workflow,
    restore_workflow,
    trash_workflow,
)

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()
router.include_router(bulk_router)

REFUSALS: dict[str, tuple[int, str]] = {
    "workflow-already-trashed": (409, "This workflow is already in Recently Deleted."),
    "workflow-recovery-graph-invalid": (
        409,
        "This workflow's deletion details could not be verified.",
    ),
    "workflow-recovery-in-use": (
        409,
        "This workflow is still selected or required by work. Check its deletion details again.",
    ),
    "project-already-trashed": (409, "This project is already in Recently Deleted."),
    "project-recovery-graph-invalid": (
        409,
        "This project's deletion details could not be verified.",
    ),
    "project-recovery-accepted-work-invalid": (
        409,
        "This project's accepted work could not be preserved. "
        "Let it finish before deleting the project.",
    ),
    "media-already-trashed": (409, "This Media Library item is already in Recently Deleted."),
    "media-recovery-graph-invalid": (
        409,
        "This Media Library item's deletion details could not be verified.",
    ),
    "media-recovery-membership-invalid": (
        409,
        "This Media Library item's recovery state could not be verified.",
    ),
    "recovery-cursor-invalid": (400, "This recovery page has expired. Refresh the list."),
    "recovery-page-size-invalid": (400, "Choose between one and twenty deleted items."),
    "recovery-item-not-found": (404, "This deleted item no longer exists."),
    "recovery-subject-not-found": (404, "This item no longer exists."),
    "recovery-item-not-recoverable": (409, "This deleted item cannot be changed now."),
    "chat-already-trashed": (409, "This conversation is already in Recently Deleted."),
    "chat-recovery-active-work": (
        409,
        "Wait for this conversation's work to finish before deleting it.",
    ),
    "recovery-impact-stale": (409, "The item changed. Check its deletion details again."),
    "recovery-operation-conflict": (
        409,
        "This request key was already used for a different action.",
    ),
    "recovery-operation-invalid": (409, "The saved recovery result is invalid."),
    "recovery-original-project-missing": (
        409,
        "The original project is gone. Choose to restore this chat unfiled.",
    ),
    "recovery-window-expired": (409, "This item's recovery window has ended."),
    "recovery-transaction-not-clean": (
        409,
        "Another change must finish before recovery can continue.",
    ),
    "recovery-database-unsupported": (503, "Recovery is not available for this database."),
    "recovery-database-busy": (503, "The database is busy. Try again in a moment."),
    "recovery-snapshot-invalid": (409, "The deletion details could not be verified."),
}


def _error(error: RecoveryPreviewConflict) -> ApiError:
    status, message = REFUSALS.get(error.code, (409, "Recovery could not be completed."))
    return api_error(status, error.code, message)


def _commit[Result: (RecoveryImpactV1, RecoveryItemV1, RecoveryResultV1, RecoveryPageV1)](
    session: Session, operation: Callable[[], Result]
) -> Result:
    """Do not await between the reserved writer and its commit or rollback."""

    try:
        result = operation()
        session.commit()
        return result
    except RecoveryPreviewConflict as error:
        session.rollback()
        raise _error(error) from None
    except ChatRecoveryGraphError:
        session.rollback()
        raise api_error(
            409,
            "chat-recovery-graph-invalid",
            "This conversation's deletion details could not be verified.",
        ) from None


def _standard_subject(session: Session, chat_id: str) -> None:
    if session.scalar(select(Chat.scope).where(Chat.id == chat_id)) != STANDARD_CHAT_SCOPE:
        raise api_error(404, "chat-not-found", "chat not found")
    session.rollback()


def _recovery_subject(
    session: Session, deletion_id: str, operation_key: str | None = None
) -> tuple[RecoveryKind, str]:
    supported = (
        RecoveryKind.CHAT.value,
        RecoveryKind.PROJECT.value,
        RecoveryKind.MEDIA_LIBRARY_ENTRY.value,
        RecoveryKind.WORKFLOW_FAMILY.value,
    )
    identity = session.execute(
        select(RecoveryItem.kind, RecoveryItem.subject_id).where(
            RecoveryItem.deletion_id == deletion_id, RecoveryItem.kind.in_(supported)
        )
    ).first()
    if identity is None and operation_key is not None:
        identity = session.execute(
            select(RecoveryOperation.kind, RecoveryOperation.subject_id).where(
                RecoveryOperation.deletion_id == deletion_id,
                RecoveryOperation.kind.in_(supported),
                RecoveryOperation.operation_key == operation_key,
            )
        ).first()
    if identity is not None and identity.kind == RecoveryKind.CHAT.value:
        scope = session.scalar(select(Chat.scope).where(Chat.id == identity.subject_id))
        if scope is not None and scope != STANDARD_CHAT_SCOPE:
            identity = None
    session.rollback()
    if identity is None:
        raise api_error(404, "recovery-item-not-found", "This deleted item no longer exists.")
    return RecoveryKind(identity.kind), identity.subject_id


def _services(request: Request) -> Services:
    return cast("Services", request.app.state.services)


@router.get("/recovery-items", response_model=RecoveryPageV1)
def deleted_items(
    response: Response,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
    cursor: Annotated[str | None, Query(max_length=2048)] = None,
    state: RecoveryState | None = None,
    kind: RecoveryKind | None = None,
    deleted_since: datetime | None = None,
) -> RecoveryPageV1:
    response.headers["Cache-Control"] = "no-store"
    return _commit(
        session,
        lambda: recovery_page(
            session,
            datetime.now(UTC),
            limit=limit,
            cursor=cursor,
            state=state,
            kind=kind,
            deleted_since=deleted_since,
        ),
    )


@router.get("/chats/{chat_id}/deletion-impact", response_model=RecoveryImpactV1)
async def chat_deletion_impact(
    chat_id: str,
    request: Request,
    response: Response,
    session: SessionDep,
    delete_generated_media: bool = False,
) -> RecoveryImpactV1:
    _standard_subject(session, chat_id)
    response.headers["Cache-Control"] = "no-store"
    async with _services(request).orchestrator.chat_guard(chat_id):
        return _commit(
            session,
            lambda: preview_chat_trash(
                session, chat_id, datetime.now(UTC), delete_generated_media=delete_generated_media
            ),
        )


@router.post("/chats/{chat_id}/trash", response_model=RecoveryItemV1)
async def move_chat_to_trash(
    chat_id: str,
    payload: TrashChatV1,
    request: Request,
    session: SessionDep,
) -> RecoveryItemV1:
    _standard_subject(session, chat_id)
    async with _services(request).orchestrator.chat_guard(chat_id):
        result = _commit(session, lambda: trash_chat(session, chat_id, payload, datetime.now(UTC)))
    await _services(request).events.publish("chat.updated", chat_id, {})
    await _services(request).events.publish("recovery.updated", result.deletion_id, {})
    return result


@router.get("/artifact-library/{entry_id}/deletion-impact", response_model=RecoveryImpactV1)
def media_deletion_impact(
    entry_id: str, response: Response, session: SessionDep
) -> RecoveryImpactV1:
    response.headers["Cache-Control"] = "no-store"
    return _commit(session, lambda: preview_media_trash(session, entry_id, datetime.now(UTC)))


@router.post("/artifact-library/{entry_id}/trash", response_model=RecoveryItemV1)
async def move_media_to_trash(
    entry_id: str, payload: RecoveryCommandV1, request: Request, session: SessionDep
) -> RecoveryItemV1:
    result = _commit(session, lambda: trash_media(session, entry_id, payload, datetime.now(UTC)))
    await _services(request).events.publish("recovery.updated", result.deletion_id, {})
    return result


@router.get("/projects/{project_id}/deletion-impact", response_model=RecoveryImpactV1)
def project_deletion_impact(
    project_id: str, response: Response, session: SessionDep
) -> RecoveryImpactV1:
    response.headers["Cache-Control"] = "no-store"
    return _commit(session, lambda: preview_project_trash(session, project_id, datetime.now(UTC)))


@router.post("/projects/{project_id}/trash", response_model=RecoveryItemV1)
async def move_project_to_trash(
    project_id: str, payload: RecoveryCommandV1, request: Request, session: SessionDep
) -> RecoveryItemV1:
    result = _commit(
        session,
        lambda: trash_project(
            session,
            project_id,
            payload,
            datetime.now(UTC),
            before_trash=_services(request).orchestrator.freeze_project_pending_context,
        ),
    )
    await _services(request).events.publish("project.updated", project_id, {})
    await _services(request).events.publish("recovery.updated", result.deletion_id, {})
    return result


@router.get("/workflow-families/{family_id}/deletion-impact", response_model=RecoveryImpactV1)
def workflow_deletion_impact(
    family_id: str, response: Response, session: SessionDep
) -> RecoveryImpactV1:
    response.headers["Cache-Control"] = "no-store"
    return _commit(session, lambda: preview_workflow_trash(session, family_id, datetime.now(UTC)))


@router.post("/workflow-families/{family_id}/trash", response_model=RecoveryItemV1)
async def move_workflow_to_trash(
    family_id: str, payload: RecoveryCommandV1, request: Request, session: SessionDep
) -> RecoveryItemV1:
    result = _commit(
        session, lambda: trash_workflow(session, family_id, payload, datetime.now(UTC))
    )
    await _services(request).events.publish("workflow.updated", family_id, {})
    await _services(request).events.publish("recovery.updated", result.deletion_id, {})
    return result


@router.get("/recovery-items/{deletion_id}/impact", response_model=RecoveryImpactV1)
async def recovery_item_impact(
    deletion_id: str, request: Request, response: Response, session: SessionDep
) -> RecoveryImpactV1:
    kind, subject_id = _recovery_subject(session, deletion_id)
    response.headers["Cache-Control"] = "no-store"
    if kind == RecoveryKind.WORKFLOW_FAMILY:
        return _commit(
            session, lambda: preview_workflow_recovery(session, deletion_id, datetime.now(UTC))
        )
    if kind == RecoveryKind.MEDIA_LIBRARY_ENTRY:
        return _commit(
            session, lambda: preview_media_recovery(session, deletion_id, datetime.now(UTC))
        )
    if kind == RecoveryKind.PROJECT:
        return _commit(
            session, lambda: preview_project_recovery(session, deletion_id, datetime.now(UTC))
        )
    async with _services(request).orchestrator.chat_guard(subject_id):
        return _commit(
            session, lambda: preview_chat_recovery(session, deletion_id, datetime.now(UTC))
        )


async def _transition_and_publish(
    deletion_id: str,
    payload: RestoreRecoveryV1 | PurgeRecoveryV1,
    request: Request,
    session: Session,
) -> RecoveryResultV1:
    kind, subject_id = _recovery_subject(session, deletion_id, payload.operation_key)
    if kind == RecoveryKind.CHAT:
        async with _services(request).orchestrator.chat_guard(subject_id):
            result = _commit(
                session,
                lambda: (
                    restore_chat(session, deletion_id, payload, datetime.now(UTC))
                    if isinstance(payload, RestoreRecoveryV1)
                    else purge_chat(session, deletion_id, payload, datetime.now(UTC))
                ),
            )
    elif kind == RecoveryKind.PROJECT:
        result = _commit(
            session,
            lambda: (
                restore_project(session, deletion_id, payload, datetime.now(UTC))
                if isinstance(payload, RestoreRecoveryV1)
                else purge_project(session, deletion_id, payload, datetime.now(UTC))
            ),
        )
    elif kind == RecoveryKind.WORKFLOW_FAMILY:
        result = _commit(
            session,
            lambda: (
                restore_workflow(session, deletion_id, payload, datetime.now(UTC))
                if isinstance(payload, RestoreRecoveryV1)
                else purge_workflow(session, deletion_id, payload, datetime.now(UTC))
            ),
        )
    else:
        result = _commit(
            session,
            lambda: (
                restore_media(session, deletion_id, payload, datetime.now(UTC))
                if isinstance(payload, RestoreRecoveryV1)
                else purge_media(session, deletion_id, payload, datetime.now(UTC))
            ),
        )
    if kind == RecoveryKind.CHAT:
        await _services(request).events.publish("chat.updated", subject_id, {})
    elif kind == RecoveryKind.PROJECT:
        await _services(request).events.publish("project.updated", subject_id, {})
    elif kind == RecoveryKind.WORKFLOW_FAMILY:
        await _services(request).events.publish("workflow.updated", subject_id, {})
    await _services(request).events.publish("recovery.updated", result.deletion_id, {})
    return result


@router.post("/recovery-items/{deletion_id}/restore", response_model=RecoveryResultV1)
async def restore_deleted_item(
    deletion_id: str, payload: RestoreRecoveryV1, request: Request, session: SessionDep
) -> RecoveryResultV1:
    return await _transition_and_publish(deletion_id, payload, request, session)


@router.post("/recovery-items/{deletion_id}/purge", response_model=RecoveryResultV1)
async def permanently_delete_item(
    deletion_id: str, payload: PurgeRecoveryV1, request: Request, session: SessionDep
) -> RecoveryResultV1:
    return await _transition_and_publish(deletion_id, payload, request, session)
