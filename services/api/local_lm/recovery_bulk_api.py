"""Hold selected chat guards through one atomic bulk recovery transaction."""

from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from .api_errors import ApiError, api_error
from .chat_recovery_graph import ChatRecoveryGraphError
from .db import get_session
from .recovery_bulk import apply_batch, batch_chat_ids, batch_record, materialize_batch
from .recovery_bulk_v1 import (
    RecoveryBatchApplyV1,
    RecoveryBatchPreviewV1,
    RecoveryBatchResultV1,
    RecoveryBatchSelectionV1,
)
from .recovery_previews import RecoveryPreviewConflict

if TYPE_CHECKING:
    from .main import Services

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _refusal(error: RecoveryPreviewConflict) -> ApiError:
    status = (
        404
        if error.code in {"recovery-batch-not-found", "recovery-item-not-found"}
        else 422
        if error.code.startswith("recovery-batch-acknowledgement-")
        else 503
        if error.code in {"recovery-database-busy", "recovery-database-unsupported"}
        else 409
    )
    return api_error(
        status,
        error.code,
        "The selected items could not be changed together. Check their recovery details again.",
    )


@router.post("/recovery-items/batches", response_model=RecoveryBatchPreviewV1)
async def preview_batch(
    payload: RecoveryBatchSelectionV1, request: Request, response: Response, session: SessionDep
) -> RecoveryBatchPreviewV1:
    services = cast("Services", request.app.state.services)
    ids = batch_chat_ids(session, payload.deletion_ids)
    session.rollback()
    async with AsyncExitStack() as stack:
        for identity in ids:
            await stack.enter_async_context(services.orchestrator.chat_guard(identity))
        try:
            result = materialize_batch(session, payload, datetime.now(UTC))
            session.commit()
        except RecoveryPreviewConflict as error:
            session.rollback()
            raise _refusal(error) from None
        except ChatRecoveryGraphError:
            session.rollback()
            raise api_error(
                409,
                "chat-recovery-graph-invalid",
                "The selected deletion details could not be verified.",
            ) from None
        except Exception:
            session.rollback()
            raise
    response.headers["Cache-Control"] = "no-store"
    return result


@router.post("/recovery-items/batches/{batch_id}/apply", response_model=RecoveryBatchResultV1)
async def apply_recovery_batch(
    batch_id: str, payload: RecoveryBatchApplyV1, request: Request, session: SessionDep
) -> RecoveryBatchResultV1:
    services = cast("Services", request.app.state.services)
    try:
        _, preview = batch_record(session, batch_id)
    except RecoveryPreviewConflict as error:
        session.rollback()
        raise _refusal(error) from None
    ids = sorted(
        item.impact.subject_id for item in preview.items if item.impact.kind.value == "chat"
    )
    session.rollback()
    async with AsyncExitStack() as stack:
        for identity in ids:
            await stack.enter_async_context(services.orchestrator.chat_guard(identity))
        try:
            result = apply_batch(session, batch_id, payload, datetime.now(UTC))
            session.commit()
        except RecoveryPreviewConflict as error:
            session.rollback()
            raise _refusal(error) from None
        except ChatRecoveryGraphError:
            session.rollback()
            raise api_error(
                409,
                "chat-recovery-graph-invalid",
                "The selected deletion details could not be verified.",
            ) from None
        except Exception:
            session.rollback()
            raise
    for item in result.results:
        event = {
            "chat": "chat.updated",
            "project": "project.updated",
            "workflow_family": "workflow.updated",
        }.get(item.kind.value)
        if event:
            await services.events.publish(event, item.subject_id, {})
    await services.events.publish("recovery.updated", result.batch_id, {})
    return result
