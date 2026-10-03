"""Keep bounded recovery command results without copying resource content."""

from __future__ import annotations

import hashlib
import hmac

from pydantic import ValidationError
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from .models import RecoveryBatchRecord, RecoveryItem, RecoveryOperation
from .recovery_bulk_v1 import RecoveryBatchPreviewV1
from .recovery_previews import RecoveryPreviewConflict
from .recovery_v1 import (
    RecoveryAction,
    RecoveryCommandV1,
    RecoveryItemV1,
    RecoveryKind,
    RecoveryResultV1,
)


def forget_recovery_labels(session: Session, item: RecoveryItem) -> None:
    """Keep retry identities while erasing names after permanent deletion."""
    item.display_label = "Deleted item"
    item.original_project_label = None
    if item.kind == RecoveryKind.PROJECT.value:
        for child in session.scalars(
            select(RecoveryItem).where(RecoveryItem.original_project_id == item.subject_id)
        ):
            child.original_project_label = None
        for operation in session.scalars(
            select(RecoveryOperation).where(
                RecoveryOperation.action == RecoveryAction.TRASH.value,
                func.json_extract(RecoveryOperation.response_json, "$.original_location.project_id")
                == item.subject_id,
            )
        ):
            value = RecoveryItemV1.model_validate(operation.response_json)
            operation.response_json = value.model_copy(
                update={
                    "original_location": value.original_location.model_copy(
                        update={"project_label": None}
                    )
                }
            ).model_dump(mode="json")
    for operation in session.scalars(
        select(RecoveryOperation).where(
            RecoveryOperation.kind == item.kind,
            RecoveryOperation.subject_id == item.subject_id,
            RecoveryOperation.action == RecoveryAction.TRASH.value,
        )
    ):
        value = RecoveryItemV1.model_validate(operation.response_json)
        operation.response_json = value.model_copy(
            update={
                "display_label": "Deleted item",
                "original_location": value.original_location.model_copy(
                    update={"project_label": None}
                ),
            }
        ).model_dump(mode="json")
    members = func.json_each(RecoveryBatchRecord.preview_json, "$.items").table_valued("value")
    for record in session.scalars(
        select(RecoveryBatchRecord).where(
            exists(
                select(1)
                .select_from(members)
                .where(
                    func.json_extract(members.c.value, "$.impact.kind") == item.kind,
                    func.json_extract(members.c.value, "$.impact.subject_id") == item.subject_id,
                )
            )
        )
    ):
        preview = RecoveryBatchPreviewV1.model_validate(record.preview_json)
        draft = preview.model_copy(
            update={
                "impact_sha256": "0" * 64,
                "items": tuple(
                    member.model_copy(update={"display_label": "Deleted item"})
                    if member.impact.kind.value == item.kind
                    and member.impact.subject_id == item.subject_id
                    else member
                    for member in preview.items
                ),
            }
        )
        digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
        record.preview_json = draft.model_copy(update={"impact_sha256": digest}).model_dump(
            mode="json"
        )


def _request_digest(command: RecoveryCommandV1) -> str:
    return hashlib.sha256(command.model_dump_json().encode("utf-8")).hexdigest()


def _replay(
    operation: RecoveryOperation | None, action: RecoveryAction, command: RecoveryCommandV1
) -> RecoveryItemV1 | RecoveryResultV1 | None:
    if operation is None:
        return None
    if operation.action != action.value or not hmac.compare_digest(
        operation.request_sha256, _request_digest(command)
    ):
        raise RecoveryPreviewConflict("recovery-operation-conflict")
    try:
        result = (
            RecoveryItemV1.model_validate(operation.response_json)
            if action == RecoveryAction.TRASH
            else RecoveryResultV1.model_validate(operation.response_json)
        )
    except ValidationError:
        raise RecoveryPreviewConflict("recovery-operation-invalid") from None
    if (
        result.kind.value != operation.kind
        or result.subject_id != operation.subject_id
        or result.deletion_id != operation.deletion_id
        or (isinstance(result, RecoveryResultV1) and result.action != action)
    ):
        raise RecoveryPreviewConflict("recovery-operation-invalid")
    return result


def _remember(
    session: Session,
    action: RecoveryAction,
    command: RecoveryCommandV1,
    result: RecoveryItemV1 | RecoveryResultV1,
) -> None:
    if len(result.model_dump_json().encode("utf-8")) > 16 * 1024:
        raise RecoveryPreviewConflict("recovery-operation-invalid")
    session.add(
        RecoveryOperation(
            kind=result.kind.value,
            subject_id=result.subject_id,
            operation_key=command.operation_key,
            action=action.value,
            deletion_id=result.deletion_id,
            request_sha256=_request_digest(command),
            response_json=result.model_dump(mode="json"),
        )
    )
    session.flush()
