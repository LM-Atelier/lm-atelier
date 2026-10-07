"""A committed workflow deletion fences stale writers and new accepted consumers."""

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from test_chat_recovery import _command, _impact
from test_workflow_recovery_api import _seed

from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    Job,
    RecoveryItem,
    WorkflowFamily,
    WorkflowRevision,
)


@pytest.mark.parametrize("operation", ["metadata", "selection", "job"])
async def test_committed_workflow_trash_refuses_a_previously_read_writer(
    client: AsyncClient,
    operation: str,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        session.add(Chat(id="chat-workflow-admission", title="Garden notes"))
        session.commit()
    with SessionLocal() as stale:
        family = stale.get(WorkflowFamily, family_id)
        revision = stale.get(WorkflowRevision, revision_id)
        assert family is not None and family.enabled and revision is not None
        preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
        trashed = await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-before-stale-writer"),
        )
        assert trashed.status_code == 200
        if operation == "metadata":
            family.description = "Updated garden layout"
        elif operation == "selection":
            stale.add(
                ChatWorkflowSelection(
                    chat_id="chat-workflow-admission",
                    selector_capability="image",
                    mode="family",
                    workflow_family_id=family_id,
                )
            )
        else:
            stale.add(
                Job(
                    id="job-after-workflow-trash",
                    status="queued",
                    payload_json={"accepted": {"workflow_revision_id": revision_id}},
                )
            )
        with pytest.raises(IntegrityError, match="workflow-recovery-write-refused"):
            stale.commit()
        stale.rollback()
    with SessionLocal() as session:
        family = session.get(WorkflowFamily, family_id)
        assert family is not None and not family.enabled and family.description == ""
        assert session.get(Job, "job-after-workflow-trash") is None
        assert (
            session.scalar(
                select(ChatWorkflowSelection).where(
                    ChatWorkflowSelection.chat_id == "chat-workflow-admission"
                )
            )
            is None
        )
        assert (
            session.scalar(
                select(RecoveryItem).where(
                    RecoveryItem.kind == "workflow_family", RecoveryItem.subject_id == family_id
                )
            )
            is not None
        )
