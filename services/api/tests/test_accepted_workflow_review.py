from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from test_workflow_revision_review import _approved
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision, WorkflowRevisionReview


@pytest.mark.parametrize("change", ["review_revoked", "new_review_for_changed_graph"])
async def test_accepted_workflow_cannot_reuse_superseded_review_authority(
    client: AsyncClient, change: str
) -> None:
    from local_lm.accepted_turn_context import capture_workflow, resolve_accepted_workflow

    workflow_id, revision_id, _ = await _approved(client)
    with SessionLocal() as session:
        snapshot = capture_workflow(session, revision_id)
        assert snapshot is not None
        assert resolve_accepted_workflow(session, snapshot) is not None
        revision = session.get(WorkflowRevision, revision_id)
        review = session.get(WorkflowRevisionReview, revision_id)
        assert revision is not None and review is not None
        if change == "review_revoked":
            review.state = "revoked"
        else:
            revision.api_graph_json = {
                "1": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": 640, "height": 512, "batch_size": 1},
                }
            }
        session.commit()
        assert revision.trusted is True
    if change == "new_review_for_changed_graph":
        url = f"/api/workflows/{workflow_id}/revisions/{revision_id}/review"
        preview = await client.get(url)
        assert preview.status_code == 200, preview.text
        approved = await client.post(
            url, json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]}
        )
        assert approved.status_code == 200, approved.text
    with (
        SessionLocal() as session,
        pytest.raises(RuntimeError, match="unavailable or no longer trusted"),
    ):
        resolve_accepted_workflow(session, snapshot)
