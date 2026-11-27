"""A workflow write answers with each revision's trust as a read of that workflow reports it.

A revision's trust is its review read as workflow selection reads it. A review
whose revision has changed since leaves the stored flag set while selection
refuses the revision, and the workflow detail and list read trust the way
selection does. Renaming such a workflow, cloning it, or creating one used to
answer with the stored flag, so the write could call trusted what the next read
of the same workflow did not.
"""

from __future__ import annotations

from typing import Any

from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowRevision,
    WorkflowRevisionReview,
)
from local_lm.workflow_revision_reviews import _review_subject, revision_identity

BINDINGS = {
    "KSampler": {"kind": "core", "module": "nodes", "contract": "0" * 64},
    "SaveImage": {"kind": "core", "module": "nodes", "contract": "0" * 64},
}


def _graph() -> dict[str, Any]:
    return {
        "1": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }


def _reviewed_workflow_whose_revision_changed_since() -> str:
    """A workflow approved on review, whose revision then changed; its id."""

    with SessionLocal() as session:
        family = WorkflowFamily(name="Neutral reviewed family", use_case="neutral")
        definition = WorkflowDefinition(
            family=family,
            variant_key="main",
            name="Neutral reviewed workflow",
            operation=Operation.TEXT_TO_IMAGE.value,
        )
        session.add_all([family, definition])
        session.flush()
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            api_graph_json=_graph(),
            input_schema_json={"type": "object", "properties": {}},
            trusted=True,
        )
        session.add(revision)
        session.flush()
        identity = revision_identity(definition, revision)
        session.add(
            WorkflowRevisionReview(
                workflow_revision_id=revision.id,
                revision_sha256=identity,
                subject_sha256=_review_subject(identity, BINDINGS),
                node_bindings_json=BINDINGS,
                state="approved",
            )
        )
        definition.current_revision_id = revision.id
        # What the review approved is no longer what the revision holds.
        revision.engine_version = "a later engine version"
        session.commit()
        return definition.id


def _trust(workflow: dict[str, Any]) -> dict[str, bool]:
    return {revision["id"]: revision["trusted"] for revision in workflow["revisions"]}


async def _read_trust(client: AsyncClient, workflow_id: str) -> dict[str, bool]:
    read = await client.get(f"/api/workflows/{workflow_id}")
    assert read.status_code == 200, read.text
    return _trust(read.json())


async def test_renaming_a_workflow_answers_with_the_trust_a_read_reports(
    client: AsyncClient,
) -> None:
    workflow_id = _reviewed_workflow_whose_revision_changed_since()
    before = await _read_trust(client, workflow_id)
    assert list(before.values()) == [False], "the changed revision still reads as trusted"

    renamed = await client.patch(f"/api/workflows/{workflow_id}", json={"name": "Renamed"})

    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Renamed"
    assert _trust(renamed.json()) == before == await _read_trust(client, workflow_id)


async def test_cloning_a_workflow_answers_with_the_trust_a_read_of_the_clone_reports(
    client: AsyncClient,
) -> None:
    workflow_id = _reviewed_workflow_whose_revision_changed_since()

    cloned = await client.post(f"/api/workflows/{workflow_id}/clone", json={"name": "Copy"})

    assert cloned.status_code == 201, cloned.text
    assert _trust(cloned.json()) == await _read_trust(client, cloned.json()["id"])


async def test_creating_a_workflow_answers_with_the_trust_a_read_reports(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral created workflow",
            "operation": Operation.TEXT_TO_IMAGE.value,
            "engine": "mock",
            "api_graph": _graph(),
            "input_schema": {"type": "object", "properties": {}},
        },
    )

    assert created.status_code == 201, created.text
    assert _trust(created.json()) == await _read_trust(client, created.json()["id"])
    assert list(_trust(created.json()).values()) == [False]
