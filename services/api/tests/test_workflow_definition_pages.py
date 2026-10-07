from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_workflow_summary_reads import client, session, workflow

from local_lm.models import WorkflowDefinition, WorkflowRevision
from local_lm.workflow_package_drafts import WORKFLOW_PACKAGE_DRAFT_MARKER

__all__ = ["client", "session"]


def seed_definitions(session: Session) -> None:
    for identifier in ("e", "d", "c", "b", "a", "draft-before", "draft-between"):
        workflow(session, identifier, revisions=3)
        definition = session.get(WorkflowDefinition, identifier)
        assert definition is not None
        definition.name = "A hidden draft" if identifier == "draft-before" else "Same name"
        if identifier.startswith("draft-"):
            revision = session.get(WorkflowRevision, identifier + "-r3")
            assert revision is not None
            revision.dependencies_json = {WORKFLOW_PACKAGE_DRAFT_MARKER: {}}
    session.commit()
    session.expunge_all()


async def test_definition_pages_filter_drafts_before_loading_selected_revision_graphs(
    session: Session, client: AsyncClient
) -> None:
    seed_definitions(session)
    loaded: list[object] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    event.listen(session, "loaded_as_persistent", track)
    try:
        response = await client.get("/api/workflows?limit=2&offset=1")
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["b", "c"]
    assert {row.id for row in loaded if isinstance(row, WorkflowDefinition)} == {"b", "c"}
    assert {row.id for row in loaded if isinstance(row, WorkflowRevision)} == {
        "b-r1",
        "b-r2",
        "b-r3",
        "c-r1",
        "c-r2",
        "c-r3",
    }
    assert all(len(row["revisions"]) == 3 for row in response.json())
    assert "draft-" not in response.text


async def test_equal_named_definitions_remain_distinct_across_pages(
    session: Session, client: AsyncClient
) -> None:
    seed_definitions(session)
    pages = [await client.get(f"/api/workflows?limit=2&offset={offset}") for offset in (0, 2, 4, 6)]
    assert all(page.status_code == 200 for page in pages)
    assert [[row["id"] for row in page.json()] for page in pages] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
        [],
    ]


async def test_legacy_definition_list_retains_all_visible_revisions(
    session: Session, client: AsyncClient
) -> None:
    seed_definitions(session)
    response = await client.get("/api/workflows")
    assert response.status_code == 200
    assert {row["id"] for row in response.json()} == {"a", "b", "c", "d", "e"}
    assert all(len(row["revisions"]) == 3 for row in response.json())


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=-1", "limit=201", "offset=-1", "offset=9223372036854775808"]
)
async def test_definition_page_bounds_are_validated(client: AsyncClient, query: str) -> None:
    response = await client.get("/api/workflows?" + query)
    assert response.status_code == 422
