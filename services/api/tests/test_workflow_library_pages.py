from __future__ import annotations

from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_family_pages import _seed

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition


async def test_selected_variant_finds_its_family_beyond_both_page_limits(
    client: AsyncClient,
) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        session.add(
            WorkflowDefinition(
                id="selected-late-variant",
                family_id=ids[2],
                variant_key="last",
                name="Late selected variant",
                operation="text_to_video",
            )
        )
        session.commit()
    response = await client.get(
        "/api/workflow-families",
        params={
            "workflow_ids": "selected-late-variant",
            "limit": 1,
            "variant_limit": 1,
        },
    )
    assert response.status_code == 200
    rows = response.json()
    assert [row["id"] for row in rows] == ids[2:]
    assert [variant["id"] for variant in rows[0]["variants"]] == ["selected-late-variant"]
    assert rows[0]["variant_count"] == 1


async def test_removed_variant_does_not_match_another_family(client: AsyncClient) -> None:
    _seed(["Paging Alpha"])
    response = await client.get(
        "/api/workflow-families",
        params={
            "workflow_ids": "missing-variant",
            "limit": 1,
            "variant_limit": 1,
        },
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_selected_variant_lookup_rejects_oversized_identity_batches(
    client: AsyncClient,
) -> None:
    response = await client.get(
        "/api/workflow-families", params=[("workflow_ids", str(index)) for index in range(201)]
    )
    assert response.status_code == 422


async def test_ungrouped_summaries_are_filtered_before_the_page_limit(client: AsyncClient) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == ids[2])
        )
        assert definition is not None
        identifier = definition.id
        definition.family_id = None
        session.commit()
    response = await client.get(
        "/api/workflow-summaries",
        params={
            "search": "Paging",
            "ungrouped_only": "true",
            "limit": 1,
        },
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [identifier]
    response = await client.get(
        "/api/workflow-summaries",
        params={
            "search": "Paging",
            "ungrouped_only": "true",
            "limit": 1,
            "offset": 1,
        },
    )
    assert response.status_code == 200
    assert response.json() == []
