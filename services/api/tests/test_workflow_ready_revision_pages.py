from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session
from test_workflow_library_api import _family

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowRevision
from local_lm.workflow_package_drafts import WORKFLOW_PACKAGE_DRAFT_MARKER


def _seed() -> list[str]:
    revisions = []
    with SessionLocal() as session:
        for name in ("Ready Alpha", "Ready Beta"):
            family, definition, revision, preference, *_ = _family(name=name)
            session.add_all([family, definition, revision, preference])
            session.flush()
            definition.current_revision_id = revision.id
            revisions.append(revision.id)
            for index in range(3):
                extra = WorkflowDefinition(
                    family_id=family.id,
                    name=f"{name} variant {index}",
                    variant_key=f"extra-{index}",
                    operation="text_to_image",
                )
                version = WorkflowRevision(
                    definition=extra,
                    version=1,
                    engine="mock",
                    trusted=True,
                    api_graph_json={"node": {"class_type": "MockImage"}},
                )
                session.add_all([extra, version])
                session.flush()
                extra.current_revision_id = version.id
                revisions.append(version.id)
        session.commit()
    return revisions


async def test_ready_revision_pages_bound_variants_across_families(client: AsyncClient) -> None:
    expected = _seed()
    found: list[str] = []
    for offset in (0, 3, 6):
        response = await client.get(
            "/api/workflow-ready-revisions",
            params={"search": "Ready", "limit": 3, "offset": offset},
        )
        assert response.status_code == 200
        assert len(response.json()) <= 3
        found.extend(row["revision_id"] for row in response.json())
        assert "api_graph_json" not in str(response.json())
    assert found == expected


async def test_ready_revision_exact_lookup_ignores_browsing_offset(client: AsyncClient) -> None:
    expected = _seed()
    response = await client.get(
        "/api/workflow-ready-revisions", params={"revision_id": expected[-1], "limit": 1}
    )
    assert response.status_code == 200
    assert [row["revision_id"] for row in response.json()] == expected[-1:]


async def test_ready_revision_exact_lookup_does_not_load_unrelated_graphs(
    client: AsyncClient,
) -> None:
    expected = _seed()
    loaded = []

    def track(_session: Session, instance: object) -> None:
        if isinstance(instance, WorkflowRevision):
            loaded.append(instance.id)

    event.listen(Session, "loaded_as_persistent", track)
    try:
        response = await client.get(
            "/api/workflow-ready-revisions",
            params={"search": "Ready", "revision_id": expected[-1], "limit": 1},
        )
    finally:
        event.remove(Session, "loaded_as_persistent", track)
    assert response.status_code == 200
    assert [row["revision_id"] for row in response.json()] == expected[-1:]
    assert loaded == expected[-1:]


@pytest.mark.parametrize(
    "excluded", ["unready", "disabled", "archived", "preference", "operation", "draft"]
)
async def test_ready_revision_eligibility_precedes_page_limits(
    client: AsyncClient,
    excluded: str,
) -> None:
    with SessionLocal() as session:
        first = _family(name="Ready Alpha")
        second = _family(name="Ready Beta")
        session.add_all([*first[:4], *second[:4]])
        session.flush()
        first[1].current_revision_id = first[2].id
        second[1].current_revision_id = second[2].id
        if excluded == "unready":
            first[2].trusted = False
        elif excluded == "disabled":
            first[0].enabled = False
        elif excluded == "archived":
            first[0].archived = True
        elif excluded == "preference":
            first[3].enabled = False
        elif excluded == "operation":
            first[1].operation = "image_to_image"
        else:
            first[2].dependencies_json = {WORKFLOW_PACKAGE_DRAFT_MARKER: {}}
        session.commit()
        expected = second[2].id
    response = await client.get(
        "/api/workflow-ready-revisions", params={"search": "Ready", "limit": 1}
    )
    assert response.status_code == 200
    assert [row["revision_id"] for row in response.json()] == [expected]


async def test_ready_revision_search_is_literal_and_casefolded(client: AsyncClient) -> None:
    expected = _seed()
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(
                WorkflowDefinition.current_revision_id == expected[-1],
            )
        )
        assert definition is not None
        definition.name = "Straße %_ studio"
        session.commit()
    response = await client.get(
        "/api/workflow-ready-revisions", params={"search": "STRASSE %_", "limit": 1}
    )
    assert response.status_code == 200
    assert [row["revision_id"] for row in response.json()] == expected[-1:]


async def test_ready_revision_pages_resolve_and_deduplicate_legacy_profile_choices(
    client: AsyncClient,
) -> None:
    for name in ("Legacy Ready Alpha", "Legacy Ready Beta"):
        created = await client.post(
            "/api/profiles",
            json={"name": name, "role": "image", "engine": "mock"},
        )
        assert created.status_code == 201
    families = await client.get("/api/workflow-families", params={"search": "Legacy Ready"})
    assert families.status_code == 200
    expected = {
        variant["current_revision_id"]
        for family in families.json()
        for variant in family["variants"]
        if variant["operation"] == "text_to_image" and variant["readiness"] == "ready"
    }
    assert len(expected) == 1
    response = await client.get(
        "/api/workflow-ready-revisions",
        params={"search": "Legacy Ready", "limit": 1},
    )
    assert response.status_code == 200
    assert {row["revision_id"] for row in response.json()} == expected
    next_page = await client.get(
        "/api/workflow-ready-revisions",
        params={"search": "Legacy Ready", "limit": 1, "offset": 1},
    )
    assert next_page.json() == []
    exact = await client.get(
        "/api/workflow-ready-revisions",
        params={"search": "Legacy Ready", "revision_id": list(expected)},
    )
    assert {row["revision_id"] for row in exact.json()} == expected


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "201"},
        {"offset": "-1"},
        {"offset": str(2**63)},
        {"search": "a" * 501},
        {"revision_id": [str(i) for i in range(201)]},
    ],
)
async def test_ready_revision_pages_reject_invalid_bounds(
    client: AsyncClient, params: dict[str, str | list[str]]
) -> None:
    response = await client.get("/api/workflow-ready-revisions", params=params)
    assert response.status_code == 422
