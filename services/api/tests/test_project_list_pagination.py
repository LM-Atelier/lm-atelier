from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.models import Project


def _projects(*names: str) -> list[str]:
    identities = [f"paged-project-{number}" for number in range(len(names))]
    with SessionLocal() as session:
        for identity, name in zip(identities, names, strict=True):
            session.add(
                Project(
                    id=identity,
                    name=name,
                    updated_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
        session.commit()
    return identities


async def test_project_pages_bound_results_without_truncating_legacy_reads(
    client: AsyncClient,
) -> None:
    _projects(*(f"Page notebook {number}" for number in range(5)))
    complete = await client.get("/api/projects", params={"query": "Page notebook"})
    page = await client.get(
        "/api/projects", params={"query": "Page notebook", "limit": 2, "offset": 1}
    )
    end = await client.get(
        "/api/projects", params={"query": "Page notebook", "limit": 2, "offset": 5}
    )
    assert complete.status_code == page.status_code == end.status_code == 200
    assert len(complete.json()) == 5
    assert page.json() == complete.json()[1:3]
    assert end.json() == []


async def test_project_pages_keep_pins_then_recency_then_identity(
    client: AsyncClient,
) -> None:
    identities = _projects(*(f"Page order {number}" for number in range(5)))
    with SessionLocal() as session:
        pinned = session.get(Project, identities[0])
        recent = session.get(Project, identities[1])
        assert pinned is not None and recent is not None
        pinned.pinned = True
        recent.updated_at = datetime(2026, 2, 1, tzinfo=UTC)
        session.commit()
    actual: list[str] = []
    for offset in (0, 2, 4):
        page = await client.get(
            "/api/projects",
            params={"query": "Page order", "limit": 2, "offset": offset},
        )
        assert page.status_code == 200
        actual.extend(row["id"] for row in page.json())
    assert actual == [
        identities[0],
        identities[1],
        *sorted(identities[2:], reverse=True),
    ]


async def test_project_pages_filter_archives_before_selecting_the_page(
    client: AsyncClient,
) -> None:
    kept, archived, _other = _projects("Page match", "Page archived", "Other notebook")
    with SessionLocal() as session:
        row = session.get(Project, archived)
        assert row is not None
        row.archived = True
        session.commit()
    page = await client.get("/api/projects", params={"query": "Page", "limit": 1})
    inclusive = await client.get(
        "/api/projects", params={"query": "Page", "include_archived": True, "limit": 1}
    )
    assert page.status_code == inclusive.status_code == 200
    assert [row["id"] for row in page.json()] == [kept]
    assert [row["id"] for row in inclusive.json()] == [archived]


@pytest.mark.parametrize(
    "parameters",
    [{"limit": 0}, {"limit": -1}, {"limit": 201}, {"offset": -1}, {"offset": 2**63}],
)
async def test_project_pages_refuse_invalid_bounds(
    client: AsyncClient, parameters: dict[str, int]
) -> None:
    response = await client.get("/api/projects", params=parameters)
    assert response.status_code == 422


@pytest.mark.parametrize("query", ["café", "%_"])
async def test_literal_project_search_matches_unicode_and_punctuation_before_paging(
    client: AsyncClient, query: str
) -> None:
    identities = _projects("CAFÉ %_ zero", "Unrelated notebook", "CAFÉ %_ one", "CAFÉ %_ two")
    response = await client.get(
        "/api/projects",
        params={"query": query, "literal_search": True, "limit": 1, "offset": 1},
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [identities[2]]


@pytest.mark.parametrize("literal_search", [False, True])
async def test_project_pages_hydrate_only_returned_projects(
    client: AsyncClient, literal_search: bool
) -> None:
    _projects(*(f"Page hydration {number}" for number in range(6)))
    loaded: list[str] = []

    def record_load(project: Project, _context: object) -> None:
        loaded.append(project.id)

    event.listen(Project, "load", record_load)
    try:
        response = await client.get(
            "/api/projects",
            params={
                "query": "Page hydration",
                "literal_search": literal_search,
                "limit": 2,
                "offset": 2,
            },
        )
    finally:
        event.remove(Project, "load", record_load)
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert loaded == [row["id"] for row in response.json()]


async def test_project_identity_filter_recovers_parents_outside_the_first_page(
    client: AsyncClient,
) -> None:
    identities = _projects(*(f"Parent project {number}" for number in range(6)))
    response = await client.get(
        "/api/projects",
        params=[
            ("project_id", identities[0]),
            ("project_id", identities[2]),
            ("project_id", identities[0]),
            ("limit", "2"),
        ],
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [identities[2], identities[0]]


async def test_project_identity_filter_is_bounded(client: AsyncClient) -> None:
    response = await client.get(
        "/api/projects",
        params=[("project_id", f"project-{number}") for number in range(201)],
    )
    assert response.status_code == 422


async def test_selected_project_can_be_read_even_when_archived(
    client: AsyncClient,
) -> None:
    (identity,) = _projects("Selected notebook")
    with SessionLocal() as session:
        row = session.get(Project, identity)
        assert row is not None
        row.archived = True
        row.generation_settings_json = {"image": {"width": 768}}
        session.commit()
    response = await client.get(f"/api/projects/{identity}")
    assert response.status_code == 200
    assert response.json()["id"] == identity
    assert response.json()["archived"] is True
    assert response.json()["generation_settings_json"] == {"image": {"width": 768}}


async def test_missing_selected_project_has_a_stable_refusal(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/projects/missing-project")
    assert response.status_code == 404
    assert response.json()["code"] == "project-not-found"
