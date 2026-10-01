from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.models import GenerationPreset, ModelInstall, ModelProfile

KINDS = ("models", "profiles", "presets")


def _seed(kind: str, names: list[str]) -> list[str]:
    ids = [f"paged-{kind}-{index}" for index in range(len(names))]
    with SessionLocal() as session:
        for index, name in enumerate(names):
            values = {"id": ids[index], "name": name, "role": "image"}
            if kind == "models":
                session.add(
                    ModelInstall(
                        **values,
                        engine="mock",
                        local_path=f"C:/neutral-fixture/{index}",
                        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
                    )
                )
            elif kind == "profiles":
                session.add(ModelProfile(**values, engine="mock"))
            else:
                session.add(GenerationPreset(**values))
        session.commit()
    return ids


@pytest.mark.parametrize("kind", KINDS)
async def test_library_pages_bound_reads_and_keep_legacy_lists(
    client: AsyncClient, kind: str
) -> None:
    ids = _seed(kind, [f"Pagination fixture {index}" for index in range(5)])
    complete = await client.get(f"/api/{kind}")
    page = await client.get(
        f"/api/{kind}", params={"search": "Pagination fixture", "limit": 2, "offset": 1}
    )
    end = await client.get(
        f"/api/{kind}", params={"search": "Pagination fixture", "limit": 2, "offset": 5}
    )
    assert complete.status_code == page.status_code == end.status_code == 200
    assert set(ids) <= {row["id"] for row in complete.json()}
    assert [row["id"] for row in page.json()] == ids[1:3]
    assert end.json() == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("query", ["STRASSE", "%_"])
async def test_library_search_is_literal_and_applies_before_pages(
    client: AsyncClient, kind: str, query: str
) -> None:
    ids = _seed(kind, ["Straße %_ 0", "Other fixture", "Straße %_ 1", "Straße %_ 2"])
    response = await client.get(f"/api/{kind}", params={"search": query, "limit": 1, "offset": 1})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ids[2:3]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "parameters",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"offset": 2**63}, {"search": "x" * 501}],
)
async def test_library_pages_refuse_invalid_bounds(
    client: AsyncClient, kind: str, parameters: dict[str, str | int]
) -> None:
    response = await client.get(f"/api/{kind}", params=parameters)
    assert response.status_code == 422


@pytest.mark.parametrize("kind", KINDS)
async def test_library_identity_reads_recover_choices_outside_the_first_page(
    client: AsyncClient, kind: str
) -> None:
    ids = _seed(kind, [f"Pagination identity {index}" for index in range(5)])
    key = {"models": "model_id", "profiles": "profile_id", "presets": "preset_id"}[kind]
    response = await client.get(
        f"/api/{kind}", params=[(key, ids[3]), (key, ids[4]), (key, ids[3]), ("limit", "2")]
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ids[3:]


@pytest.mark.parametrize("kind", KINDS)
async def test_library_identity_reads_refuse_oversized_batches(
    client: AsyncClient, kind: str
) -> None:
    key = {"models": "model_id", "profiles": "profile_id", "presets": "preset_id"}[kind]
    response = await client.get(
        f"/api/{kind}", params=[(key, f"item-{index}") for index in range(201)]
    )
    assert response.status_code == 422


@pytest.mark.parametrize("kind", ("profiles", "presets"))
async def test_library_default_reads_filter_before_the_limit(
    client: AsyncClient, kind: str
) -> None:
    ids = _seed(kind, [f"Pagination default {index}" for index in range(4)])
    with SessionLocal() as session:
        row = session.get(ModelProfile if kind == "profiles" else GenerationPreset, ids[-1])
        assert row is not None
        row.is_default = True
        session.commit()
    response = await client.get(
        f"/api/{kind}", params={"search": "Pagination default", "defaults_only": True, "limit": 1}
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ids[-1:]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("search", ["", "Pagination hydration"])
async def test_library_pages_hydrate_only_returned_rows(
    client: AsyncClient, kind: str, search: str
) -> None:
    _seed(kind, [f"Pagination hydration {index}" for index in range(8)])
    model = {"models": ModelInstall, "profiles": ModelProfile, "presets": GenerationPreset}[kind]
    loaded: list[str] = []

    def track(row: ModelInstall | ModelProfile | GenerationPreset, _context: object) -> None:
        loaded.append(row.id)

    event.listen(model, "load", track)
    try:
        response = await client.get(
            f"/api/{kind}", params={"search": search, "limit": 2, "offset": 2}
        )
    finally:
        event.remove(model, "load", track)
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert loaded == [row["id"] for row in response.json()]
