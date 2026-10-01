from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, select, update
from sqlalchemy.orm import Session
from test_workflow_library_api import _family

from local_lm.db import SessionLocal
from local_lm.models import (
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)


def _seed(names: list[str]) -> list[str]:
    identifiers = []
    with SessionLocal() as session:
        for name in names:
            rows = _family(name=name)
            session.add_all(rows[:4])
            session.flush()
            rows[1].current_revision_id = rows[2].id
            identifiers.append(rows[0].id)
        session.commit()
    return identifiers


async def _page(client: AsyncClient, **params: str | int) -> list[dict[str, object]]:
    response = await client.get(
        "/api/workflow-families",
        params={"search": "Paging", "limit": 2, "variant_limit": 2, **params},
    )
    assert response.status_code == 200, response.json()
    return response.json()


async def test_family_capability_counts_exclude_unrelated_ready_variants(
    client: AsyncClient,
) -> None:
    identifier = _seed(["Paging mixed"])[0]
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == identifier)
        )
        assert definition is not None
        definition.operation = "text_to_video"
        session.add(
            WorkflowDefinition(
                family_id=identifier,
                variant_key="edit",
                name="Image edit",
                operation="image_to_image",
            )
        )
        session.commit()
    for url, params in [
        ("/api/workflow-families", {"family_ids": identifier, "limit": 1}),
        (f"/api/workflow-families/{identifier}", {}),
    ]:
        response = await client.get(
            url,
            params={
                **params,
                "variant_limit": 1,
                "variant_capability": "image",
            },
        )
        assert response.status_code == 200
        row = response.json()[0] if isinstance(response.json(), list) else response.json()
        assert row["variant_count"] == 1
        assert row["ready_variant_count"] == 0
        assert row["best_readiness"] == "setup_required"
        assert row["variants"][0]["operation"] == "image_to_image"


@pytest.mark.parametrize("detail", [False, True])
async def test_family_offer_capabilities_include_variants_outside_the_page(
    client: AsyncClient, detail: bool
) -> None:
    identifier = _seed(["Paging mixed offers"])[0]
    with SessionLocal() as session:
        image = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == identifier)
        )
        assert image is not None
        image_id = image.id
        session.add(
            WorkflowDefinition(
                family_id=identifier,
                variant_key="video",
                name="Video choice",
                operation="text_to_video",
            )
        )
        session.commit()
    response = await client.get(
        f"/api/workflow-families/{identifier}" if detail else "/api/workflow-families",
        params={
            "family_ids": identifier,
            "workflow_ids": image_id,
            "limit": 1,
            "variant_limit": 1,
            "variant_capability": "image",
        },
    )
    assert response.status_code == 200
    row = response.json() if detail else response.json()[0]
    assert [variant["id"] for variant in row["variants"]] == [image_id]
    assert row["variant_count"] == 1
    assert row["supported_selector_capabilities"] == ["image", "video"]


@pytest.mark.parametrize("declared", [False, True])
async def test_family_capability_filter_precedes_family_limits(
    client: AsyncClient,
    declared: bool,
) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta"])
    with SessionLocal() as session:
        for definition in session.scalars(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id.in_(ids))
        ):
            if definition.family_id == ids[0] or declared:
                definition.operation = "text_to_video"
            if definition.family_id == ids[1] and declared:
                revision = session.get(WorkflowRevision, definition.current_revision_id)
                assert revision is not None
                revision.capabilities_json = ["image"]
        session.commit()
    rows = await _page(client, variant_capability="image", limit=1, variant_limit=1)
    assert [row["id"] for row in rows] == ids[1:]
    assert rows[0]["ready_variant_count"] == 1


async def test_family_pages_are_bounded_and_have_stable_ties(client: AsyncClient) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        for family in session.scalars(select(WorkflowFamily).where(WorkflowFamily.id.in_(ids))):
            family.name = "Paging same name"
        session.commit()
    first = await _page(client)
    second = await _page(client, offset=2)
    assert [row["id"] for row in first] == sorted(ids)[:2]
    assert [row["id"] for row in second] == sorted(ids)[2:]
    assert await _page(client, offset=3) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", "0"),
        ("limit", "201"),
        ("offset", "-1"),
        ("offset", str(2**63)),
        ("variant_limit", "0"),
        ("variant_limit", "201"),
        ("variant_offset", "-1"),
        ("variant_offset", str(2**63)),
        ("search", "x" * 501),
        ("readiness", "not-a-readiness"),
        ("source", "not-a-source"),
        ("order", "not-an-order"),
    ],
)
async def test_family_pages_reject_invalid_bounds_and_filters(
    client: AsyncClient, field: str, value: str
) -> None:
    response = await client.get("/api/workflow-families", params={field: value})
    assert response.status_code == 422


async def test_family_search_matches_off_page_metadata_and_literal_unicode(
    client: AsyncClient,
) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        family = session.get(WorkflowFamily, ids[2])
        assert family is not None
        family.tags_json = ["Straße_100%"]
        session.commit()
    rows = await _page(client, search="STRASSE_100%", limit=1)
    assert [row["id"] for row in rows] == ids[2:]
    assert await _page(client, search="STRASSE_100%", offset=2**63 - 1) == []


async def test_family_search_matches_variants_beyond_the_variant_page(client: AsyncClient) -> None:
    identifier = _seed(["Paging Alpha"])[0]
    with SessionLocal() as session:
        session.add(
            WorkflowDefinition(
                family_id=identifier,
                variant_key="late",
                name="Searchable late variant",
                operation="text_to_video",
            )
        )
        session.commit()
    rows = await _page(client, search="Searchable late", variant_limit=1)
    assert [row["id"] for row in rows] == [identifier]
    assert len(rows[0]["variants"]) == 1


async def test_family_pages_filter_variants_and_readiness_before_limiting(
    client: AsyncClient,
) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == ids[2])
        )
        assert definition is not None
        definition.operation = "image_to_image"
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        assert revision is not None
        revision.trusted = False
        session.commit()
    rows = await _page(client, operation="image_to_image", readiness="review_required", limit=1)
    assert [row["id"] for row in rows] == ids[2:]
    assert rows[0]["variant_count"] == 1
    assert rows[0]["ready_variant_count"] == 0
    assert await _page(client, operation="image_to_image", readiness="ready") == []


async def test_family_variant_pages_keep_whole_family_readiness(client: AsyncClient) -> None:
    identifier = _seed(["Paging Alpha"])[0]
    with SessionLocal() as session:
        for index in range(4):
            definition = WorkflowDefinition(
                id=f"workflow_page_extra_{index}",
                family_id=identifier,
                variant_key=f"extra-{index}",
                name=f"Extra {index}",
                operation="image_to_image",
            )
            session.add(definition)
        session.commit()
    response = await client.get(f"/api/workflow-families/{identifier}", params={"variant_limit": 2})
    assert response.status_code == 200
    row = response.json()
    assert len(row["variants"]) == 2
    assert all(variant["readiness"] == "setup_required" for variant in row["variants"])
    assert row["variant_count"] == 5
    assert row["ready_variant_count"] == 1
    assert row["best_readiness"] == "ready"
    next_page = await client.get(
        f"/api/workflow-families/{identifier}",
        params={"variant_limit": 2, "variant_offset": 4},
    )
    assert next_page.status_code == 200
    assert len(next_page.json()["variants"]) == 1
    assert next_page.json()["variants"][0]["readiness"] == "ready"


async def test_family_readiness_order_is_global_before_page_limits(client: AsyncClient) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        first = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == ids[0])
        )
        assert first is not None
        first.current_revision_id = None
        second = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == ids[1])
        )
        assert second is not None
        revision = session.get(WorkflowRevision, second.current_revision_id)
        assert revision is not None
        revision.trusted = False
        session.commit()
    first_page = await _page(client, order="readiness", limit=1)
    assert [row["id"] for row in first_page] == ids[2:]
    assert [row["id"] for row in await _page(client, order="readiness", limit=1, offset=1)] == ids[
        :1
    ]


async def test_family_pages_resolve_exact_off_page_ids(client: AsyncClient) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    response = await client.get(
        "/api/workflow-families", params={"family_ids": ids[2], "limit": 1, "variant_limit": 1}
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ids[2:]
    too_many = await client.get(
        "/api/workflow-families", params=[("family_ids", str(index)) for index in range(201)]
    )
    assert too_many.status_code == 422


async def test_family_preference_order_and_defaults_apply_before_pages(client: AsyncClient) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        preferences = list(
            session.scalars(
                select(WorkflowPreference).where(WorkflowPreference.selector_capability == "image")
            )
        )
        for preference in preferences:
            preference.is_default = False
        session.flush()
        for preference in preferences:
            if preference.workflow_family_id in ids:
                preference.sort_order = 10 - ids.index(preference.workflow_family_id)
                preference.is_default = preference.workflow_family_id == ids[2]
                preference.enabled = preference.workflow_family_id != ids[0]
        session.commit()
    rows = await _page(
        client, selector_capability="image", order="preference", enabled_only="true", limit=1
    )
    assert [row["id"] for row in rows] == ids[2:]
    assert [row["id"] for row in await _page(client, defaults_only="true", limit=1)] == ids[2:]
    assert len(await _page(client, selector_capability="image", enabled_only="true", limit=10)) == 2


async def test_family_source_and_archive_filters_apply_before_pages(client: AsyncClient) -> None:
    ids = _seed(["Paging Alpha", "Paging Beta", "Paging Gamma"])
    with SessionLocal() as session:
        profile = ModelProfile(name="Paging compatibility", role="image", engine="mock")
        session.add(profile)
        session.flush()
        session.add(
            WorkflowProfileCompatibility(
                workflow_family_id=ids[2],
                model_profile_id=profile.id,
                source_fingerprint_sha256="a" * 64,
            )
        )
        family = session.get(WorkflowFamily, ids[2])
        assert family is not None
        family.archived = True
        session.commit()
    assert await _page(client, source="profile", limit=1) == []
    rows = await _page(client, source="profile", include_archived="true", limit=1)
    assert [row["id"] for row in rows] == ids[2:]
    assert rows[0]["best_readiness"] == "unavailable"


async def test_family_pages_search_dependency_names_without_exposing_graphs(
    client: AsyncClient,
) -> None:
    from test_workflow_family_dependency_search import _seed_family

    with SessionLocal() as session:
        identifier, _, _, _ = _seed_family(session, "Paging dependency")
        session.commit()
    rows = await _page(client, search="dependency checkpoint", include_dependencies="true", limit=1)
    assert [row["id"] for row in rows] == [identifier]
    assert rows[0]["dependency_summary"]["dependency_count"] == 1
    assert "api_graph_json" not in str(rows)


async def test_family_operation_choices_include_off_page_variants(client: AsyncClient) -> None:
    identifier = _seed(["Paging Alpha"])[0]
    with SessionLocal() as session:
        session.add(
            WorkflowDefinition(
                family_id=identifier, variant_key="video", name="Video", operation="text_to_video"
            )
        )
        session.commit()
    response = await client.get("/api/workflow-family-operations")
    assert response.status_code == 200
    assert "text_to_video" in response.json()


async def test_family_pages_keep_unreviewed_package_drafts_out_of_choices(
    client: AsyncClient,
) -> None:
    from local_lm.workflow_package_drafts import WORKFLOW_PACKAGE_DRAFT_MARKER

    ids = _seed(["Paging Alpha", "Paging Beta"])
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == ids[0])
        )
        assert definition is not None
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        assert revision is not None
        revision.dependencies_json = {WORKFLOW_PACKAGE_DRAFT_MARKER: {}}
        definition.operation = "text_to_video"
        session.execute(
            update(WorkflowDefinition)
            .where(
                WorkflowDefinition.id != definition.id,
                WorkflowDefinition.operation == "text_to_video",
            )
            .values(operation="text_to_image")
        )
        session.commit()
    assert [row["id"] for row in await _page(client, limit=1)] == ids[1:]
    operations = await client.get("/api/workflow-family-operations")
    assert operations.status_code == 200
    assert "text_to_video" not in operations.json()


async def test_family_variant_pages_do_not_hydrate_the_whole_family(client: AsyncClient) -> None:
    identifier = _seed(["Paging Alpha"])[0]
    with SessionLocal() as session:
        for index in range(150):
            definition = WorkflowDefinition(
                id=f"workflow_page_bounded_{index}",
                family_id=identifier,
                variant_key=f"extra-{index}",
                name=f"Extra {index}",
                operation="image_to_image",
            )
            revision = WorkflowRevision(
                id=f"wfrev_page_bounded_{index}",
                definition=definition,
                version=1,
                engine="mock",
                trusted=True,
                api_graph_json={"neutral": "x" * 2000},
            )
            session.add_all([definition, revision])
            session.flush()
            definition.current_revision_id = revision.id
        session.commit()
    peaks = {"definitions": 0, "revisions": 0}

    def track(session: Session, _instance: object) -> None:
        for key, model in (("definitions", WorkflowDefinition), ("revisions", WorkflowRevision)):
            peaks[key] = max(
                peaks[key], sum(isinstance(row, model) for row in session.identity_map.values())
            )

    event.listen(Session, "loaded_as_persistent", track)
    try:
        response = await client.get(
            f"/api/workflow-families/{identifier}", params={"variant_limit": 2}
        )
    finally:
        event.remove(Session, "loaded_as_persistent", track)
    assert response.status_code == 200
    assert len(response.json()["variants"]) == 2
    assert response.json()["variant_count"] == 151
    assert peaks["definitions"] <= 101, peaks
    assert peaks["revisions"] <= 2
