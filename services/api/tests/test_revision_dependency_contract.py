"""A revision's declared dependency contract, stored where the install gate reads it.

Every case drives a real entry point - the creation and revision routes, the
review route, the offer route - rather than calling the helper directly, because
the defect this closes was a path that never called anything at all. A helper
test would have passed throughout.

EXPECTED VALUES ARE BUILT INDEPENDENTLY of the helper under test. Canonical slot
order is written out by hand, and the expected digests come from the existing
identity functions applied to a hand-built canonical contract, never from
re-parsing the input that was sent.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowDependencySlot, WorkflowRevision
from local_lm.revision_dependency_contract import persist_dependency_contract
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    WorkflowDependencySlotContract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)

pytestmark = pytest.mark.asyncio

_UI_GRAPH: dict[str, Any] = {
    "version": 0.4,
    "nodes": [
        {
            "id": 1,
            "type": "EmptyLatentImage",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": [512, 512, 1],
        }
    ],
    "links": [],
}


def _slot(name: str, requirement_keys: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "resource_kind": "custom_node",
        "required": True,
        "satisfaction": "all_of",
        "requirements": [{"key": key, "constraints": {}} for key in requirement_keys],
    }


async def _create(
    client: AsyncClient, dependencies: object, name: str = "Declared workflow"
) -> Any:
    return await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": "text_to_image",
            "engine": "comfyui",
            "ui_graph": _UI_GRAPH,
            "api_graph": _GRAPH,
            "input_schema": {},
            "dependencies": dependencies,
        },
    )


def _stored(revision_id: str) -> tuple[str | None, list[WorkflowDependencySlot]]:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        rows = list(
            session.scalars(
                select(WorkflowDependencySlot)
                .where(WorkflowDependencySlot.workflow_revision_id == revision_id)
                .order_by(WorkflowDependencySlot.ordinal)
            ).all()
        )
        session.expunge_all()
        return revision.dependency_contract_sha256, rows


async def test_slots_are_stored_in_the_parsers_order_not_the_authors(client: AsyncClient) -> None:
    """Declared zeta-then-alpha with requirements reversed; stored alpha-then-zeta.

    The install gate rebuilds the contract from rows in ordinal order, so rows
    in declaration order would read back as drift.
    """

    response = await _create(
        client,
        {
            "version": 1,
            "slots": [_slot("zeta-nodes", ["zz", "aa"]), _slot("alpha-nodes", ["mm", "bb"])],
        },
    )
    assert response.status_code == 201, response.text
    digest, rows = _stored(response.json()["current_revision_id"])

    # Written by hand, not derived from what was sent.
    expected_alpha = WorkflowDependencySlotContract(
        name="alpha-nodes",
        resource_kind="custom_node",
        required=True,
        satisfaction="all_of",
        requirements=(
            WorkflowDependencyRequirement("bb", {}),
            WorkflowDependencyRequirement("mm", {}),
        ),
    )
    expected_zeta = WorkflowDependencySlotContract(
        name="zeta-nodes",
        resource_kind="custom_node",
        required=True,
        satisfaction="all_of",
        requirements=(
            WorkflowDependencyRequirement("aa", {}),
            WorkflowDependencyRequirement("zz", {}),
        ),
    )
    expected = WorkflowDependencyContract(version=1, slots=(expected_alpha, expected_zeta))

    assert [(row.ordinal, row.name) for row in rows] == [(0, "alpha-nodes"), (1, "zeta-nodes")]
    assert [item["key"] for item in rows[0].requirements_json] == ["bb", "mm"]
    assert rows[0].contract_sha256 == workflow_dependency_slot_sha256(expected_alpha)
    assert rows[1].contract_sha256 == workflow_dependency_slot_sha256(expected_zeta)
    assert digest == workflow_dependency_contract_sha256(expected)


async def test_an_empty_object_declares_nothing_and_stores_nothing(client: AsyncClient) -> None:
    """`{}` is not a claim that nothing is needed, so nothing is inferred from it."""

    response = await _create(client, {})
    assert response.status_code == 201, response.text
    digest, rows = _stored(response.json()["current_revision_id"])

    assert digest is None
    assert rows == []


async def test_an_explicit_empty_contract_gets_the_canonical_empty_digest(
    client: AsyncClient,
) -> None:
    response = await _create(client, {"version": 1, "slots": []})
    assert response.status_code == 201, response.text
    digest, rows = _stored(response.json()["current_revision_id"])

    assert rows == []
    assert digest == workflow_dependency_contract_sha256(
        WorkflowDependencyContract(version=1, slots=())
    )


@pytest.mark.parametrize(
    "declared",
    [
        {"version": 1, "slots": "not a list"},
        {"version": 2, "slots": []},
        {"version": "1", "slots": []},
        {"version": 1, "slots": [{"name": "incomplete"}]},
    ],
    ids=["slots-not-a-list", "unsupported-version", "version-not-an-integer", "incomplete-slot"],
)
async def test_a_malformed_versioned_declaration_is_refused_and_leaves_nothing(
    client: AsyncClient, declared: dict[str, Any]
) -> None:
    """Refused, never downgraded to legacy - and refused before anything is written."""

    name = f"Refused {declared!r}"
    response = await _create(client, declared, name=name)

    assert response.status_code == 422, response.text
    with SessionLocal() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(WorkflowDefinition)
                .where(WorkflowDefinition.name == name)
            )
            == 0
        )


async def test_a_new_revision_owns_its_own_rows(client: AsyncClient) -> None:
    first = await _create(client, {"version": 1, "slots": [_slot("first-nodes", ["a"])]})
    assert first.status_code == 201, first.text
    workflow_id = first.json()["id"]
    first_revision = first.json()["current_revision_id"]
    first_digest, first_rows = _stored(first_revision)

    second = await client.post(
        f"/api/workflows/{workflow_id}/revisions",
        json={
            "ui_graph": _UI_GRAPH,
            "api_graph": _GRAPH,
            "input_schema": {},
            "dependencies": {"version": 1, "slots": [_slot("second-nodes", ["b"])]},
        },
    )
    assert second.status_code in (200, 201), second.text
    second_digest, second_rows = _stored(second.json()["id"])

    assert [row.name for row in second_rows] == ["second-nodes"]
    assert second_digest != first_digest
    # The earlier revision's identity is exactly as it was: same digest, same rows.
    after_digest, after_rows = _stored(first_revision)
    assert after_digest == first_digest
    assert [(row.id, row.contract_sha256) for row in after_rows] == [
        (row.id, row.contract_sha256) for row in first_rows
    ]
    assert {row.id for row in first_rows}.isdisjoint({row.id for row in second_rows})


async def test_a_malformed_declaration_on_a_new_revision_is_refused(client: AsyncClient) -> None:
    first = await _create(client, {})
    assert first.status_code == 201, first.text

    second = await client.post(
        f"/api/workflows/{first.json()['id']}/revisions",
        json={
            "ui_graph": _UI_GRAPH,
            "api_graph": _GRAPH,
            "input_schema": {},
            "dependencies": {"version": 1, "slots": "not a list"},
        },
    )

    assert second.status_code == 422, second.text


async def test_the_contract_is_written_once_by_the_creating_transaction(
    client: AsyncClient,
) -> None:
    """A second write would duplicate identity or silently disagree with it."""

    response = await _create(client, {"version": 1, "slots": [_slot("only-nodes", ["a"])]})
    assert response.status_code == 201, response.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, response.json()["current_revision_id"])
        assert revision is not None
        with pytest.raises(RuntimeError, match="already has dependency slots"):
            persist_dependency_contract(session, revision)


_REFERENCE = "styles/detail.safetensors"


def _lora_ui_graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoraLoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [_REFERENCE],
                "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
            }
        ],
        "links": [],
    }


_LORA_API_GRAPH = {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": _REFERENCE}}}


def _seed_model_plan() -> str:
    """The model download the offer selects - a resource plan, not trust or a digest."""

    import hashlib

    from local_lm.model_planner import INSTALL_RESOLVER_VERSION
    from local_lm.models import InstallPlan

    with SessionLocal() as session:
        plan = InstallPlan(
            id="plan_contract_acceptance",
            provider="civitai",
            remote_id="101",
            revision="202",
            role="image",
            engine="comfyui",
            plan_hash=hashlib.sha256(b"plan_contract_acceptance").hexdigest(),
            resolver_version=INSTALL_RESOLVER_VERSION,
            compatibility="supported",
            artifacts_json=[
                {
                    "path": _REFERENCE,
                    "kind": "lora",
                    "target_folder": "loras",
                    "size_bytes": 17,
                    "sha256": "a" * 64,
                    "required": True,
                    "reuse": "download",
                    "source_version_id": "202",
                    "source_file_id": "301",
                }
            ],
            runtime_contract_json={"auxiliary_kind": "lora", "comfy_paths": {"loras": "styles"}},
            activation_probe_json={},
            status="planned",
        )
        session.add(plan)
        session.commit()
        return plan.id


async def test_a_created_workflow_reaches_an_install_offer_through_real_review(
    app: Any,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance this increment exists for.

    Created through the route, trusted through the review route, offered through
    the offer route. Nothing seeded by hand stands in for trust or for the
    contract digest; the only fixture is the model plan an offer has to select,
    since an offer with nothing to install is refused as a malformed request.
    """

    async def object_info() -> dict[str, Any]:
        return {
            "LoraLoader": {
                "python_module": "nodes",
                "input": {"required": {"lora_name": [[_REFERENCE], {}]}},
                "output": ["MODEL", "CLIP"],
            }
        }

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info, raising=False)

    created = await client.post(
        "/api/workflows",
        json={
            "name": "Offer acceptance workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "ui_graph": _lora_ui_graph(),
            "api_graph": _LORA_API_GRAPH,
            "input_schema": {},
            "dependencies": {"version": 1, "slots": []},
        },
    )
    assert created.status_code == 201, created.text
    workflow_id = created.json()["id"]
    revision_id = created.json()["current_revision_id"]

    review_url = f"/api/workflows/{workflow_id}/revisions/{revision_id}/review"
    preview = await client.get(review_url)
    assert preview.status_code == 200, preview.text
    approved = await client.post(
        review_url, json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]}
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["trusted"] is True

    plan_id = _seed_model_plan()
    offer = await client.post(
        f"/api/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
        json={
            "selections": [
                {
                    "reference_filename": _REFERENCE,
                    "install_plan_id": plan_id,
                    "artifact_path": _REFERENCE,
                }
            ]
        },
    )
    assert offer.status_code == 201, offer.text


def _portable(dependencies: dict[str, Any], suffix: str) -> Any:
    from local_lm.project_dependencies import (
        PortableDependencies,
        PortableWorkflow,
        PortableWorkflowRevision,
    )

    revision = PortableWorkflowRevision(
        source_id=f"source_revision_{suffix}",
        source_version=1,
        engine="comfyui",
        ui_graph=_UI_GRAPH,
        api_graph=_GRAPH,
        dependencies=dependencies,
        # Declared trusted by the archive; import must not honour it.
        trusted=True,
    )
    return PortableDependencies(
        workflows=[
            PortableWorkflow(
                source_id=f"source_workflow_{suffix}",
                name=f"Imported workflow {suffix}",
                operation="text_to_image",
                current_revision_source_id=revision.source_id,
                revisions=[revision],
            )
        ]
    )


async def test_an_imported_revision_carries_its_declared_contract(client: AsyncClient) -> None:
    """The declaration crosses an archive boundary; trust still does not."""

    from local_lm.project_dependencies import install_dependency_manifest

    with SessionLocal() as session:
        imported = install_dependency_manifest(
            session, _portable({"version": 1, "slots": [_slot("imported-nodes", ["k"])]}, "valid")
        )
        session.commit()
        revision_id = imported.revision_ids["source_revision_valid"]

    digest, rows = _stored(revision_id)
    expected_slot = WorkflowDependencySlotContract(
        name="imported-nodes",
        resource_kind="custom_node",
        required=True,
        satisfaction="all_of",
        requirements=(WorkflowDependencyRequirement("k", {}),),
    )
    assert [row.name for row in rows] == ["imported-nodes"]
    assert rows[0].contract_sha256 == workflow_dependency_slot_sha256(expected_slot)
    assert digest == workflow_dependency_contract_sha256(
        WorkflowDependencyContract(version=1, slots=(expected_slot,))
    )
    with SessionLocal() as session:
        stored = session.get(WorkflowRevision, revision_id)
        assert stored is not None and stored.trusted is False


async def test_an_archive_with_a_malformed_declaration_is_refused(client: AsyncClient) -> None:
    from local_lm.project_dependencies import install_dependency_manifest
    from local_lm.workflow_dependencies import WorkflowDependencyError

    with SessionLocal() as session, pytest.raises(WorkflowDependencyError):
        install_dependency_manifest(
            session, _portable({"version": 1, "slots": "not a list"}, "malformed")
        )
