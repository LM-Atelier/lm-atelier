from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_workflow_library_api import _family

from local_lm import api
from local_lm.db import SessionLocal
from local_lm.models import (
    ModelInstall,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowRevision,
)


def _seed_family(session: Session, name: str) -> tuple[str, str, str, str]:
    model = ModelInstall(
        name=f"{name} checkpoint",
        role="image",
        engine="mock",
        local_path="managed/neutral-checkpoint",
    )
    session.add(model)
    session.flush()
    rows = _family(name=name, model=model)
    session.add_all(rows)
    session.flush()
    rows[1].current_revision_id = rows[2].id
    session.flush()
    assert rows[4] is not None
    return rows[0].id, rows[1].id, rows[2].id, rows[4].id


async def _listed(client: AsyncClient, **params: str) -> dict[str, Any]:
    response = await client.get("/api/workflow-families", params=params)
    assert response.status_code == 200, response.json()
    return {row["id"]: row for row in response.json()}


async def test_dependency_summaries_are_requested_explicitly(client: AsyncClient) -> None:
    with SessionLocal() as session:
        family_id, _, _, _ = _seed_family(session, "Landscape")
        session.commit()
    plain = await _listed(client)
    assert plain[family_id].get("dependency_summary") is None
    expanded = await _listed(client, include_dependencies="true")
    assert expanded[family_id]["dependency_summary"] == {
        "dependency_count": 1,
        "names": ["Landscape checkpoint", "primary"],
    }


async def test_dependency_summary_uses_current_revision_and_active_bindings(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        family_id, workflow_id, revision_id, activation_id = _seed_family(session, "Landscape")
        _seed_family(session, "Unrelated")
        old = WorkflowRevision(
            id="wfrev_old_search",
            workflow_id=workflow_id,
            version=0,
            engine="mock",
            api_graph_json={},
            trusted=True,
        )
        session.add(old)
        session.flush()
        session.add(
            WorkflowDependencySlot(
                workflow_revision_id=old.id,
                name="old revision only",
                resource_kind="runtime",
                requirements_json=[],
                contract_sha256="a" * 64,
                ordinal=0,
            )
        )
        inactive = WorkflowActivation(
            id="wfact_inactive_search",
            workflow_revision_id=revision_id,
            resolver_version="resolver-v1",
            dependency_contract_sha256="a" * 64,
            binding_sha256="e" * 64,
            state="ready",
            is_active=False,
        )
        session.add(inactive)
        session.flush()
        slot = (
            session.query(WorkflowDependencySlot).filter_by(workflow_revision_id=revision_id).one()
        )
        stale_model = ModelInstall(
            name="Obsolete checkpoint", role="image", engine="mock", local_path="managed/obsolete"
        )
        session.add(stale_model)
        session.flush()
        session.add(
            WorkflowDependencyBinding(
                workflow_revision_id=revision_id,
                workflow_activation_id=inactive.id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="primary",
                model_install_id=stale_model.id,
                resource_identity_json={},
                resource_identity_sha256="f" * 64,
            )
        )
        session.add(
            WorkflowDependencySlot(
                workflow_revision_id=revision_id,
                name="Unbound encoder",
                resource_kind="runtime",
                requirements_json=[],
                contract_sha256="b" * 64,
                ordinal=1,
            )
        )
        session.commit()
    listed = await _listed(client, include_dependencies="true")
    assert listed[family_id]["dependency_summary"] == {
        "dependency_count": 2,
        "names": ["Landscape checkpoint", "primary", "Unbound encoder"],
    }
    with SessionLocal() as session:
        activation = session.get(WorkflowActivation, activation_id)
        assert activation is not None
        activation.is_active = False
        session.commit()
    refreshed = await _listed(client, include_dependencies="true")
    assert refreshed[family_id]["dependency_summary"] == {
        "dependency_count": 2,
        "names": ["primary", "Unbound encoder"],
    }


async def test_empty_or_foreign_current_revisions_do_not_borrow_dependencies(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        first, first_workflow, _, _ = _seed_family(session, "First")
        second, _, other_revision, _ = _seed_family(session, "Second")
        definition = session.get(WorkflowDefinition, first_workflow)
        assert definition is not None
        definition.current_revision_id = other_revision
        empty = WorkflowFamily(id="wffamily_empty_search", name="Empty")
        session.add(empty)
        session.commit()
    listed = await _listed(client, include_dependencies="true")
    assert listed[first]["dependency_summary"] == {"dependency_count": 0, "names": []}
    assert listed["wffamily_empty_search"]["dependency_summary"] == {
        "dependency_count": 0,
        "names": [],
    }
    assert listed[second]["dependency_summary"]["dependency_count"] == 1


async def test_dependency_projection_batches_scalar_queries_for_many_families(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        family_ids = [_seed_family(session, f"Family {index}")[0] for index in range(12)]
        session.commit()
    projection = getattr(api, "workflow_family_dependency_summaries", None)
    assert callable(projection), "The family dependency projection is missing"
    statements: list[str] = []
    loaded: list[object] = []
    with SessionLocal() as session:
        connection = session.connection()

        def record_sql(
            _conn: Any, _cursor: Any, statement: str, _parameters: Any, _context: Any, _many: bool
        ) -> None:
            statements.append(statement)

        def record_load(_session: Session, instance: object) -> None:
            loaded.append(instance)

        event.listen(connection, "before_cursor_execute", record_sql)
        event.listen(session, "loaded_as_persistent", record_load)
        try:
            result = projection(session, family_ids)
        finally:
            event.remove(connection, "before_cursor_execute", record_sql)
            event.remove(session, "loaded_as_persistent", record_load)
        assert set(result) == set(family_ids)
        assert all(item.dependency_count == 1 for item in result.values())
        assert len(statements) == 3
        assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
        assert loaded == []


@pytest.mark.parametrize(
    "kind",
    [
        "model_profile",
        "model_install",
        "model_asset",
        "custom_node",
        "registry_package",
        "runtime",
    ],
)
async def test_dependency_search_names_each_supported_resource_kind(
    client: AsyncClient,
    kind: str,
) -> None:
    from local_lm.models import (
        ComfyRegistryInstall,
        CustomNodeInstall,
        ModelAssetInstall,
        ModelProfile,
    )

    with SessionLocal() as session:
        family_id, _, revision_id, _ = _seed_family(session, "Resource")
        slot = (
            session.query(WorkflowDependencySlot).filter_by(workflow_revision_id=revision_id).one()
        )
        binding = (
            session.query(WorkflowDependencyBinding)
            .filter_by(workflow_dependency_slot_id=slot.id)
            .one()
        )
        slot.resource_kind = kind
        resources: dict[
            str,
            tuple[
                str,
                ModelProfile
                | ModelInstall
                | ModelAssetInstall
                | CustomNodeInstall
                | ComfyRegistryInstall,
                str,
            ],
        ] = {
            "model_profile": (
                "model_profile_id",
                ModelProfile(name="Display profile", role="image", engine="mock"),
                "Display profile",
            ),
            "model_install": (
                "model_install_id",
                ModelInstall(
                    name="Display checkpoint",
                    role="image",
                    engine="mock",
                    local_path="managed/model",
                ),
                "Display checkpoint",
            ),
            "model_asset": (
                "model_asset_install_id",
                ModelAssetInstall(
                    name="Display adapter",
                    kind="lora",
                    family="flux",
                    local_path="managed/adapter",
                    size_bytes=1,
                ),
                "Display adapter",
            ),
            "custom_node": (
                "custom_node_install_id",
                CustomNodeInstall(
                    name="Display package",
                    source_url="https://example.org/package",
                    revision="a" * 40,
                    installed_path="managed/package",
                    tree_hash="a" * 64,
                ),
                "Display package",
            ),
            "registry_package": (
                "comfy_registry_install_id",
                ComfyRegistryInstall(
                    package_id="display-registry",
                    package_version="1",
                    registry_record_id="record",
                    repository_url="https://example.org/source",
                    download_url="https://example.org/archive",
                    archive_sha256="a" * 64,
                    manifest_sha256="b" * 64,
                    installed_path="managed/registry",
                ),
                "display-registry",
            ),
        }
        if kind == "runtime":
            binding.model_install_id = None
            binding.runtime_key = "comfyui"
            expected = "comfyui"
        else:
            field, resource, expected = resources[kind]
            session.add(resource)
            session.flush()
            binding.model_install_id = None
            setattr(binding, field, resource.id)
        session.commit()
    listed = await _listed(client, include_dependencies="true")
    assert listed[family_id]["dependency_summary"] == {
        "dependency_count": 1,
        "names": [expected, "primary"],
    }


async def test_multiple_current_bindings_count_one_slot(client: AsyncClient) -> None:
    with SessionLocal() as session:
        family_id, _, revision_id, activation_id = _seed_family(session, "Resource")
        slot = (
            session.query(WorkflowDependencySlot).filter_by(workflow_revision_id=revision_id).one()
        )
        model = ModelInstall(
            name="Second checkpoint", role="image", engine="mock", local_path="managed/second"
        )
        session.add(model)
        session.flush()
        session.add(
            WorkflowDependencyBinding(
                workflow_revision_id=revision_id,
                workflow_activation_id=activation_id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="secondary",
                model_install_id=model.id,
                resource_identity_json={},
                resource_identity_sha256="f" * 64,
            )
        )
        session.commit()
    listed = await _listed(client, include_dependencies="true")
    assert listed[family_id]["dependency_summary"] == {
        "dependency_count": 1,
        "names": ["primary", "Resource checkpoint", "Second checkpoint"],
    }


async def test_selector_queries_do_not_run_the_dependency_projection(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        family_id, _, _, _ = _seed_family(session, "Resource")
        session.commit()

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A selector requested dependency summaries")

    monkeypatch.setattr(api, "workflow_family_dependency_summaries", unexpected)
    listed = await _listed(client, selector_capability="image")
    assert listed[family_id]["dependency_summary"] is None


async def test_projection_does_not_flush_pending_caller_changes(client: AsyncClient) -> None:
    projection = getattr(api, "workflow_family_dependency_summaries", None)
    assert callable(projection), "The family dependency projection is missing"
    with SessionLocal() as session:
        family_id, _, _, _ = _seed_family(session, "Resource")
        session.commit()
        pending = WorkflowFamily(id="wffamily_pending_search", name="Unflushed family")
        session.add(pending)

        def unexpected_flush(*_args: Any) -> None:
            raise AssertionError("A read-only summary flushed caller changes")

        event.listen(session, "before_flush", unexpected_flush)
        try:
            result = projection(session, [family_id])
            assert result[family_id].dependency_count == 1
            assert pending in session.new
        finally:
            event.remove(session, "before_flush", unexpected_flush)


async def test_real_profile_creation_exposes_its_checkpoint_to_family_search(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        model = ModelInstall(
            name="Granite checkpoint", role="image", engine="mock", local_path="managed/granite"
        )
        session.add(model)
        session.commit()
        model_id = model.id
    created = await client.post(
        "/api/profiles",
        json={
            "name": "Outdoor collection",
            "role": "image",
            "engine": "mock",
            "model_install_id": model_id,
        },
    )
    assert created.status_code == 201, created.json()
    rows = await _listed(client, include_dependencies="true")
    family = next(row for row in rows.values() if row["name"] == "Outdoor collection")
    assert family["compatibility"] is True
    assert "Granite checkpoint" in family["dependency_summary"]["names"]
    assert family["dependency_summary"]["dependency_count"] == 1


async def test_real_workflow_creation_and_revision_use_current_declared_model_names(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        first = ModelInstall(
            name="Granite checkpoint", role="image", engine="mock", local_path="managed/granite"
        )
        second = ModelInstall(
            name="Slate checkpoint", role="image", engine="mock", local_path="managed/slate"
        )
        session.add_all([first, second])
        session.commit()
        first_id, second_id = first.id, second.id
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Outdoor collection",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "dependencies": {"model_install_ids": [first_id]},
        },
    )
    assert created.status_code == 201, created.json()
    family_id = created.json()["family_id"]
    rows = await _listed(client, include_dependencies="true")
    assert rows[family_id]["dependency_summary"]["names"] == ["Granite checkpoint"]
    assert rows[family_id]["dependency_summary"]["dependency_count"] == 1
    revised = await client.post(
        f"/api/workflows/{created.json()['id']}/revisions",
        json={
            "api_graph": {},
            "dependencies": {"model_install_ids": [second_id]},
        },
    )
    assert revised.status_code == 201, revised.json()
    rows = await _listed(client, include_dependencies="true")
    assert rows[family_id]["dependency_summary"]["names"] == ["Slate checkpoint"]


async def test_real_workflow_declarations_include_missing_models_and_named_packages(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Outdoor collection",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "dependencies": {
                "models": [{"name": "Missing checkpoint"}, {"path": "models/encoder.safetensors"}],
                "custom_nodes": [{"name": "Utility nodes"}],
                "registry_packages": [{"package_id": "image-tools", "package_version": "1.0"}],
                "unrelated_metadata": "Not a dependency name",
            },
        },
    )
    assert created.status_code == 201, created.json()
    rows = await _listed(client, include_dependencies="true")
    summary = rows[created.json()["family_id"]]["dependency_summary"]
    assert summary["names"] == [
        "encoder.safetensors",
        "image-tools",
        "Missing checkpoint",
        "Utility nodes",
    ]
    assert summary["dependency_count"] == 4


async def test_real_workflow_aliases_count_one_installed_model(client: AsyncClient) -> None:
    with SessionLocal() as session:
        model = ModelInstall(
            name="Granite checkpoint", role="image", engine="mock", local_path="managed/granite"
        )
        session.add(model)
        session.commit()
        model_id = model.id
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Outdoor collection",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "dependencies": {
                "model_install_ids": [model_id, model_id],
                "models": [
                    model_id,
                    {"id": model_id},
                    {"name": "Granite checkpoint"},
                    {"path": "managed/granite"},
                ],
            },
        },
    )
    assert response.status_code == 201, response.json()
    rows = await _listed(client, include_dependencies="true")
    assert rows[response.json()["family_id"]]["dependency_summary"] == {
        "dependency_count": 1,
        "names": ["Granite checkpoint"],
    }


async def test_real_workflow_summary_ignores_malformed_labels_and_locator_credentials(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Outdoor collection",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "dependencies": {
                "models": [
                    True,
                    [],
                    {"name": ""},
                    {"path": "https://[invalid"},
                    {"path": "https://user:pass@example.org/model.safetensors?token=omit"},
                ],
                "unrelated_metadata": "Not a dependency name",
            },
        },
    )
    assert response.status_code == 201, response.json()
    rows = await _listed(client, include_dependencies="true")
    assert rows[response.json()["family_id"]]["dependency_summary"] == {
        "dependency_count": 2,
        "names": ["model.safetensors"],
    }


@pytest.mark.parametrize("local_ids", [False, True])
async def test_real_workflow_component_names_use_folder_and_hash(
    client: AsyncClient,
    local_ids: bool,
) -> None:
    from local_lm.models import ModelComponentManifest

    with SessionLocal() as session:
        model = ModelInstall(
            name="Granite checkpoint", role="image", engine="mock", local_path="managed/granite"
        )
        unrelated = ModelInstall(
            name="Unrelated checkpoint", role="image", engine="mock", local_path="managed/unrelated"
        )
        session.add_all([model, unrelated])
        session.flush()
        session.add_all(
            [
                ModelComponentManifest(
                    model_install_id=model.id,
                    kind="checkpoint",
                    relative_path="granite.safetensors",
                    target_folder="checkpoints",
                    sha256="a" * 64,
                ),
                ModelComponentManifest(
                    model_install_id=unrelated.id,
                    kind="vae",
                    relative_path="other.safetensors",
                    target_folder="vae",
                    sha256="a" * 64,
                ),
                ModelComponentManifest(
                    model_install_id=unrelated.id,
                    kind="checkpoint",
                    relative_path="different.safetensors",
                    target_folder="checkpoints",
                    sha256="b" * 64,
                ),
            ]
        )
        session.commit()
        model_id = model.id
    dependencies: dict[str, Any] = {
        "model_components": [
            {"target_folder": "checkpoints", "sha256": "A" * 64},
            {"target_folder": "checkpoints", "sha256": "a" * 64},
        ]
    }
    if local_ids:
        dependencies["model_install_ids"] = [model_id]
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Outdoor collection",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "dependencies": dependencies,
        },
    )
    assert created.status_code == 201, created.json()
    rows = await _listed(client, include_dependencies="true")
    assert rows[created.json()["family_id"]]["dependency_summary"] == {
        "dependency_count": 1,
        "names": ["Granite checkpoint"],
    }


async def test_unresolved_components_count_without_exposing_hashes(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Outdoor collection",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "dependencies": {
                "model_components": [
                    {"target_folder": "checkpoints", "sha256": "a" * 64},
                    {"target_folder": "vae", "sha256": "b" * 64},
                    {"target_folder": "", "sha256": "c" * 64},
                    {"target_folder": "checkpoints", "sha256": "invalid"},
                ]
            },
        },
    )
    assert created.status_code == 201, created.json()
    rows = await _listed(client, include_dependencies="true")
    assert rows[created.json()["family_id"]]["dependency_summary"] == {
        "dependency_count": 2,
        "names": [],
    }
