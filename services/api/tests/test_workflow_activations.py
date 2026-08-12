from __future__ import annotations

import hashlib
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import utcnow
from local_lm.models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    ModelProfile,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
)
from local_lm.workflow_activations import (
    WorkflowActivationError,
    activate_workflow_revision,
    revalidate_workflow_activation,
)
from local_lm.workflow_bindings import (
    MaterializedWorkflowDependency,
    WorkflowBindingSelection,
    materialize_runtime,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _contract(*slots: dict[str, object]) -> WorkflowDependencyContract:
    return parse_workflow_dependency_contract({"version": 1, "slots": list(slots)})


def _slot(
    name: str,
    resource_kind: str,
    *,
    required: bool = True,
    satisfaction: str = "all_of",
    requirements: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "resource_kind": resource_kind,
        "required": required,
        "satisfaction": satisfaction,
        "requirements": requirements or [{"key": "default", "constraints": {}}],
    }


def _revision(
    session: Session,
    contract: WorkflowDependencyContract | None,
    *,
    suffix: str = "one",
) -> WorkflowRevision:
    definition = WorkflowDefinition(
        id=f"workflow_{suffix}",
        name=f"Workflow {suffix}",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        id=f"wfrev_{suffix}",
        workflow_id=definition.id,
        version=1,
        dependency_contract_sha256=(
            workflow_dependency_contract_sha256(contract) if contract is not None else None
        ),
    )
    session.add_all([definition, revision])
    session.flush()
    if contract is not None:
        for ordinal, slot in enumerate(contract.slots):
            session.add(
                WorkflowDependencySlot(
                    id=f"wfslot_{suffix}_{slot.name}",
                    workflow_revision_id=revision.id,
                    name=slot.name,
                    resource_kind=slot.resource_kind,
                    required=slot.required,
                    satisfaction=slot.satisfaction,
                    requirements_json=[
                        {"key": item.key, "constraints": item.constraints}
                        for item in slot.requirements
                    ],
                    contract_sha256=workflow_dependency_slot_sha256(slot),
                    ordinal=ordinal,
                )
            )
        session.flush()
    return revision


def _write(root: Path, relative_path: str, content: bytes) -> str:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _model(
    session: Session,
    root: Path,
    *,
    suffix: str,
    content: bytes = b"model",
    target_folder: str = "checkpoints",
    relative_path: str = "weights/model.safetensors",
    runtime_root: str = "weights",
) -> ModelInstall:
    digest = _write(root, relative_path, content)
    install = ModelInstall(
        id=f"model_{suffix}",
        name=f"Model {suffix}",
        role="image",
        engine="comfyui",
        local_path=str(root),
        manifest_json={"comfy_paths": {target_folder: runtime_root}},
        active=True,
    )
    session.add(install)
    session.flush()
    session.add(
        ModelComponentManifest(
            id=f"component_{suffix}",
            model_install_id=install.id,
            kind="checkpoint",
            relative_path=relative_path,
            target_folder=target_folder,
            sha256=digest,
            size_bytes=len(content),
            required=True,
        )
    )
    session.flush()
    return install


def _profile(session: Session, install: ModelInstall, *, suffix: str) -> ModelProfile:
    profile = ModelProfile(
        id=f"profile_{suffix}",
        model_install_id=install.id,
        name=f"Profile {suffix}",
        role=install.role,
        engine=install.engine,
        load_settings_json={"precision": "auto"},
    )
    session.add(profile)
    session.flush()
    return profile


def _asset(
    session: Session,
    root: Path,
    *,
    suffix: str,
    content: bytes = b"asset",
    runtime_reference: str = "styles/style.safetensors",
) -> ModelAssetInstall:
    digest = _write(root, runtime_reference, content)
    asset = ModelAssetInstall(
        id=f"asset_{suffix}",
        name=f"Asset {suffix}",
        kind="lora",
        local_path=str(root),
        manifest_json={"sha256": digest, "comfy_name": runtime_reference},
        active=True,
        verified_at=utcnow(),
    )
    session.add(asset)
    session.flush()
    return asset


def _runtime(
    _requirement: WorkflowDependencyRequirement,
    selection: WorkflowBindingSelection,
) -> MaterializedWorkflowDependency:
    return materialize_runtime(
        engine="comfyui",
        runtime_build=f"ComfyUI {selection.local_id}",
        adapter_contract_version=1,
        launch_contract_version="v1",
    )


def test_activation_persists_reuses_and_returns_exact_immutable_scope(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(
        _slot("asset", "model_asset"),
        _slot("profile", "model_profile"),
        _slot("runtime", "runtime"),
    )
    revision = _revision(session, contract)
    install = _model(session, tmp_path / "model", suffix="primary")
    profile = _profile(session, install, suffix="primary")
    asset = _asset(session, tmp_path / "asset", suffix="style")
    selections = [
        WorkflowBindingSelection("runtime", "default", "runtime", "build-1"),
        WorkflowBindingSelection("asset", "default", "model_asset", asset.id),
        WorkflowBindingSelection("profile", "default", "model_profile", profile.id),
    ]

    first = activate_workflow_revision(session, revision, selections, runtime_materializer=_runtime)
    second = activate_workflow_revision(
        session, revision.id, list(reversed(selections)), runtime_materializer=_runtime
    )

    assert second.activation_id == first.activation_id
    assert second.binding_sha256 == first.binding_sha256
    assert second.launch_sha256 == first.launch_sha256
    assert first.model_install_ids == (install.id,)
    assert first.model_asset_install_ids == (asset.id,)
    assert first.custom_node_install_ids == ()
    assert first.registry_install_ids == ()
    assert first.runtime_keys == ("build-1",)
    assert first.models[0].base_path == (tmp_path / "model").resolve()
    assert first.models[0].comfy_paths == (("checkpoints", "weights"),)
    assert first.assets[0].runtime_reference == "styles/style.safetensors"
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 1
    assert session.scalar(select(func.count(WorkflowDependencyBinding.id))) == 3
    row = session.get(WorkflowActivation, first.activation_id)
    assert row is not None
    assert row.is_active is True
    assert row.state == "ready"
    assert row.details_json == {"launch_sha256": first.launch_sha256}


def test_legacy_revision_is_explicitly_unsupported(session: Session) -> None:
    revision = _revision(session, None)

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(session, revision, [])

    assert raised.value.code == "legacy_workflow_revision_unsupported"
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 0


@pytest.mark.parametrize("target", ["revision", "slot"])
def test_hydration_rejects_contract_and_slot_digest_drift(session: Session, target: str) -> None:
    contract = _contract(_slot("runtime", "runtime"))
    revision = _revision(session, contract)
    if target == "revision":
        revision.dependency_contract_sha256 = "f" * 64
    else:
        slot = session.scalar(select(WorkflowDependencySlot))
        assert slot is not None
        slot.contract_sha256 = "f" * 64
    session.flush()

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [WorkflowBindingSelection("runtime", "default", "runtime", "build-1")],
            runtime_materializer=_runtime,
        )

    assert raised.value.code == "workflow_contract_drift"


def test_missing_named_required_selection_is_repairable_and_not_persisted(
    session: Session,
) -> None:
    contract = _contract(_slot("runtime", "runtime"))
    revision = _revision(session, contract)

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(session, revision, [], runtime_materializer=_runtime)

    assert raised.value.code == "workflow_activation_incomplete"
    assert {issue.code for issue in raised.value.issues} == {"missing_required_dependency"}
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 0


@pytest.mark.parametrize("resource", ["model", "asset"])
def test_revalidation_rehashes_bytes_and_marks_drifted_activation_stale(
    session: Session, tmp_path: Path, resource: str
) -> None:
    kind = "model_install" if resource == "model" else "model_asset"
    contract = _contract(_slot("primary", kind))
    revision = _revision(session, contract)
    if resource == "model":
        install = _model(session, tmp_path / "model", suffix="drift")
        local_id = install.id
        changed_path = tmp_path / "model" / "weights" / "model.safetensors"
    else:
        asset = _asset(session, tmp_path / "asset", suffix="drift")
        local_id = asset.id
        changed_path = tmp_path / "asset" / "styles" / "style.safetensors"
    scope = activate_workflow_revision(
        session,
        revision,
        [WorkflowBindingSelection("primary", "default", kind, local_id)],
    )
    session.commit()
    changed_path.write_bytes(b"changed after activation")

    with pytest.raises(WorkflowActivationError) as raised:
        revalidate_workflow_activation(session, scope.activation_id)

    assert raised.value.code == "dependency_content_drift"
    stale = session.get(WorkflowActivation, scope.activation_id)
    assert stale is not None
    assert stale.state == "stale"
    assert stale.is_active is False
    assert stale.invalidation_code == "dependency_content_drift"
    assert stale.invalidated_at is not None


def test_failed_new_selection_does_not_deactivate_current_snapshot(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(_slot("primary", "model_install"))
    revision = _revision(session, contract)
    first = _model(session, tmp_path / "first", suffix="first", content=b"first")
    second = _model(session, tmp_path / "second", suffix="second", content=b"second")
    first_scope = activate_workflow_revision(
        session,
        revision,
        [WorkflowBindingSelection("primary", "default", "model_install", first.id)],
    )
    (tmp_path / "second" / "weights" / "model.safetensors").write_bytes(b"drift")

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [WorkflowBindingSelection("primary", "default", "model_install", second.id)],
        )

    assert raised.value.code == "dependency_content_drift"
    current = session.get(WorkflowActivation, first_scope.activation_id)
    assert current is not None
    assert current.is_active is True
    assert current.state == "ready"
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 1


def test_successful_new_snapshot_replaces_active_and_revalidates_from_stored_locators(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(_slot("primary", "model_install"))
    revision = _revision(session, contract)
    first = _model(session, tmp_path / "first", suffix="replace-first", content=b"first")
    second = _model(session, tmp_path / "second", suffix="replace-second", content=b"second")
    first_scope = activate_workflow_revision(
        session,
        revision,
        [WorkflowBindingSelection("primary", "default", "model_install", first.id)],
    )
    second_scope = activate_workflow_revision(
        session,
        revision,
        [WorkflowBindingSelection("primary", "default", "model_install", second.id)],
    )

    first_row = session.get(WorkflowActivation, first_scope.activation_id)
    second_row = session.get(WorkflowActivation, second_scope.activation_id)
    assert first_row is not None and first_row.is_active is False
    assert second_row is not None and second_row.is_active is True
    restored = revalidate_workflow_activation(session, second_scope.activation_id)
    assert restored == second_scope
    assert restored.model_install_ids == (second.id,)


def test_reuse_rejects_tampered_persisted_binding_and_marks_snapshot_stale(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(_slot("primary", "model_install"))
    revision = _revision(session, contract)
    install = _model(session, tmp_path / "model", suffix="tamper")
    selections = [WorkflowBindingSelection("primary", "default", "model_install", install.id)]
    scope = activate_workflow_revision(session, revision, selections)
    binding = session.scalar(
        select(WorkflowDependencyBinding).where(
            WorkflowDependencyBinding.workflow_activation_id == scope.activation_id
        )
    )
    assert binding is not None
    binding.mount_json = {"loader": "tampered"}
    session.flush()

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(session, revision, selections)

    assert raised.value.code == "invalid_activation_snapshot"
    row = session.get(WorkflowActivation, scope.activation_id)
    assert row is not None
    assert row.state == "stale"
    assert row.is_active is False


def test_model_and_asset_loader_names_collide_case_insensitively(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(
        _slot("base", "model_install"),
        _slot("style", "model_asset"),
    )
    revision = _revision(session, contract)
    install = _model(
        session,
        tmp_path / "model",
        suffix="collision",
        target_folder="loras",
        relative_path="weights/STYLE.safetensors",
        runtime_root="weights",
    )
    asset = _asset(
        session,
        tmp_path / "asset",
        suffix="collision",
        runtime_reference="style.SAFETENSORS",
    )

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [
                WorkflowBindingSelection("base", "default", "model_install", install.id),
                WorkflowBindingSelection("style", "default", "model_asset", asset.id),
            ],
        )

    assert raised.value.code == "runtime_loader_collision"
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 0


def _custom_node(
    session: Session,
    root: Path,
    *,
    suffix: str,
    security_json: dict[str, object],
) -> CustomNodeInstall:
    folder = root / f"lm-atelier-node_{suffix}"
    folder.mkdir(parents=True)
    install = CustomNodeInstall(
        id=f"node_{suffix}",
        name=f"Node {suffix}",
        source_url=f"https://github.com/example/{suffix}",
        revision="a" * 40,
        installed_path=folder.name,
        tree_hash="b" * 40,
        trusted=True,
        active=True,
        security_json=security_json,
    )
    session.add(install)
    session.flush()
    return install


def _registry(
    session: Session,
    custom_root: Path,
    environment_root: Path,
    *,
    suffix: str,
    node_types: list[str],
) -> ComfyRegistryInstall:
    closure = hashlib.sha256(f"closure-{suffix}".encode()).hexdigest()
    folder = custom_root / f"lm-atelier-registry_{suffix}"
    folder.mkdir(parents=True)
    environment_name = f"registry-wheels-v3-{closure}"
    (environment_root / environment_name / "site-packages").mkdir(parents=True)
    install = ComfyRegistryInstall(
        id=f"registry_{suffix}",
        package_id=f"example-{suffix}",
        package_version="1.2.3",
        registry_record_id=f"example-{suffix}@1.2.3",
        repository_url=f"https://github.com/example/{suffix}",
        download_url=f"https://example.invalid/{suffix}.zip",
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        installed_path=folder.name,
        node_types_json=node_types,
        pip_dependencies_json=[],
        review_json={},
        wheel_closure_sha256=closure,
        wheel_environment_sha256="d" * 64,
        wheel_environment_path=environment_name,
        trusted=True,
        active=True,
    )
    session.add(install)
    session.flush()
    return install


def test_arbitrary_git_node_without_bounded_node_evidence_is_unavailable(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(_slot("node", "custom_node"))
    revision = _revision(session, contract)
    node = _custom_node(
        session,
        tmp_path / "nodes",
        suffix="missing-evidence",
        security_json={},
    )

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [WorkflowBindingSelection("node", "default", "custom_node", node.id)],
            custom_node_root=tmp_path / "nodes",
        )

    assert raised.value.code == "workflow_activation_incomplete"
    assert {issue.code for issue in raised.value.issues} == {
        "dependency_unavailable",
        "missing_required_dependency",
    }


def test_registry_and_custom_node_types_collide_case_insensitively(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(
        _slot("custom", "custom_node"),
        _slot("registry", "registry_package"),
    )
    revision = _revision(session, contract)
    custom_root = tmp_path / "nodes"
    custom_root.mkdir()
    environment_root = tmp_path / "environments"
    environment_root.mkdir()
    node = _custom_node(
        session,
        custom_root,
        suffix="custom",
        security_json={"node_types": ["SharedNode"]},
    )
    registry = _registry(
        session,
        custom_root,
        environment_root,
        suffix="registry",
        node_types=["sharednode"],
    )

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [
                WorkflowBindingSelection("custom", "default", "custom_node", node.id),
                WorkflowBindingSelection("registry", "default", "registry_package", registry.id),
            ],
            custom_node_root=custom_root,
            registry_environment_root=environment_root,
        )

    assert raised.value.code == "node_type_collision"
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 0


def test_custom_node_install_folders_cannot_alias(session: Session, tmp_path: Path) -> None:
    contract = _contract(
        _slot("first", "custom_node"),
        _slot("second", "custom_node"),
    )
    revision = _revision(session, contract)
    custom_root = tmp_path / "nodes"
    custom_root.mkdir()
    first = _custom_node(
        session,
        custom_root,
        suffix="folder-first",
        security_json={"node_types": ["FirstNode"]},
    )
    second = _custom_node(
        session,
        custom_root,
        suffix="folder-second",
        security_json={"node_types": ["SecondNode"]},
    )
    second.installed_path = first.installed_path
    session.flush()

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [
                WorkflowBindingSelection("first", "default", "custom_node", first.id),
                WorkflowBindingSelection("second", "default", "custom_node", second.id),
            ],
            custom_node_root=custom_root,
        )

    assert raised.value.code == "extension_folder_collision"


def test_one_runtime_key_cannot_resolve_to_two_identities(session: Session) -> None:
    contract = _contract(
        _slot(
            "first",
            "runtime",
            requirements=[{"key": "default", "constraints": {"runtime_build": "ComfyUI first"}}],
        ),
        _slot(
            "second",
            "runtime",
            requirements=[{"key": "default", "constraints": {"runtime_build": "ComfyUI second"}}],
        ),
    )
    revision = _revision(session, contract)

    def requirement_runtime(
        requirement: WorkflowDependencyRequirement,
        _selection: WorkflowBindingSelection,
    ) -> MaterializedWorkflowDependency:
        return materialize_runtime(
            engine="comfyui",
            runtime_build=str(requirement.constraints["runtime_build"]),
            adapter_contract_version=1,
            launch_contract_version="v1",
        )

    with pytest.raises(WorkflowActivationError) as raised:
        activate_workflow_revision(
            session,
            revision,
            [
                WorkflowBindingSelection("first", "default", "runtime", "shared"),
                WorkflowBindingSelection("second", "default", "runtime", "shared"),
            ],
            runtime_materializer=requirement_runtime,
        )

    assert raised.value.code == "runtime_identity_collision"


def test_selected_extensions_are_the_only_paths_in_launch_scope(
    session: Session, tmp_path: Path
) -> None:
    contract = _contract(
        _slot("custom", "custom_node"),
        _slot("registry", "registry_package"),
    )
    revision = _revision(session, contract)
    custom_root = tmp_path / "nodes"
    custom_root.mkdir()
    environment_root = tmp_path / "environments"
    environment_root.mkdir()
    selected_node = _custom_node(
        session,
        custom_root,
        suffix="selected",
        security_json={"node_types": ["SelectedNode"]},
    )
    _custom_node(
        session,
        custom_root,
        suffix="unselected",
        security_json={"node_types": ["UnselectedNode"]},
    )
    selected_registry = _registry(
        session,
        custom_root,
        environment_root,
        suffix="selected",
        node_types=["RegistryNode"],
    )
    _registry(
        session,
        custom_root,
        environment_root,
        suffix="unselected",
        node_types=["OtherRegistryNode"],
    )

    scope = activate_workflow_revision(
        session,
        revision,
        [
            WorkflowBindingSelection("custom", "default", "custom_node", selected_node.id),
            WorkflowBindingSelection(
                "registry", "default", "registry_package", selected_registry.id
            ),
        ],
        custom_node_root=custom_root,
        registry_environment_root=environment_root,
    )

    assert scope.custom_node_install_ids == (selected_node.id,)
    assert scope.registry_install_ids == (selected_registry.id,)
    assert scope.custom_nodes[0].installed_path.name == selected_node.installed_path
    assert scope.custom_nodes[0].node_types == ("SelectedNode",)
    assert scope.registry_packages[0].installed_path.name == selected_registry.installed_path
    assert scope.registry_packages[0].site_packages.name == "site-packages"
    assert scope.registry_packages[0].node_types == ("RegistryNode",)
