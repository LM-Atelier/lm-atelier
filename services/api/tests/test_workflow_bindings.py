from __future__ import annotations

import ast
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import local_lm.workflow_bindings as binding_module
from local_lm.db import Base
from local_lm.domain import utcnow
from local_lm.models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    ModelProfile,
)
from local_lm.workflow_bindings import (
    WORKFLOW_DEPENDENCY_MATERIALIZER_ISSUE_CODES,
    MaterializedWorkflowDependency,
    WorkflowBindingError,
    WorkflowBindingSelection,
    materialize_custom_node,
    materialize_model_asset,
    materialize_model_install,
    materialize_model_profile,
    materialize_registry_package,
    materialize_runtime,
    resolve_workflow_activation,
    workflow_resource_identity_sha256,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    WorkflowDependencyResourceKind,
    parse_workflow_dependency_contract,
)

WORKFLOW_BINDINGS_MODULE = Path(binding_module.__file__).resolve()


def _contract() -> WorkflowDependencyContract:
    return parse_workflow_dependency_contract(
        {
            "version": 1,
            "slots": [
                {
                    "name": "primary",
                    "resource_kind": "runtime",
                    "required": True,
                    "satisfaction": "any_of",
                    "requirements": [
                        {"key": "fast", "constraints": {"engine": "fast"}},
                        {"key": "quality", "constraints": {"engine": "quality"}},
                    ],
                },
                {
                    "name": "encoders",
                    "resource_kind": "runtime",
                    "required": True,
                    "satisfaction": "all_of",
                    "requirements": [
                        {"key": "clip", "constraints": {"runtime_build": "clip"}},
                        {"key": "t5", "constraints": {"runtime_build": "t5"}},
                    ],
                },
                {
                    "name": "addons",
                    "resource_kind": "runtime",
                    "required": False,
                    "satisfaction": "all_of",
                    "requirements": [
                        {"key": "first", "constraints": {"runtime_build": "addon-a"}},
                        {"key": "second", "constraints": {"runtime_build": "addon-b"}},
                    ],
                },
            ],
        }
    )


def _runtime(engine: str, build: str) -> MaterializedWorkflowDependency:
    return materialize_runtime(
        engine=engine,
        runtime_build=build,
        adapter_contract_version=1,
        launch_contract_version="v1",
    )


def _materializer(
    values: Mapping[str, MaterializedWorkflowDependency | None],
) -> Callable[
    [WorkflowDependencyRequirement, WorkflowBindingSelection],
    MaterializedWorkflowDependency | None,
]:
    def materialize(
        requirement: WorkflowDependencyRequirement,
        selection: WorkflowBindingSelection,
    ) -> MaterializedWorkflowDependency | None:
        del requirement
        return values.get(selection.local_id)

    return materialize


def _required_selections() -> list[WorkflowBindingSelection]:
    return [
        WorkflowBindingSelection("primary", "quality", "runtime", "local-quality"),
        WorkflowBindingSelection("encoders", "clip", "runtime", "local-clip"),
        WorkflowBindingSelection("encoders", "t5", "runtime", "local-t5"),
    ]


def _required_materializations() -> dict[str, MaterializedWorkflowDependency]:
    return {
        "local-quality": _runtime("quality", "quality-v1"),
        "local-clip": _runtime("encoder", "clip"),
        "local-t5": _runtime("encoder", "t5"),
    }


def _literal_materializer_issue_codes() -> frozenset[str]:
    tree = ast.parse(WORKFLOW_BINDINGS_MODULE.read_text(encoding="utf-8"))
    materializer_start = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "materialize_model_install"
    )
    materializer_end = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_local_locator"
    )
    return frozenset(
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and materializer_start <= call.lineno < materializer_end
        and isinstance(call.func, ast.Name)
        and call.func.id == "WorkflowBindingError"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    )


def test_complete_activation_materializes_all_required_groups() -> None:
    result = resolve_workflow_activation(
        _contract(),
        _required_selections(),
        _materializer(_required_materializations()),
    )

    assert result.complete is True
    assert result.issues == ()
    assert result.missing_required_slots == ()
    assert [(item.slot_name, item.requirement_key) for item in result.bindings] == [
        ("encoders", "clip"),
        ("encoders", "t5"),
        ("primary", "quality"),
    ]
    assert result.binding_sha256 is not None and len(result.binding_sha256) == 64


@pytest.mark.parametrize(
    ("selections", "expected_missing", "expected_issue", "complete"),
    [
        (
            _required_selections()[1:],
            ("primary",),
            "missing_required_dependency",
            False,
        ),
        (
            _required_selections()[:2],
            ("encoders",),
            "missing_required_dependency",
            False,
        ),
        (_required_selections(), (), None, True),
        (
            [
                *_required_selections(),
                WorkflowBindingSelection("addons", "first", "runtime", "local-addon-a"),
            ],
            (),
            "partial_optional_dependency",
            False,
        ),
    ],
)
def test_activation_completeness_reports_missing_or_partial_groups(
    selections: list[WorkflowBindingSelection],
    expected_missing: tuple[str, ...],
    expected_issue: str | None,
    complete: bool,
) -> None:
    values = {
        **_required_materializations(),
        "local-addon-a": _runtime("addon", "addon-a"),
    }
    result = resolve_workflow_activation(_contract(), selections, _materializer(values))

    assert result.complete is complete
    assert result.missing_required_slots == expected_missing
    assert (expected_issue is None) is (not result.issues)
    if expected_issue:
        assert expected_issue in {item.code for item in result.issues}
        assert result.binding_sha256 is None


@pytest.mark.parametrize(
    ("selections", "code"),
    [
        (
            [
                *_required_selections(),
                WorkflowBindingSelection("primary", "fast", "runtime", "local-fast"),
            ],
            "ambiguous_dependency_binding",
        ),
        (
            [WorkflowBindingSelection("missing", "fast", "runtime", "local-fast")],
            "unknown_dependency_slot",
        ),
        (
            [WorkflowBindingSelection("primary", "missing", "runtime", "local-fast")],
            "unknown_dependency_requirement",
        ),
        (
            [
                *_required_selections(),
                WorkflowBindingSelection("primary", "quality", "runtime", "other-local"),
            ],
            "duplicate_dependency_binding",
        ),
    ],
)
def test_binding_selection_must_be_unambiguous_and_declared(
    selections: list[WorkflowBindingSelection], code: str
) -> None:
    with pytest.raises(WorkflowBindingError) as raised:
        resolve_workflow_activation(
            _contract(),
            selections,
            _materializer({**_required_materializations(), "local-fast": _runtime("fast", "v1")}),
        )

    assert raised.value.code == code


@pytest.mark.parametrize(
    ("replacement", "recorded", "issue"),
    [
        (None, None, "dependency_unavailable"),
        (_runtime("wrong", "quality-v1"), None, "dependency_requirement_mismatch"),
        (_runtime("quality", "quality-v1"), "0" * 64, "dependency_binding_drift"),
    ],
)
def test_materialization_drift_and_requirement_mismatch_fail_closed(
    replacement: MaterializedWorkflowDependency | None,
    recorded: str | None,
    issue: str,
) -> None:
    selections = _required_selections()
    selections[0] = WorkflowBindingSelection(
        "primary",
        "quality",
        "runtime",
        "local-quality",
        recorded_resource_identity_sha256=recorded,
    )
    values: dict[str, MaterializedWorkflowDependency | None] = {**_required_materializations()}
    values["local-quality"] = replacement

    result = resolve_workflow_activation(_contract(), selections, _materializer(values))

    assert result.complete is False
    assert issue in {item.code for item in result.issues}
    assert result.binding_sha256 is None


def test_materializer_unavailability_is_a_repairable_activation_issue() -> None:
    def unavailable(
        requirement: WorkflowDependencyRequirement,
        selection: WorkflowBindingSelection,
    ) -> MaterializedWorkflowDependency:
        del requirement, selection
        raise WorkflowBindingError("dependency_unavailable", "not ready")

    result = resolve_workflow_activation(_contract(), _required_selections(), unavailable)

    assert result.complete is False
    assert "dependency_unavailable" in {item.code for item in result.issues}
    assert result.binding_sha256 is None


def test_materializer_issue_vocabulary_covers_every_literal_producer() -> None:
    assert _literal_materializer_issue_codes() == (WORKFLOW_DEPENDENCY_MATERIALIZER_ISSUE_CODES)


def test_all_materializer_refusals_accumulate_across_dependency_slots() -> None:
    codes_by_local_id = {
        "local-quality": "dependency_unavailable",
        "local-clip": "legacy_environment_manifest",
        "local-t5": "invalid_dependency_identity",
    }

    def refuse(
        requirement: WorkflowDependencyRequirement,
        selection: WorkflowBindingSelection,
    ) -> MaterializedWorkflowDependency:
        del requirement
        raise WorkflowBindingError(codes_by_local_id[selection.local_id], "not ready")

    result = resolve_workflow_activation(_contract(), _required_selections(), refuse)

    assert [
        (issue.code, issue.slot_name, issue.requirement_key)
        for issue in result.issues
        if issue.requirement_key is not None
    ] == [
        ("legacy_environment_manifest", "encoders", "clip"),
        ("invalid_dependency_identity", "encoders", "t5"),
        ("dependency_unavailable", "primary", "quality"),
    ]
    assert result.bindings == ()
    assert result.missing_required_slots == ("encoders", "primary")
    assert result.complete is False
    assert result.binding_sha256 is None


def test_nonportable_selection_and_identity_use_binding_error_contract() -> None:
    selections = _required_selections()
    selections[0] = WorkflowBindingSelection(
        "primary", "quality", "runtime", "local-quality", mount={"authToken": "secret"}
    )
    with pytest.raises(WorkflowBindingError) as mount_error:
        resolve_workflow_activation(
            _contract(), selections, _materializer(_required_materializations())
        )
    assert mount_error.value.code == "invalid_dependency_bindings"

    values = _required_materializations()
    values["local-quality"] = MaterializedWorkflowDependency(
        "runtime", {"kind": "runtime", "engine": "quality", "local_path": "C:/private"}
    )
    result = resolve_workflow_activation(_contract(), _required_selections(), _materializer(values))
    assert result.complete is False
    assert "invalid_dependency_identity" in {item.code for item in result.issues}


def test_binding_digest_is_portable_and_order_independent() -> None:
    first = resolve_workflow_activation(
        _contract(),
        _required_selections(),
        _materializer(_required_materializations()),
    )
    second_selections = [
        WorkflowBindingSelection("encoders", "t5", "runtime", "different-t5"),
        WorkflowBindingSelection("primary", "quality", "runtime", "different-quality"),
        WorkflowBindingSelection("encoders", "clip", "runtime", "different-clip"),
    ]
    second_values = {
        "different-quality": _runtime("quality", "quality-v1"),
        "different-clip": _runtime("encoder", "clip"),
        "different-t5": _runtime("encoder", "t5"),
    }
    second = resolve_workflow_activation(
        _contract(), second_selections, _materializer(second_values)
    )

    assert first.binding_sha256 == second.binding_sha256

    changed_values = dict(second_values)
    changed_values["different-t5"] = _runtime("encoder", "t5-v2")
    changed = resolve_workflow_activation(
        _contract(), second_selections, _materializer(changed_values)
    )
    assert changed.binding_sha256 is None
    assert "dependency_requirement_mismatch" in {item.code for item in changed.issues}


def test_digest_helpers_reject_incompatible_or_incomplete_snapshots() -> None:
    with pytest.raises(WorkflowBindingError) as identity_error:
        workflow_resource_identity_sha256("runtime", {"kind": "model_asset"})
    assert identity_error.value.code == "dependency_identity_kind_mismatch"

    with pytest.raises(WorkflowBindingError) as unknown_kind:
        workflow_resource_identity_sha256(
            cast(WorkflowDependencyResourceKind, "made_up"), {"kind": "made_up"}
        )
    assert unknown_kind.value.code == "dependency_identity_kind_mismatch"

    with pytest.raises(WorkflowBindingError) as activation_error:
        binding_module._workflow_activation_binding_sha256(_contract(), [])
    assert activation_error.value.code == "incomplete_dependency_bindings"


def test_resource_identity_digest_canonicalizes_unordered_collections() -> None:
    first = workflow_resource_identity_sha256(
        "registry_package",
        {"kind": "registry_package", "node_types": ["First", "Second"]},
    )
    second = workflow_resource_identity_sha256(
        "registry_package",
        {"kind": "registry_package", "node_types": ["Second", "First"]},
    )

    assert first == second


def test_ordered_constraint_arrays_do_not_match_a_different_order() -> None:
    contract = parse_workflow_dependency_contract(
        {
            "version": 1,
            "slots": [
                {
                    "name": "primary",
                    "resource_kind": "runtime",
                    "required": True,
                    "satisfaction": "any_of",
                    "requirements": [{"key": "default", "constraints": {"values": ["a", "b"]}}],
                }
            ],
        }
    )
    selection = WorkflowBindingSelection("primary", "default", "runtime", "local")
    reversed_identity = MaterializedWorkflowDependency(
        "runtime", {"kind": "runtime", "values": ["b", "a"]}
    )

    result = resolve_workflow_activation(
        contract, [selection], _materializer({"local": reversed_identity})
    )

    assert result.complete is False
    assert "dependency_requirement_mismatch" in {item.code for item in result.issues}


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _profile(
    session: Session,
    *,
    install_id: str,
    profile_id: str,
    local_path: str,
    relative_path: str,
    load_settings: dict[str, object] | None = None,
) -> ModelProfile:
    install = ModelInstall(
        id=install_id,
        name=f"Display {install_id}",
        role="chat",
        engine="llama.cpp",
        local_path=local_path,
        active=True,
    )
    profile = ModelProfile(
        id=profile_id,
        model_install_id=install_id,
        name=f"Profile {profile_id}",
        role="chat",
        engine="llama.cpp",
        load_settings_json=load_settings or {"context_length": 4096},
        request_settings_json={"temperature": 0.7},
    )
    session.add(install)
    session.flush()
    session.add_all(
        [
            profile,
            ModelComponentManifest(
                id=f"component_{install_id}",
                model_install_id=install_id,
                kind="gguf_model",
                relative_path=relative_path,
                target_folder="models",
                sha256="a" * 64,
                required=True,
            ),
        ]
    )
    session.flush()
    return profile


def test_content_identical_profiles_at_different_paths_have_one_identity(
    session: Session,
) -> None:
    first = _profile(
        session,
        install_id="model_first",
        profile_id="profile_first",
        local_path="C:/private/models/first",
        relative_path="weights/model.gguf",
    )
    second = _profile(
        session,
        install_id="model_second",
        profile_id="profile_second",
        local_path="D:/different/location",
        relative_path="weights/model.gguf",
    )

    first_identity = materialize_model_profile(session, first)
    second_identity = materialize_model_profile(session, second)

    assert first_identity.identity == second_identity.identity
    assert workflow_resource_identity_sha256(
        first_identity.resource_kind, first_identity.identity
    ) == workflow_resource_identity_sha256(second_identity.resource_kind, second_identity.identity)
    assert "temperature" not in str(first_identity.identity)
    assert "private" not in str(first_identity.identity)

    second.load_settings_json = {"context_length": 8192}
    changed = materialize_model_profile(session, second)
    assert workflow_resource_identity_sha256(
        changed.resource_kind, changed.identity
    ) != workflow_resource_identity_sha256(first_identity.resource_kind, first_identity.identity)


def test_component_closure_rejects_case_collisions_and_missing_hashes(session: Session) -> None:
    profile = _profile(
        session,
        install_id="model_collision",
        profile_id="profile_collision",
        local_path="C:/models/collision",
        relative_path="weights/model.gguf",
    )
    session.add(
        ModelComponentManifest(
            id="component_collision_second",
            model_install_id="model_collision",
            kind="gguf_model",
            relative_path="WEIGHTS/MODEL.GGUF",
            target_folder="models",
            sha256="b" * 64,
            required=True,
        )
    )
    session.flush()

    with pytest.raises(WorkflowBindingError) as collision:
        materialize_model_profile(session, profile)
    assert collision.value.code == "invalid_dependency_identity"

    second = _profile(
        session,
        install_id="model_unhashed",
        profile_id="profile_unhashed",
        local_path="C:/models/unhashed",
        relative_path="model.gguf",
    )
    component = session.get(ModelComponentManifest, "component_model_unhashed")
    assert component is not None
    component.sha256 = None

    with pytest.raises(WorkflowBindingError) as unhashed:
        materialize_model_profile(session, second)
    assert unhashed.value.code == "invalid_dependency_identity"


def test_comfy_component_identity_uses_loader_visible_runtime_references(
    session: Session,
) -> None:
    def install(suffix: str, *, relative_path: str, runtime_root: str) -> ModelInstall:
        value = ModelInstall(
            id=f"model_comfy_{suffix}",
            name=f"Comfy {suffix}",
            role="image",
            engine="comfyui",
            local_path=f"C:/models/{suffix}",
            manifest_json={"comfy_paths": {"diffusion_models": runtime_root}},
            active=True,
        )
        session.add(value)
        session.flush()
        session.add(
            ModelComponentManifest(
                id=f"component_comfy_{suffix}",
                model_install_id=value.id,
                kind="diffusion_model",
                relative_path=relative_path,
                target_folder="diffusion_models",
                sha256="a" * 64,
                required=True,
            )
        )
        session.flush()
        return value

    nested = install(
        "nested",
        relative_path="split_files/diffusion_models/z_image_turbo_int8.safetensors",
        runtime_root="./split_files/diffusion_models",
    )
    root = install(
        "root",
        relative_path="z_image_turbo_int8.safetensors",
        runtime_root=".",
    )
    wider = install(
        "wider",
        relative_path="split_files/diffusion_models/z_image_turbo_int8.safetensors",
        runtime_root="split_files",
    )

    nested_identity = materialize_model_install(session, nested)
    root_identity = materialize_model_install(session, root)
    wider_identity = materialize_model_install(session, wider)

    assert nested_identity.identity == root_identity.identity
    assert nested_identity.identity["components"][0]["runtime_reference"] == (
        "z_image_turbo_int8.safetensors"
    )
    assert nested_identity.identity != wider_identity.identity

    nested.manifest_json = {"comfy_paths": {"diffusion_models": "other"}}
    with pytest.raises(WorkflowBindingError) as raised:
        materialize_model_install(session, nested)
    assert raised.value.code == "invalid_dependency_identity"


@pytest.mark.parametrize(
    "runtime_reference",
    ["weights\\\\model.gguf", "./weights/model.gguf", "weights//model.gguf"],
)
def test_component_closure_rejects_noncanonical_runtime_references(
    session: Session, runtime_reference: str
) -> None:
    profile = _profile(
        session,
        install_id="model_noncanonical",
        profile_id="profile_noncanonical",
        local_path="C:/models/noncanonical",
        relative_path=runtime_reference,
    )

    with pytest.raises(WorkflowBindingError) as raised:
        materialize_model_profile(session, profile)
    assert raised.value.code == "invalid_dependency_identity"


def test_model_asset_identity_includes_the_runtime_loader_name() -> None:
    first = ModelAssetInstall(
        name="First",
        kind="lora",
        local_path="C:/models/first.safetensors",
        manifest_json={"sha256": "a" * 64, "comfy_name": "styles/first.safetensors"},
        active=True,
        verified_at=utcnow(),
    )
    second = ModelAssetInstall(
        name="Second",
        kind="lora",
        local_path="D:/models/second.safetensors",
        manifest_json={"sha256": "a" * 64, "comfy_name": "styles/second.safetensors"},
        active=True,
        verified_at=utcnow(),
    )

    first_identity = materialize_model_asset(first)
    second_identity = materialize_model_asset(second)

    assert first_identity.identity["runtime_reference"] == "styles/first.safetensors"
    assert workflow_resource_identity_sha256(
        first_identity.resource_kind, first_identity.identity
    ) != workflow_resource_identity_sha256(second_identity.resource_kind, second_identity.identity)

    first.kind = "unsupported"
    with pytest.raises(WorkflowBindingError) as raised:
        materialize_model_asset(first)
    assert raised.value.code == "dependency_unavailable"


def test_runtime_identity_accepts_bounded_printable_build_descriptions() -> None:
    runtime = materialize_runtime(
        engine="comfyui",
        runtime_build="ComfyUI 0.3.50 (CUDA)",
        adapter_contract_version=1,
        launch_contract_version="v1",
    )

    assert runtime.identity["runtime_build"] == "ComfyUI 0.3.50 (CUDA)"


def _registry_install() -> ComfyRegistryInstall:
    closure = "c" * 64
    return ComfyRegistryInstall(
        package_id="example-node",
        package_version="1.2.3",
        registry_record_id="example-node@1.2.3",
        repository_url="https://github.com/example/node",
        download_url="https://example.invalid/node.zip",
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        installed_path="lm-atelier-registry_example-node",
        node_types_json=["Power Lora Loader", "ExampleNode"],
        pip_dependencies_json=[],
        review_json={},
        wheel_closure_sha256=closure,
        wheel_environment_sha256="d" * 64,
        wheel_environment_path=f"registry-wheels-v3-{closure}",
        trusted=True,
        active=True,
    )


def test_registry_identity_accepts_semver_and_requires_the_managed_environment() -> None:
    install = _registry_install()

    identity = materialize_registry_package(install)

    assert identity.identity["package_version"] == "1.2.3"
    assert identity.identity["node_types"] == ["ExampleNode", "Power Lora Loader"]
    assert "wheel_environment_path" not in identity.identity

    install.wheel_environment_path = None
    with pytest.raises(WorkflowBindingError) as raised:
        materialize_registry_package(install)
    assert raised.value.code == "dependency_unavailable"

    install.wheel_environment_path = f"registry-wheels-{'c' * 64}"
    with pytest.raises(WorkflowBindingError) as legacy:
        materialize_registry_package(install)
    assert legacy.value.code == "legacy_environment_manifest"


@pytest.mark.parametrize("node_types", [[], ["ExampleNode", "ExampleNode"]])
def test_registry_identity_rejects_invalid_node_type_closures(node_types: list[str]) -> None:
    install = _registry_install()
    install.node_types_json = node_types

    with pytest.raises(WorkflowBindingError) as raised:
        materialize_registry_package(install)
    assert raised.value.code == "invalid_dependency_identity"


def test_custom_node_identity_is_canonical_and_content_addressed() -> None:
    install = CustomNodeInstall(
        name="Example",
        source_url="https://github.com/Example/Node",
        revision="A" * 40,
        installed_path="C:/private/custom-node",
        tree_hash="B" * 40,
        trusted=True,
        active=True,
    )

    identity = materialize_custom_node(install)

    assert identity.identity == {
        "kind": "custom_node",
        "source_url": "https://github.com/example/node.git",
        "revision": "a" * 40,
        "tree_hash": "b" * 40,
    }
    assert "private" not in str(identity.identity)
