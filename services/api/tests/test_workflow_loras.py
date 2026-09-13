from __future__ import annotations

import json
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Literal

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.comfy_package_widgets import POWER_LORA_LOADER
from local_lm.db import Base, SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
)
from local_lm.schemas import WorkflowLoraControlsOut
from local_lm.workflow_bindings import (
    ResolvedWorkflowBinding,
    workflow_activation_binding_sha256,
    workflow_resource_identity_sha256,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    WorkflowDependencyResourceKind,
    WorkflowDependencySlotContract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_payload,
    workflow_dependency_slot_sha256,
)
from local_lm.workflow_loras import workflow_lora_controls

PRIVATE_LOADER_ID = "node-that-must-never-be-public"
REFERENCE = "styles/detail.safetensors"
ASSET_DIGEST = "a" * 64
RGTHREE_REVISION = "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"
CORE_PROVIDER_REGISTRY_ID = "registry_lora_core_collision"
OPAQUE_CUSTOM_NODE_ID = "node_lora_opaque_provider"
CoreExtension = Literal["advertised_registry", "opaque_custom", "forged_custom_types"]


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _slot(
    name: str,
    resource_kind: WorkflowDependencyResourceKind,
    *,
    required: bool = True,
) -> WorkflowDependencySlotContract:
    return WorkflowDependencySlotContract(
        name=name,
        resource_kind=resource_kind,
        required=required,
        satisfaction="any_of",
        requirements=(WorkflowDependencyRequirement("default", {}),),
    )


def _core_api_graph() -> dict[str, Any]:
    return {
        "source": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base.safetensors"},
            "_meta": {"title": "Base"},
        },
        PRIVATE_LOADER_ID: {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["source", 0],
                "clip": ["source", 1],
                "lora_name": REFERENCE,
                "strength_model": 0.75,
                "strength_clip": 0.5,
            },
            "_meta": {"title": "Private embedded style"},
        },
    }


def _rgthree_api_graph() -> dict[str, Any]:
    return {
        "source": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base.safetensors"},
            "_meta": {"title": "Base"},
        },
        PRIVATE_LOADER_ID: {
            "class_type": POWER_LORA_LOADER,
            "inputs": {
                "model": ["source", 0],
                "clip": ["source", 1],
                "lora_1": {
                    "on": True,
                    "lora": REFERENCE,
                    "strength": 0.8,
                    "strengthTwo": 0.6,
                },
            },
            "_meta": {"title": "Private power loader"},
        },
    }


def _ui_graph(*, rgthree: bool, mismatched: bool = False) -> dict[str, Any]:
    if rgthree:
        widgets: list[Any] = [
            {},
            {"type": "PowerLoraLoaderHeaderWidget"},
            {
                "on": True,
                "lora": REFERENCE,
                "strength": 0.1 if mismatched else 0.8,
                "strengthTwo": 0.6,
            },
            {},
            "",
        ]
        loader_type = POWER_LORA_LOADER
        properties = {
            "cnr_id": "rgthree-comfy",
            "aux_id": "rgthree/rgthree-comfy",
            "ver": RGTHREE_REVISION,
        }
    else:
        widgets = [REFERENCE, 0.1 if mismatched else 0.75, 0.5]
        loader_type = "LoraLoader"
        properties = {"cnr_id": "comfy-core", "ver": "0.28.0"}
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": "source",
                "type": "CheckpointLoaderSimple",
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [1]},
                    {"name": "CLIP", "type": "CLIP", "links": [2]},
                ],
                "widgets_values": ["base.safetensors"],
                "properties": {"cnr_id": "comfy-core"},
            },
            {
                "id": PRIVATE_LOADER_ID,
                "type": loader_type,
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 1},
                    {"name": "clip", "type": "CLIP", "link": 2},
                ],
                "outputs": [],
                "widgets_values": widgets,
                "properties": properties,
            },
        ],
        "links": [
            [1, "source", 0, PRIVATE_LOADER_ID, 0, "MODEL"],
            [2, "source", 1, PRIVATE_LOADER_ID, 1, "CLIP"],
        ],
    }


def _runtime_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "runtime",
        "engine": "comfyui",
        "runtime_build": "ComfyUI 0.28.0",
        "adapter_contract_version": 1,
        "launch_contract_version": "v1",
    }
    return ResolvedWorkflowBinding(
        slot_name="comfy-runtime",
        requirement_key="default",
        resource_kind="runtime",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("runtime", identity),
        mount={},
    )


def _asset_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "model_asset",
        "asset_kind": "lora",
        "runtime_reference": REFERENCE,
        "sha256": ASSET_DIGEST,
    }
    return ResolvedWorkflowBinding(
        slot_name="style",
        requirement_key="default",
        resource_kind="model_asset",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("model_asset", identity),
        mount={},
    )


def _package_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "registry_package",
        "package_id": "rgthree-comfy",
        "package_version": RGTHREE_REVISION,
        "registry_record_id": f"rgthree-comfy@{RGTHREE_REVISION}",
        "node_types": [POWER_LORA_LOADER],
        "archive_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "wheel_closure_sha256": "3" * 64,
        "wheel_environment_sha256": "4" * 64,
    }
    return ResolvedWorkflowBinding(
        slot_name="rgthree",
        requirement_key="default",
        resource_kind="registry_package",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("registry_package", identity),
        mount={},
    )


def _core_provider_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "registry_package",
        "package_id": "forged-core-provider",
        "package_version": "1.2.3",
        "registry_record_id": "forged-core-provider@1.2.3",
        "node_types": ["LoraLoader"],
        "archive_sha256": "5" * 64,
        "manifest_sha256": "6" * 64,
        "wheel_closure_sha256": "7" * 64,
        "wheel_environment_sha256": "8" * 64,
    }
    return ResolvedWorkflowBinding(
        slot_name="extension",
        requirement_key="default",
        resource_kind="registry_package",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("registry_package", identity),
        mount={},
    )


def _opaque_custom_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "custom_node",
        "source_url": "https://github.com/example/opaque-provider.git",
        "revision": "c" * 40,
        "tree_hash": "d" * 40,
    }
    return ResolvedWorkflowBinding(
        slot_name="extension",
        requirement_key="default",
        resource_kind="custom_node",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("custom_node", identity),
        mount={},
    )


def _forged_custom_types_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "custom_node",
        "source_url": "https://github.com/example/opaque-provider.git",
        "revision": "c" * 40,
        "tree_hash": "d" * 40,
        "node_types": ["UnrelatedNode"],
    }
    return ResolvedWorkflowBinding(
        slot_name="extension",
        requirement_key="default",
        resource_kind="custom_node",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("custom_node", identity),
        mount={},
    )


def _seed_revision(
    session: Session,
    *,
    rgthree: bool = False,
    with_contract: bool = True,
    invalid_activation: bool = False,
    mismatched_ui: bool = False,
    core_extension: CoreExtension | None = None,
) -> tuple[WorkflowDefinition, WorkflowRevision]:
    suffix = "rgthree" if rgthree else f"core_{core_extension or 'plain'}"
    definition = WorkflowDefinition(
        id=f"workflow_lora_{suffix}",
        name=f"LoRA projection {suffix}",
        operation="text_to_image",
    )
    session.add(definition)
    session.flush()
    extension_slots = (
        (
            _slot(
                "extension",
                "registry_package" if core_extension == "advertised_registry" else "custom_node",
            ),
        )
        if core_extension is not None
        else ()
    )
    contract = WorkflowDependencyContract(
        version=1,
        slots=(
            _slot("rgthree", "registry_package") if rgthree else _slot("comfy-runtime", "runtime"),
            *extension_slots,
            _slot("style", "model_asset", required=False),
        ),
    )
    revision = WorkflowRevision(
        id=f"wfrev_lora_{suffix}",
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        ui_graph_json=_ui_graph(rgthree=rgthree, mismatched=mismatched_ui),
        api_graph_json=_rgthree_api_graph() if rgthree else _core_api_graph(),
        input_schema_json={},
        dependencies_json={},
        dependency_contract_sha256=(
            workflow_dependency_contract_sha256(contract) if with_contract else None
        ),
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    if not with_contract:
        session.commit()
        return definition, revision

    slots: dict[str, WorkflowDependencySlot] = {}
    for ordinal, contract_slot in enumerate(contract.slots):
        payload = workflow_dependency_slot_payload(contract_slot)
        row = WorkflowDependencySlot(
            id=f"wfslot_{suffix}_{ordinal}",
            workflow_revision_id=revision.id,
            name=contract_slot.name,
            resource_kind=contract_slot.resource_kind,
            required=contract_slot.required,
            satisfaction=contract_slot.satisfaction,
            requirements_json=payload["requirements"],
            contract_sha256=workflow_dependency_slot_sha256(contract_slot),
            ordinal=ordinal,
        )
        session.add(row)
        slots[contract_slot.name] = row

    asset = ModelAssetInstall(
        id=f"asset_lora_{suffix}",
        name="Private local LoRA",
        kind="lora",
        family="test",
        local_path="C:/private/models/detail.safetensors",
        size_bytes=17,
        manifest_json={"sha256": ASSET_DIGEST, "comfy_name": REFERENCE},
        active=True,
    )
    session.add(asset)
    if rgthree:
        session.add(
            ComfyRegistryInstall(
                id="registry_lora_rgthree",
                package_id="rgthree-comfy",
                package_version=RGTHREE_REVISION,
                registry_record_id=f"rgthree-comfy@{RGTHREE_REVISION}",
                repository_url="https://github.com/rgthree/rgthree-comfy",
                download_url="https://example.invalid/rgthree.zip",
                archive_sha256="1" * 64,
                manifest_sha256="2" * 64,
                installed_path="C:/private/custom_nodes/rgthree",
                node_types_json=[POWER_LORA_LOADER],
                pip_dependencies_json=[],
                review_json={},
                wheel_closure_sha256="3" * 64,
                wheel_environment_sha256="4" * 64,
                wheel_environment_path=f"registry-wheels-v3-{'3' * 64}",
                trusted=True,
                active=True,
            )
        )
    elif core_extension == "advertised_registry":
        session.add(
            ComfyRegistryInstall(
                id=CORE_PROVIDER_REGISTRY_ID,
                package_id="forged-core-provider",
                package_version="1.2.3",
                registry_record_id="forged-core-provider@1.2.3",
                repository_url="https://github.com/example/forged-core-provider",
                download_url="https://example.invalid/forged-core-provider.zip",
                archive_sha256="5" * 64,
                manifest_sha256="6" * 64,
                installed_path="C:/private/custom_nodes/forged-core-provider",
                node_types_json=["LoraLoader"],
                pip_dependencies_json=[],
                review_json={},
                wheel_closure_sha256="7" * 64,
                wheel_environment_sha256="8" * 64,
                wheel_environment_path=f"registry-wheels-v3-{'7' * 64}",
                trusted=True,
                active=True,
            )
        )
    elif core_extension in {"opaque_custom", "forged_custom_types"}:
        session.add(
            CustomNodeInstall(
                id=OPAQUE_CUSTOM_NODE_ID,
                name="Opaque manual provider",
                source_url="https://github.com/example/opaque-provider.git",
                revision="c" * 40,
                installed_path="C:/private/custom_nodes/opaque-provider",
                tree_hash="d" * 40,
                trusted=True,
                active=True,
                security_json={},
            )
        )
    session.flush()

    extension_bindings: tuple[ResolvedWorkflowBinding, ...]
    if core_extension == "advertised_registry":
        extension_bindings = (_core_provider_binding(),)
    elif core_extension == "opaque_custom":
        extension_bindings = (_opaque_custom_binding(),)
    elif core_extension == "forged_custom_types":
        extension_bindings = (_forged_custom_types_binding(),)
    else:
        extension_bindings = ()
    bindings = (
        (_package_binding(), _asset_binding())
        if rgthree
        else (_runtime_binding(), *extension_bindings, _asset_binding())
    )
    binding_sha256 = workflow_activation_binding_sha256(contract, bindings)
    activation = WorkflowActivation(
        id=f"wfact_lora_{suffix}",
        workflow_revision_id=revision.id,
        resolver_version="workflow-activation-v1",
        dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
        binding_sha256="f" * 64 if invalid_activation else binding_sha256,
        state="ready",
        is_active=True,
        details_json={"launch_sha256": "e" * 64},
    )
    session.add(activation)
    session.flush()
    for index, binding in enumerate(bindings):
        locator: dict[str, str] = (
            {"model_asset_install_id": asset.id}
            if binding.resource_kind == "model_asset"
            else (
                {
                    "comfy_registry_install_id": (
                        "registry_lora_rgthree"
                        if binding.slot_name == "rgthree"
                        else CORE_PROVIDER_REGISTRY_ID
                    )
                }
                if binding.resource_kind == "registry_package"
                else (
                    {"custom_node_install_id": OPAQUE_CUSTOM_NODE_ID}
                    if binding.resource_kind == "custom_node"
                    else {"runtime_key": "comfyui-current"}
                )
            )
        )
        session.add(
            WorkflowDependencyBinding(
                id=f"wfbind_{suffix}_{index}",
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slots[binding.slot_name].id,
                requirement_key=binding.requirement_key,
                mount_json=binding.mount,
                resource_identity_json=binding.identity,
                resource_identity_sha256=binding.resource_identity_sha256,
                **locator,
            )
        )
    session.commit()
    return definition, revision


def test_core_projection_requires_complete_graph_runtime_and_asset_evidence(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session)

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ()
    assert projection.activation_binding_sha256 is not None
    assert len(projection.slots) == 1
    slot = projection.slots[0]
    assert slot.editability == "editable"
    assert slot.read_only_reason is None
    assert slot.editable_fields == ("model_strength", "clip_strength")
    assert slot.default_model_strength == 0.75
    assert slot.default_clip_strength == 0.5
    assert slot.asset_binding is not None
    assert slot.asset_binding.runtime_reference == REFERENCE
    assert slot.asset_binding.sha256 == ASSET_DIGEST
    assert not session.new and not session.dirty and not session.deleted
    public = WorkflowLoraControlsOut.model_validate(projection).model_dump(mode="json")
    encoded = json.dumps(public, sort_keys=True)
    assert PRIVATE_LOADER_ID not in encoded
    assert "C:/private" not in encoded
    assert revision.id not in encoded


def test_core_ui_api_mismatch_is_visible_but_never_editable(session: Session) -> None:
    definition, revision = _seed_revision(session, mismatched_ui=True)

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("core_graph_binding_unavailable",)
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"
    assert projection.slots[0].asset_binding is not None


def test_core_ui_api_connection_mismatch_cannot_authorize_the_same_node_id(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session)
    changed = deepcopy(revision.ui_graph_json)
    changed["links"][0][2] = 1
    changed["nodes"][0]["outputs"][0]["links"] = []
    changed["nodes"][0]["outputs"][1]["links"] = [1, 2]
    revision.ui_graph_json = changed
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("core_graph_binding_unavailable",)
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"


def test_ui_target_link_metadata_mismatch_cannot_authorize_core_loader(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session)
    changed = deepcopy(revision.ui_graph_json)
    changed["nodes"][1]["inputs"][0]["link"] = 999
    revision.ui_graph_json = changed
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("ui_graph_provenance_unavailable",)
    assert projection.activation_binding_sha256 is not None
    assert projection.slots[0].asset_binding is not None
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"


def test_ui_origin_output_omission_cannot_authorize_core_loader(session: Session) -> None:
    definition, revision = _seed_revision(session)
    changed = deepcopy(revision.ui_graph_json)
    changed["nodes"][0]["outputs"][0]["links"] = []
    revision.ui_graph_json = changed
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("ui_graph_provenance_unavailable",)
    assert projection.activation_binding_sha256 is not None
    assert projection.slots[0].asset_binding is not None
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"


def test_ui_origin_output_orphan_cannot_authorize_core_loader(session: Session) -> None:
    definition, revision = _seed_revision(session)
    changed = deepcopy(revision.ui_graph_json)
    changed["nodes"][0]["outputs"][0]["links"] = [1, 999]
    revision.ui_graph_json = changed
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("ui_graph_provenance_unavailable",)
    assert projection.activation_binding_sha256 is not None
    assert projection.slots[0].asset_binding is not None
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"


def test_ui_out_of_range_origin_slot_cannot_authorize_when_api_graph_agrees(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session)
    changed_ui = deepcopy(revision.ui_graph_json)
    changed_ui["links"][0][2] = 9
    revision.ui_graph_json = changed_ui
    changed_api = deepcopy(revision.api_graph_json)
    changed_api[PRIVATE_LOADER_ID]["inputs"]["model"] = ["source", 9]
    revision.api_graph_json = changed_api
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("ui_graph_provenance_unavailable",)
    assert projection.activation_binding_sha256 is not None
    assert projection.slots[0].asset_binding is not None
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"


def test_non_comfy_revision_cannot_reuse_comfy_loader_evidence(session: Session) -> None:
    definition, revision = _seed_revision(session)
    revision.engine = "other-media-engine"
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("ui_graph_provenance_unavailable",)
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].read_only_reason == "missing_core_evidence"


def test_activated_registry_provider_collision_vetoes_forged_core_claim(
    session: Session,
) -> None:
    definition, revision = _seed_revision(
        session,
        core_extension="advertised_registry",
    )

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("core_graph_binding_unavailable",)
    assert projection.activation_binding_sha256 is not None
    slot = projection.slots[0]
    assert slot.asset_binding is not None
    assert slot.loader_contract is None
    assert slot.loader_authority_sha256 is None
    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "missing_core_evidence"


def test_opaque_manual_custom_node_binding_vetoes_forged_core_claim(
    session: Session,
) -> None:
    definition, revision = _seed_revision(
        session,
        core_extension="opaque_custom",
    )

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("core_graph_binding_unavailable",)
    assert projection.activation_binding_sha256 is not None
    slot = projection.slots[0]
    assert slot.asset_binding is not None
    assert slot.loader_contract is None
    assert slot.loader_authority_sha256 is None
    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "missing_core_evidence"


def test_custom_node_forged_unrelated_types_cannot_evade_core_origin_veto(
    session: Session,
) -> None:
    definition, revision = _seed_revision(
        session,
        core_extension="forged_custom_types",
    )

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("core_graph_binding_unavailable",)
    assert projection.activation_binding_sha256 is not None
    slot = projection.slots[0]
    assert slot.asset_binding is not None
    assert slot.loader_contract is None
    assert slot.loader_authority_sha256 is None
    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "missing_core_evidence"


def test_invalid_activation_snapshot_cannot_supply_asset_or_loader_authority(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session, invalid_activation=True)

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("active_activation_invalid",)
    assert projection.activation_binding_sha256 is None
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].asset_binding is None


def test_slots_that_no_longer_add_up_to_the_recorded_contract_are_invalid(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session)
    revision.dependency_contract_sha256 = "0" * 64
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("dependency_contract_invalid",)
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].asset_binding is None


@pytest.mark.parametrize(
    "properties",
    [
        {"ver": "0.28.0"},
        {"cnr_id": "another-node-pack", "ver": "1.0.0"},
        {"cnr_id": "comfy-core", "aux_id": "someone/replacement-loader", "ver": "0.28.0"},
    ],
)
def test_a_core_loader_not_attributed_to_comfyui_itself_stays_read_only(
    session: Session,
    properties: dict[str, str],
) -> None:
    definition, revision = _seed_revision(session)
    ui_graph = deepcopy(revision.ui_graph_json)
    for node in ui_graph["nodes"]:
        if node["id"] == PRIVATE_LOADER_ID:
            node["properties"] = properties
    revision.ui_graph_json = ui_graph
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    slot = projection.slots[0]
    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "node_not_core_claimed"
    assert slot.editable_fields == ()


def test_invalid_persisted_contract_degrades_to_visible_read_only_slots(
    session: Session,
) -> None:
    definition, revision = _seed_revision(session)
    revision.dependency_slots[0].contract_sha256 = "0" * 64
    session.flush()

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("dependency_contract_invalid",)
    assert len(projection.slots) == 1
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].asset_binding is None


def test_legacy_revision_uses_an_explicit_empty_detection_contract(session: Session) -> None:
    definition, revision = _seed_revision(session, with_contract=False)

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    assert projection.evidence_gaps == ("dependency_contract_unavailable",)
    assert projection.activation_binding_sha256 is None
    assert len(projection.slots) == 1
    assert projection.slots[0].editability == "detected_read_only"
    assert projection.slots[0].asset_binding is None


@pytest.mark.parametrize("mismatched", [False, True])
def test_rgthree_requires_audited_ui_package_and_api_value_parity(
    session: Session,
    mismatched: bool,
) -> None:
    definition, revision = _seed_revision(
        session,
        rgthree=True,
        mismatched_ui=mismatched,
    )

    projection = workflow_lora_controls(
        session,
        revision_id=revision.id,
    )

    slot = projection.slots[0]
    if mismatched:
        assert projection.evidence_gaps == ("package_graph_binding_unavailable",)
        assert slot.editability == "detected_read_only"
        assert slot.read_only_reason == "missing_package_evidence"
    else:
        assert projection.evidence_gaps == ()
        assert slot.editability == "editable"
        assert slot.editable_fields == ("enabled", "model_strength", "clip_strength")
        assert slot.default_enabled is True
        assert slot.default_model_strength == 0.8
        assert slot.default_clip_strength == 0.6


@pytest.mark.asyncio
async def test_revision_scoped_api_returns_only_public_projection(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        _, revision = _seed_revision(session)
        revision_id = revision.id

    response = await client.get(f"/api/workflow-revisions/{revision_id}/lora-controls")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == 1
    assert payload["ordering_authority"] == "presentation_only"
    assert payload["evidence_gaps"] == []
    assert payload["slots"][0]["editability"] == "editable"
    assert payload["slots"][0]["asset_binding"] == {
        "dependency_slot": "style",
        "requirement_key": "default",
        "resource_identity_sha256": _asset_binding().resource_identity_sha256,
        "runtime_reference": REFERENCE,
        "sha256": ASSET_DIGEST,
    }
    encoded = response.text
    assert PRIVATE_LOADER_ID not in encoded
    assert "C:/private" not in encoded
    assert revision_id not in encoded

    unknown = await client.get("/api/workflow-revisions/wfrev_not_stored/lora-controls")
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "workflow-revision-not-found"
