"""Installation previews reserve enough space for materialized model identities."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session
from test_workflow_bindings import session as session
from test_workflow_package_install_plans import (
    _installed_resource_slot,
    _sized_optional_declaration,
)

from local_lm.model_manifests import COMFY_MODEL_FOLDERS
from local_lm.models import InstallPlan, ModelComponentManifest, ModelInstall
from local_lm.schemas import DownloadRequest
from local_lm.workflow_bindings import materialize_model_install
from local_lm.workflow_dependencies import (
    MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES,
    WorkflowDependencyError,
    canonical_workflow_dependency_json,
    parse_workflow_dependency_contract,
)
from local_lm.workflow_source_dependency_budget import validate_workflow_source_dependency_budget
from local_lm.workflow_source_dependency_contract import WorkflowSourceDependencyBuilder


@pytest.mark.parametrize("planned_kind", ["checkpoint", "unknown_safetensors", "diffusion_model"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("unicode_filename", [False, True])
def test_model_budget_bounds_the_materialized_component_closure(
    session: Session, planned_kind: str, nested: bool, unicode_filename: bool
) -> None:
    generic = planned_kind != "diffusion_model"
    final_kind = "x" * 200 if generic else planned_kind
    final_folder = max(COMFY_MODEL_FOLDERS, key=len) if generic else "diffusion_models"
    filename = "色.safetensors" if unicode_filename else "neutral.safetensors"
    path = "weights/" + filename if nested else filename
    artifact = {
        "path": path,
        "kind": planned_kind,
        "target_folder": "checkpoints" if generic else final_folder,
        "required": True,
    }
    request = DownloadRequest(
        install_plan_id="plan_neutral",
        remote_id="neutral/model",
        role="image",
        engine="comfyui",
        allow_patterns=[path],
        expected_sha256={path: "a" * 64},
    )
    plan = InstallPlan(id="plan_neutral", artifacts_json=[artifact])
    install = ModelInstall(
        id="model_neutral",
        name="Neutral model",
        role="image",
        engine="comfyui",
        local_path="neutral-model",
        manifest_json={"comfy_paths": {final_folder: "weights" if nested else "."}},
        active=True,
    )
    session.add(install)
    session.flush()
    session.add(
        ModelComponentManifest(
            id="component_neutral",
            model_install_id=install.id,
            kind=final_kind,
            target_folder=final_folder,
            relative_path=path,
            sha256="a" * 64,
            required=True,
        )
    )
    session.flush()
    actual = materialize_model_install(session, install).identity
    assert actual["components"][0]["runtime_reference"] == filename
    reserved = {**actual, "components": [{**actual["components"][0], "runtime_reference": path}]}
    additions = [
        _installed_resource_slot("model_files_1", "model_install", reserved),
        _installed_resource_slot("media_runtime_2", "runtime", {"engine": "comfyui"}),
    ]
    original_size = MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES - sum(
        len(canonical_workflow_dependency_json(slot)) + 1 for slot in additions
    )
    original = parse_workflow_dependency_contract(_sized_optional_declaration(original_size))
    validate_workflow_source_dependency_budget(original, [request], {plan.id: plan}, {}, "comfyui")
    builder = WorkflowSourceDependencyBuilder(original)
    builder.include("model_install", actual)
    builder.include("runtime", {"engine": "comfyui"})
    result = builder.payload()
    assert len(canonical_workflow_dependency_json(result)) == (
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES - (len("weights/") if nested else 0)
    )
    overflowing = parse_workflow_dependency_contract(_sized_optional_declaration(original_size + 1))
    with pytest.raises(WorkflowDependencyError) as refused:
        validate_workflow_source_dependency_budget(
            overflowing, [request], {plan.id: plan}, {}, "comfyui"
        )
    assert refused.value.code == "dependency_data_too_large"
