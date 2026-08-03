from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from local_lm.comfy_workflow_packages import WorkflowAssetReference
from local_lm.model_planner import INSTALL_RESOLVER_VERSION
from local_lm.models import InstallPlan
from local_lm.workflow_asset_bindings import (
    WorkflowAssetBindingPlan,
    WorkflowAssetPlanSelection,
    bind_workflow_assets_to_install_plans,
)
from local_lm.workflow_asset_downloads import (
    WorkflowAssetDownloadError,
    compose_workflow_asset_download_requests,
    install_plan_download_request,
)


def _artifact(
    path: str,
    *,
    kind: str = "lora",
    folder: str = "loras",
    digest: str = "a" * 64,
    size: int = 17,
    **extra: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path,
        "kind": kind,
        "target_folder": folder,
        "size_bytes": size,
        "sha256": digest,
        "required": True,
        "reuse": "download",
    }
    value.update(extra)
    return value


def _plan(
    plan_id: str = "plan_lora",
    *,
    provider: str = "civitai",
    artifacts: list[dict[str, Any]] | None = None,
    runtime: dict[str, Any] | None = None,
    **overrides: Any,
) -> InstallPlan:
    selected_artifacts = deepcopy(artifacts or [_artifact("styles/detail.safetensors")])
    if provider == "civitai":
        for index, artifact in enumerate(selected_artifacts, start=301):
            artifact.setdefault("source_version_id", "202")
            artifact.setdefault("source_file_id", str(index))
    values: dict[str, Any] = {
        "id": plan_id,
        "provider": provider,
        "remote_id": "101" if provider == "civitai" else "author/model",
        "revision": "202" if provider == "civitai" else "c" * 40,
        "role": "image",
        "engine": "comfyui",
        "plan_hash": ("b" if plan_id == "plan_lora" else "d") * 64,
        "resolver_version": INSTALL_RESOLVER_VERSION,
        "compatibility": "supported",
        "artifacts_json": selected_artifacts,
        "runtime_contract_json": runtime
        or {"auxiliary_kind": "lora", "comfy_paths": {"loras": "styles"}},
        "activation_probe_json": {},
        "status": "planned",
    }
    values.update(overrides)
    return InstallPlan(**values)


def _reference(path: str, *, kind: str = "lora") -> WorkflowAssetReference:
    return WorkflowAssetReference(
        filename=path,
        suffix=".safetensors",
        policy="supported",
        kind=kind,  # type: ignore[arg-type]
        present_locally=False,
    )


def _binding(
    references: list[WorkflowAssetReference],
    selections: list[WorkflowAssetPlanSelection],
    plans: dict[str, InstallPlan],
) -> WorkflowAssetBindingPlan:
    return bind_workflow_assets_to_install_plans(references, selections, plans)


def test_civitai_binding_becomes_one_exact_plan_request() -> None:
    plan = _plan()
    plans = {plan.id: plan}
    binding = _binding(
        [_reference("styles/detail.safetensors")],
        [
            WorkflowAssetPlanSelection(
                "styles/detail.safetensors", plan.id, "styles/detail.safetensors"
            )
        ],
        plans,
    )

    (request,) = compose_workflow_asset_download_requests(
        binding, plans, expected_binding_plan_hash=binding.plan_hash
    )

    assert request.install_plan_id == plan.id
    assert request.remote_id == "101"
    assert request.revision == "202"
    assert request.allow_patterns == ["styles/detail.safetensors"]
    assert request.expected_sha256 == {"styles/detail.safetensors": "a" * 64}
    assert request.file_sources == {}
    assert request.comfy_paths == {"loras": "styles"}
    assert request.auxiliary_kind == "lora"


def test_two_references_to_one_plan_fold_into_one_full_request() -> None:
    paths = ["models/base.safetensors", "models/detail.safetensors"]
    plan = _plan(
        artifacts=[
            _artifact(paths[0], kind="checkpoint", folder="checkpoints"),
            _artifact(paths[1], digest="e" * 64),
        ],
        runtime={"comfy_paths": {"checkpoints": "models", "loras": "models"}},
    )
    plans = {plan.id: plan}
    binding = _binding(
        [_reference(paths[0], kind="checkpoint"), _reference(paths[1])],
        [
            WorkflowAssetPlanSelection(paths[0], plan.id, paths[0]),
            WorkflowAssetPlanSelection(paths[1], plan.id, paths[1]),
        ],
        plans,
    )

    requests = compose_workflow_asset_download_requests(
        binding, plans, expected_binding_plan_hash=binding.plan_hash
    )

    assert len(requests) == 1
    assert requests[0].allow_patterns == paths
    assert requests[0].expected_sha256 == {paths[0]: "a" * 64, paths[1]: "e" * 64}


def test_distinct_plan_requests_preserve_first_reference_order() -> None:
    first = _plan()
    second_path = "models/base.safetensors"
    second = _plan(
        "plan_checkpoint",
        provider="huggingface",
        artifacts=[_artifact(second_path, kind="checkpoint", folder="checkpoints")],
        runtime={"comfy_paths": {"checkpoints": "models"}},
    )
    plans = {first.id: first, second.id: second}
    binding = _binding(
        [_reference(second_path, kind="checkpoint"), _reference("styles/detail.safetensors")],
        [
            WorkflowAssetPlanSelection(second_path, second.id, second_path),
            WorkflowAssetPlanSelection(
                "styles/detail.safetensors", first.id, "styles/detail.safetensors"
            ),
        ],
        plans,
    )

    requests = compose_workflow_asset_download_requests(
        binding, plans, expected_binding_plan_hash=binding.plan_hash
    )

    assert [request.install_plan_id for request in requests] == [second.id, first.id]


def test_hugging_face_companion_sources_are_preserved() -> None:
    primary = "models/base.safetensors"
    companion = "encoders/text.safetensors"
    plan = _plan(
        "plan_checkpoint",
        provider="huggingface",
        artifacts=[
            _artifact(primary, kind="diffusion_model", folder="diffusion_models"),
            _artifact(
                companion,
                kind="text_encoder",
                folder="text_encoders",
                digest="f" * 64,
                source_remote_id="author/text",
                source_revision="e" * 40,
                source_path="weights/text.safetensors",
            ),
        ],
        runtime={
            "comfy_paths": {"diffusion_models": "models", "text_encoders": "encoders"},
            "workflow_template_id": "image_bundle",
            "workflow_template_sha256": "9" * 64,
            "workflow_compiler_version": COMFY_TEMPLATE_COMPILER_VERSION,
        },
    )

    request = install_plan_download_request(plan)

    assert list(request.file_sources) == [companion]
    source = request.file_sources[companion]
    assert source.remote_id == "author/text"
    assert source.revision == "e" * 40
    assert source.filename == "weights/text.safetensors"
    assert source.size_bytes == 17
    assert source.sha256 == "f" * 64
    assert request.workflow_template_id == "image_bundle"
    assert request.workflow_template_sha256 == "9" * 64


def test_empty_binding_needs_no_downloads() -> None:
    binding = _binding([], [], {})
    assert (
        compose_workflow_asset_download_requests(
            binding, {}, expected_binding_plan_hash=binding.plan_hash
        )
        == ()
    )


def test_confirmation_hash_must_match_the_current_binding() -> None:
    binding = _binding([], [], {})
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        compose_workflow_asset_download_requests(binding, {}, expected_binding_plan_hash="f" * 64)
    assert caught.value.code == "binding_plan_changed"


def test_bound_plan_and_artifact_are_revalidated() -> None:
    plan = _plan()
    plans = {plan.id: plan}
    binding = _binding(
        [_reference("styles/detail.safetensors")],
        [
            WorkflowAssetPlanSelection(
                "styles/detail.safetensors", plan.id, "styles/detail.safetensors"
            )
        ],
        plans,
    )

    changed = _plan(plan_hash="e" * 64)
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        compose_workflow_asset_download_requests(
            binding, {changed.id: changed}, expected_binding_plan_hash=binding.plan_hash
        )
    assert caught.value.code == "install_plan_changed"

    artifact_changed = _plan()
    artifact_changed.artifacts_json[0]["size_bytes"] = 18
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        compose_workflow_asset_download_requests(
            binding,
            {artifact_changed.id: artifact_changed},
            expected_binding_plan_hash=binding.plan_hash,
        )
    assert caught.value.code == "binding_asset_changed"

    malformed = _plan()
    malformed.artifacts_json = None  # type: ignore[assignment]
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        compose_workflow_asset_download_requests(
            binding,
            {malformed.id: malformed},
            expected_binding_plan_hash=binding.plan_hash,
        )
    assert caught.value.code == "invalid_install_plan"


def test_missing_bound_plan_is_rejected() -> None:
    plan = _plan()
    binding = _binding(
        [_reference("styles/detail.safetensors")],
        [
            WorkflowAssetPlanSelection(
                "styles/detail.safetensors", plan.id, "styles/detail.safetensors"
            )
        ],
        {plan.id: plan},
    )
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        compose_workflow_asset_download_requests(
            binding, {}, expected_binding_plan_hash=binding.plan_hash
        )
    assert caught.value.code == "install_plan_not_found"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"provider": "unknown"}, "unsupported_install_provider"),
        ({"status": "downloading"}, "install_plan_not_pending"),
        ({"compatibility": "unsupported"}, "install_plan_not_supported"),
        ({"failure_code": "blocked"}, "install_plan_not_supported"),
        ({"resolver_version": "old"}, "install_contract_changed"),
        ({"plan_hash": "B" * 64}, "invalid_install_plan"),
        ({"revision": "main"}, "invalid_install_plan"),
        ({"artifacts_json": []}, "invalid_install_plan"),
        ({"runtime_contract_json": []}, "invalid_install_plan"),
    ],
)
def test_plan_state_fails_closed(overrides: dict[str, Any], code: str) -> None:
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(_plan(**overrides))
    assert caught.value.code == code


def test_workflow_compiler_version_must_still_match() -> None:
    plan = _plan(
        runtime={
            "workflow_template_id": "image_bundle",
            "workflow_template_sha256": "9" * 64,
            "workflow_compiler_version": COMFY_TEMPLATE_COMPILER_VERSION - 1,
        }
    )
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(plan)
    assert caught.value.code == "workflow_contract_changed"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"size_bytes": 0}, "unverified_install_artifact"),
        ({"size_bytes": True}, "unverified_install_artifact"),
        ({"sha256": None}, "unverified_install_artifact"),
        ({"sha256": "A" * 64}, "unverified_install_artifact"),
        ({"path": "../escape.safetensors"}, "invalid_install_artifact"),
    ],
)
def test_every_required_artifact_is_safe_and_verified(mutation: dict[str, Any], code: str) -> None:
    artifact = _artifact("styles/detail.safetensors")
    artifact.update(mutation)
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(_plan(artifacts=[artifact]))
    assert caught.value.code == code


def test_duplicate_required_artifact_paths_are_rejected() -> None:
    artifact = _artifact("styles/detail.safetensors")
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(_plan(artifacts=[artifact, deepcopy(artifact)]))
    assert caught.value.code == "ambiguous_install_artifact"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"source_remote_id": None}, "incomplete_artifact_source"),
        ({"source_revision": None}, "incomplete_artifact_source"),
        ({"source_revision": "main"}, "incomplete_artifact_source"),
        ({"source_path": None}, "incomplete_artifact_source"),
        ({"source_path": "../text.safetensors"}, "invalid_install_artifact"),
    ],
)
def test_hugging_face_companion_source_must_be_exact(mutation: dict[str, Any], code: str) -> None:
    artifact = _artifact(
        "encoders/text.safetensors",
        kind="text_encoder",
        folder="text_encoders",
        source_remote_id="author/text",
        source_revision="e" * 40,
        source_path="weights/text.safetensors",
    )
    artifact.update(mutation)
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(
            _plan("plan_checkpoint", provider="huggingface", artifacts=[artifact])
        )
    assert caught.value.code == code


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_version_id": None},
        {"source_version_id": "203"},
        {"source_file_id": None},
        {"source_file_id": "latest"},
    ],
)
def test_civitai_artifact_needs_exact_version_and_file_identity(
    mutation: dict[str, Any],
) -> None:
    artifact = _artifact(
        "styles/detail.safetensors",
        source_version_id="202",
        source_file_id="301",
    )
    artifact.update(mutation)
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(_plan(artifacts=[artifact]))
    assert caught.value.code == "incomplete_civitai_provenance"


@pytest.mark.parametrize("required", ["yes", 1, None])
def test_malformed_required_flags_are_not_silently_ignored(required: object) -> None:
    artifact = _artifact("styles/detail.safetensors", required=required)
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(_plan(artifacts=[artifact]))
    assert caught.value.code == "invalid_install_plan"
