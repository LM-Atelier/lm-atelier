from __future__ import annotations

import pytest

from local_lm.comfy_workflow_packages import WorkflowAssetReference
from local_lm.model_planner import INSTALL_RESOLVER_VERSION
from local_lm.models import InstallPlan
from local_lm.workflow_asset_bindings import (
    WorkflowAssetBindingError,
    WorkflowAssetBindingPlan,
    WorkflowAssetPlanSelection,
    bind_workflow_assets_to_install_plans,
)
from local_lm.workflow_asset_downloads import (
    WorkflowAssetDownloadError,
    compose_workflow_asset_download_requests,
    install_plan_download_request,
)

_PATH = "detail.safetensors"


def _plan(provider: str, revision: str) -> InstallPlan:
    return InstallPlan(
        id="plan_detail",
        provider=provider,
        remote_id="101" if provider == "civitai" else "example/detail",
        revision=revision,
        role="image",
        engine="comfyui",
        plan_hash="b" * 64,
        resolver_version=INSTALL_RESOLVER_VERSION,
        compatibility="supported",
        artifacts_json=[
            {
                "path": _PATH,
                "kind": "lora",
                "target_folder": "loras",
                "size_bytes": 17,
                "sha256": "a" * 64,
                "required": True,
                "reuse": "download",
                "source_version_id": revision if provider == "civitai" else None,
                "source_file_id": "301" if provider == "civitai" else None,
            }
        ],
        runtime_contract_json={"auxiliary_kind": "lora", "comfy_paths": {"loras": "."}},
        activation_probe_json={},
        status="planned",
    )


def _bind(plan: InstallPlan) -> WorkflowAssetBindingPlan:
    return bind_workflow_assets_to_install_plans(
        [
            WorkflowAssetReference(
                filename=_PATH,
                suffix=".safetensors",
                policy="supported",
                kind="lora",
                present_locally=False,
            )
        ],
        [WorkflowAssetPlanSelection(_PATH, plan.id, _PATH)],
        {plan.id: plan},
    )


@pytest.mark.parametrize(
    ("provider", "revision"),
    [
        ("huggingface", "202"),
        ("huggingface", "a" * 64),
        ("huggingface", "release"),
        ("huggingface", "a" * 12),
        ("civitai", "a" * 40),
        ("civitai", "a" * 64),
        ("civitai", "0202"),
        ("unknown", "a" * 40),
    ],
)
def test_workflow_binding_requires_the_providers_revision_format(
    provider: str, revision: str
) -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        _bind(_plan(provider, revision))
    assert caught.value.code == "mutable_install_plan"


@pytest.mark.parametrize("revision", ["202", "a" * 64, "release"])
def test_direct_workflow_download_refuses_a_hugging_face_branch(revision: str) -> None:
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(_plan("huggingface", revision))
    assert caught.value.code == "invalid_install_plan"


@pytest.mark.parametrize("revision", ["202", "a" * 64, "release"])
def test_workflow_companion_requires_a_hugging_face_commit(revision: str) -> None:
    plan = _plan("huggingface", "c" * 40)
    plan.artifacts_json[0].update(
        source_remote_id="example/companion",
        source_revision=revision,
        source_path="weights/detail.safetensors",
    )
    with pytest.raises(WorkflowAssetDownloadError) as caught:
        install_plan_download_request(plan)
    assert caught.value.code == "incomplete_artifact_source"


@pytest.mark.parametrize(
    ("provider", "revision"),
    [("huggingface", "c" * 40), ("huggingface", "1" * 40), ("civitai", "202")],
)
def test_exact_provider_revisions_survive_binding_and_download(
    provider: str, revision: str
) -> None:
    plan = _plan(provider, revision)
    binding = _bind(plan)
    (request,) = compose_workflow_asset_download_requests(
        binding, {plan.id: plan}, expected_binding_plan_hash=binding.plan_hash
    )
    assert request.revision == revision
    assert request.remote_id == plan.remote_id
    assert request.allow_patterns == [_PATH]
    assert request.expected_sha256 == {_PATH: "a" * 64}
