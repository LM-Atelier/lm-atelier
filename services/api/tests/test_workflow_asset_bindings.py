from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.comfy_workflow_packages import WorkflowAssetReference
from local_lm.models import InstallPlan
from local_lm.workflow_asset_bindings import (
    WorkflowAssetBindingError,
    WorkflowAssetPlanSelection,
    bind_workflow_assets_to_install_plans,
)


def reference(
    filename: str = "styles/detail.safetensors",
    *,
    kind: str = "lora",
    policy: str = "supported",
    present: bool = False,
) -> WorkflowAssetReference:
    return WorkflowAssetReference(
        filename=filename,
        suffix=".safetensors",
        policy=policy,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        present_locally=present,
    )


def install_plan(
    *,
    plan_id: str = "plan_lora",
    path: str = "styles/detail.safetensors",
    kind: str = "lora",
    target_folder: str = "loras",
    artifact_overrides: dict[str, Any] | None = None,
    plan_overrides: dict[str, Any] | None = None,
) -> InstallPlan:
    artifact: dict[str, Any] = {
        "path": path,
        "kind": kind,
        "target_folder": target_folder,
        "size_bytes": 17,
        "sha256": "a" * 64,
        "required": True,
        "reuse": "download",
    }
    artifact.update(artifact_overrides or {})
    values: dict[str, Any] = {
        "id": plan_id,
        "provider": "civitai",
        "remote_id": "101",
        "revision": "202",
        "role": "image",
        "engine": "comfyui",
        "plan_hash": "b" * 64,
        "resolver_version": "test",
        "compatibility": "supported",
        "artifacts_json": [artifact],
        "runtime_contract_json": {},
        "activation_probe_json": {},
        "status": "planned",
    }
    values.update(plan_overrides or {})
    return InstallPlan(**values)


def selection(
    filename: str = "styles/detail.safetensors",
    *,
    plan_id: str = "plan_lora",
    artifact_path: str | None = None,
) -> WorkflowAssetPlanSelection:
    return WorkflowAssetPlanSelection(filename, plan_id, artifact_path or filename)


def test_binds_each_missing_asset_to_one_exact_verified_plan() -> None:
    checkpoint_name = "models/portrait.safetensors"
    plans = {
        "plan_lora": install_plan(),
        "plan_checkpoint": install_plan(
            plan_id="plan_checkpoint",
            path=checkpoint_name,
            kind="checkpoint",
            target_folder="checkpoints",
            plan_overrides={
                "provider": "huggingface",
                "remote_id": "author/portrait",
                "revision": "c" * 40,
                "plan_hash": "d" * 64,
            },
        ),
    }

    result = bind_workflow_assets_to_install_plans(
        [reference(), reference(checkpoint_name, kind="checkpoint")],
        [selection(checkpoint_name, plan_id="plan_checkpoint"), selection()],
        plans,
    )

    assert [asset.reference_filename for asset in result.assets] == [
        "styles/detail.safetensors",
        checkpoint_name,
    ]
    assert result.assets[0].provider == "civitai"
    assert result.assets[1].revision == "c" * 40
    assert len(result.plan_hash) == 64
    assert (
        result.plan_hash
        == bind_workflow_assets_to_install_plans(
            [reference(), reference(checkpoint_name, kind="checkpoint")],
            [selection(), selection(checkpoint_name, plan_id="plan_checkpoint")],
            dict(reversed(list(plans.items()))),
        ).plan_hash
    )
    changed_plan = install_plan(plan_overrides={"plan_hash": "e" * 64})
    assert (
        result.plan_hash
        != bind_workflow_assets_to_install_plans(
            [reference()],
            [selection()],
            {"plan_lora": changed_plan},
        ).plan_hash
    )


def test_already_present_assets_need_no_binding() -> None:
    result = bind_workflow_assets_to_install_plans(
        [reference(present=True)],
        [],
        {},
    )
    assert result.assets == ()


@pytest.mark.parametrize(
    ("selections", "code"),
    [
        ([], "missing_asset_selection"),
        (
            [selection(), selection("extra.safetensors")],
            "unexpected_asset_selection",
        ),
        ([selection(), selection()], "duplicate_asset_selection"),
    ],
)
def test_selection_set_must_exactly_match_missing_assets(
    selections: list[WorkflowAssetPlanSelection],
    code: str,
) -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()], selections, {"plan_lora": install_plan()}
        )
    assert caught.value.code == code


def test_selection_cannot_target_an_already_present_asset() -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference(present=True)],
            [selection()],
            {"plan_lora": install_plan()},
        )
    assert caught.value.code == "unexpected_asset_selection"


@pytest.mark.parametrize("policy", ["blocked", "unsupported"])
def test_non_installable_workflow_asset_blocks_the_binding(policy: str) -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference(policy=policy)],
            [selection()],
            {"plan_lora": install_plan()},
        )
    assert caught.value.code == "unsupported_asset_reference"


def test_case_variants_are_ambiguous_and_never_collapsed() -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference(), reference("STYLES/DETAIL.SAFETENSORS")],
            [],
            {},
        )
    assert caught.value.code == "duplicate_asset_reference"


def test_reference_suffix_must_match_its_filename() -> None:
    value = reference()
    value = WorkflowAssetReference(
        filename=value.filename,
        suffix=".gguf",
        policy=value.policy,
        kind=value.kind,
        present_locally=value.present_locally,
    )
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans([value], [selection()], {})
    assert caught.value.code == "invalid_asset_reference"


def test_selection_preserves_exact_workflow_filename_case() -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()],
            [selection("STYLES/DETAIL.SAFETENSORS", artifact_path="styles/detail.safetensors")],
            {"plan_lora": install_plan()},
        )
    assert caught.value.code == "asset_reference_case_mismatch"


def test_selected_plan_must_exist() -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans([reference()], [selection()], {})
    assert caught.value.code == "install_plan_not_found"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"status": "downloading"}, "install_plan_not_pending"),
        ({"compatibility": "unsupported"}, "install_plan_not_supported"),
        ({"failure_code": "blocked"}, "install_plan_not_supported"),
        ({"revision": "main"}, "mutable_install_plan"),
        ({"plan_hash": "B" * 64}, "mutable_install_plan"),
        ({"artifacts_json": {}}, "invalid_install_plan"),
    ],
)
def test_install_plan_must_be_supported_pending_and_immutable(
    overrides: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()],
            [selection()],
            {"plan_lora": install_plan(plan_overrides=overrides)},
        )
    assert caught.value.code == code


def test_install_plan_artifact_inventory_is_bounded() -> None:
    plan = install_plan()
    plan.artifacts_json = [deepcopy(plan.artifacts_json[0]) for _ in range(513)]
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()],
            [selection()],
            {"plan_lora": plan},
        )
    assert caught.value.code == "invalid_install_plan"


@pytest.mark.parametrize(
    ("artifact_overrides", "code"),
    [
        ({"required": False}, "artifact_not_downloadable"),
        ({"reuse": "installed"}, "artifact_not_downloadable"),
        ({"kind": "vae", "target_folder": "vae"}, "artifact_kind_mismatch"),
        ({"target_folder": "checkpoints"}, "artifact_folder_mismatch"),
        ({"size_bytes": 0}, "unverified_plan_artifact"),
        ({"size_bytes": True}, "unverified_plan_artifact"),
        ({"sha256": "A" * 64}, "unverified_plan_artifact"),
    ],
)
def test_artifact_contract_fails_closed(
    artifact_overrides: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()],
            [selection()],
            {"plan_lora": install_plan(artifact_overrides=artifact_overrides)},
        )
    assert caught.value.code == code


def test_artifact_path_must_match_the_workflow_without_substitution() -> None:
    plan = install_plan(path="styles/other.safetensors")
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()],
            [selection(artifact_path="styles/other.safetensors")],
            {"plan_lora": plan},
        )
    assert caught.value.code == "artifact_path_mismatch"


def test_duplicate_plan_artifact_path_is_ambiguous() -> None:
    plan = install_plan()
    plan.artifacts_json.append(deepcopy(plan.artifacts_json[0]))
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans(
            [reference()],
            [selection()],
            {"plan_lora": plan},
        )
    assert caught.value.code == "ambiguous_plan_artifact"


@pytest.mark.parametrize(
    "value",
    [
        "../escape.safetensors",
        "/absolute.safetensors",
        "folder\\file.safetensors",
        "line\nfeed.safetensors",
    ],
)
def test_unsafe_reference_paths_are_rejected(value: str) -> None:
    with pytest.raises(WorkflowAssetBindingError) as caught:
        bind_workflow_assets_to_install_plans([reference(value)], [], {})
    assert caught.value.code == "invalid_asset_reference"
