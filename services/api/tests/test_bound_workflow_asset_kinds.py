from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from local_lm import model_asset_types, schemas
from local_lm.comfy_workflow_packages import AssetKind, WorkflowAssetReference
from local_lm.models import InstallPlan
from local_lm.workflow_asset_bindings import (
    _REFERENCE_ARTIFACT_KINDS,
    WorkflowAssetBindingError,
    _bound_asset,
)


@pytest.mark.parametrize(
    ("kind", "folder"),
    [
        ("checkpoint", "checkpoints"),
        ("embedding", "embeddings"),
        ("lora", "loras"),
        ("upscaler", "upscale_models"),
        ("vae", "vae"),
    ],
)
def test_every_bindable_kind_survives_response_validation(kind: AssetKind, folder: str) -> None:
    reference = WorkflowAssetReference(
        filename="neutral.safetensors", suffix=".safetensors", policy="supported", kind=kind
    )
    plan = InstallPlan(
        id="neutral-plan",
        plan_hash="a" * 64,
        provider="civitai",
        remote_id="101",
        revision="202",
    )
    bound = _bound_asset(
        reference,
        plan,
        {
            "required": True,
            "reuse": "download",
            "kind": kind,
            "target_folder": folder,
            "path": "neutral.safetensors",
            "size_bytes": 17,
            "sha256": "b" * 64,
        },
    )
    assert bound.kind == kind
    assert schemas.BoundWorkflowAssetOut.model_validate(bound.as_dict()).kind == kind


@pytest.mark.parametrize("kind", ["configuration", "unknown", ""])
def test_bound_response_rejects_kinds_that_cannot_be_bound(kind: str) -> None:
    with pytest.raises(ValidationError):
        schemas.BoundWorkflowAssetOut.model_validate(
            {
                "reference_filename": "neutral.safetensors",
                "kind": kind,
                "install_plan_id": "neutral-plan",
                "install_plan_hash": "a" * 64,
                "provider": "civitai",
                "remote_id": "101",
                "revision": "202",
                "artifact_path": "neutral.safetensors",
                "artifact_kind": "lora",
                "target_folder": "loras",
                "size_bytes": 17,
                "sha256": "b" * 64,
            }
        )


def test_reference_vocabulary_extends_the_exact_binding_vocabulary() -> None:
    binding = set(get_args(model_asset_types.BoundWorkflowAssetKind))
    assert binding == set(_REFERENCE_ARTIFACT_KINDS)
    assert set(get_args(AssetKind)) == binding | {"configuration"}
    assert set(get_args(schemas.WorkflowAssetKind)) == set(get_args(AssetKind))


def test_configuration_reference_still_cannot_bind_a_model() -> None:
    reference = WorkflowAssetReference(
        filename="neutral.json", suffix=".json", policy="supported", kind="configuration"
    )
    with pytest.raises(WorkflowAssetBindingError) as caught:
        _bound_asset(
            reference,
            InstallPlan(id="neutral-plan"),
            {
                "required": True,
                "reuse": "download",
                "kind": "lora",
                "target_folder": "loras",
            },
        )
    assert caught.value.code == "artifact_kind_mismatch"
