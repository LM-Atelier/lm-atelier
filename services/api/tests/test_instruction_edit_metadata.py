from __future__ import annotations

from typing import Any

import pytest

from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.config import Settings
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.model_manifests import InspectedComponent, ModelManifestInspection
from local_lm.model_planner import ResolvedInstallPlan, resolve_install_plan
from local_lm.models import InstallPlan
from local_lm.schemas import DownloadRequest


@pytest.mark.parametrize(
    ("item", "version", "expected"),
    [
        ({"tags": ["instruction-edit"]}, {}, "declared"),
        ({"tags": ["Instruction Editing"]}, {}, "declared"),
        ({"name": "Instruction Edit"}, {}, "declared"),
        ({}, {"name": "instruction_based_image_editing"}, "declared"),
        ({"tags": ["inpainting", "image-edit", "img2img"]}, {}, "unknown"),
        ({"name": "Not instruction edit"}, {}, "unknown"),
        ({"description": "Supports instruction editing"}, {}, "unknown"),
        ({}, {"trainedWords": ["instruction-edit"]}, "unknown"),
        ({"tags": [None, False, 4]}, {}, "unknown"),
        ({}, {}, "unknown"),
    ],
)
def test_provider_instruction_declarations_are_distinct_from_generic_edit_labels(
    item: dict[str, Any], version: dict[str, Any], expected: str
) -> None:
    files = CivitaiCatalog._normalize_files(
        {"id": 101, "type": "Checkpoint", **item},
        {
            "id": 202,
            "files": [{"id": 303, "name": "model.safetensors"}],
            **version,
        },
    )
    assert files[0]["metadata"].get("instruction_edit_capability") == expected


def _plan(declarations: list[object]) -> ResolvedInstallPlan:
    files: list[dict[str, Any]] = [
        {
            "filename": f"model-{index}.safetensors",
            "size": 1024,
            "sha256": "a" * 64,
            "metadata": {"instruction_edit_capability": declaration},
        }
        for index, declaration in enumerate(declarations)
    ]
    return resolve_install_plan(
        remote_id="neutral/model",
        revision="b" * 40,
        role="image",
        engine="comfyui",
        selected_files=files,
        inspection=ModelManifestInspection(
            architecture="stable-diffusion-xl",
            family="stable-diffusion-xl",
            components=tuple(
                InspectedComponent(item["filename"], "checkpoint", "checkpoints") for item in files
            ),
            metadata_files=(),
        ),
        workflow_template_id="neutral-template",
        workflow_template_sha256="c" * 64,
    )


@pytest.mark.parametrize(
    ("declarations", "expected"),
    [
        (["declared"], "declared"),
        (["declared", "declared"], "declared"),
        (["declared", "unknown"], "unknown"),
        (["unknown", "declared"], "unknown"),
        (["declared", None], "unknown"),
        ([True], "unknown"),
        ([{"declared": True}], "unknown"),
        ([["declared"]], "unknown"),
    ],
)
def test_the_plan_freezes_only_unambiguous_instruction_declarations(
    declarations: list[object], expected: str
) -> None:
    plan = _plan(declarations)
    assert plan.runtime_contract.get("instruction_edit_capability") == expected
    assert "instruction_edit_capability" not in plan.runtime_contract.get("use_case_metadata", {})


def test_an_instruction_declaration_changes_the_accepted_plan_identity() -> None:
    declared = _plan(["declared"])
    unknown = _plan(["unknown"])
    assert declared.plan_hash != unknown.plan_hash


@pytest.mark.parametrize("declaration", ["declared", "unknown", None, False, ["declared"]])
def test_download_sources_keep_the_accepted_capability_without_new_provider_metadata(
    settings: Settings, declaration: object
) -> None:
    digest = "a" * 64
    plan = InstallPlan(
        id="plan_instruction_metadata",
        provider="civitai",
        remote_id="202",
        revision="202",
        role="image",
        engine="comfyui",
        plan_hash="b" * 64,
        resolver_version="test",
        compatibility="supported",
        artifacts_json=[
            {
                "path": "model.safetensors",
                "size_bytes": 1024,
                "sha256": digest,
                "source_file_id": "303",
                "source_version_id": "202",
            }
        ],
        runtime_contract_json={"instruction_edit_capability": declaration},
        activation_probe_json={},
    )
    request = DownloadRequest(
        install_plan_id=plan.id,
        remote_id="202",
        revision="202",
        role="image",
        engine="comfyui",
        allow_patterns=["model.safetensors"],
        expected_sha256={"model.safetensors": digest},
    )
    manager = DownloadManager(settings, EventBroker())
    _, _, _, metadata = manager._civitai_download_sources(request, plan)
    assert metadata.get("instruction_edit_capability") == (
        "declared" if declaration == "declared" else "unknown"
    )
