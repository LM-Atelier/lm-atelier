from __future__ import annotations

from typing import Any

import pytest

from local_lm.comfy_workflow_packages import WorkflowAssetReference
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.model_planner import INSTALL_RESOLVER_VERSION
from local_lm.models import InstallPlan
from local_lm.workflow_asset_aliases import (
    WorkflowAssetAliasError,
    materialize_workflow_asset_aliases,
)
from local_lm.workflow_asset_bindings import WorkflowAssetPlanSelection
from local_lm.workflow_asset_downloads import install_plan_download_request

DIGEST = "a" * 64


@pytest.fixture(autouse=True)
def _configured_database(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()


def _reference(filename: str) -> WorkflowAssetReference:
    return WorkflowAssetReference(
        filename=filename,
        suffix=".safetensors",
        policy="supported",
        kind="checkpoint",
    )


def _plan(
    *,
    plan_id: str,
    provider: str = "civitai",
    path: str = "provider-model.safetensors",
    artifact_overrides: dict[str, Any] | None = None,
    plan_overrides: dict[str, Any] | None = None,
) -> InstallPlan:
    artifact: dict[str, Any] = {
        "path": path,
        "kind": "checkpoint",
        "target_folder": "checkpoints",
        "size_bytes": 19,
        "sha256": DIGEST,
        "required": True,
        "reuse": "download",
        "source_version_id": "202" if provider == "civitai" else None,
        "source_file_id": "301" if provider == "civitai" else None,
    }
    artifact.update(artifact_overrides or {})
    values: dict[str, Any] = {
        "id": plan_id,
        "provider": provider,
        "remote_id": "101" if provider == "civitai" else "author/model",
        "revision": "202" if provider == "civitai" else "b" * 40,
        "role": "image",
        "engine": "comfyui",
        "plan_hash": ("c" if provider == "civitai" else "d") * 64,
        "resolver_version": INSTALL_RESOLVER_VERSION,
        "compatibility": "supported",
        "artifacts_json": [artifact],
        "runtime_contract_json": {},
        "activation_probe_json": {},
        "status": "planned",
    }
    values.update(plan_overrides or {})
    return InstallPlan(**values)


def test_civitai_alias_preserves_exact_source_and_is_stable() -> None:
    expected = "workflow-checkpoint.safetensors"
    source = _plan(plan_id="plan_source")
    with SessionLocal() as session:
        session.add(source)
        session.flush()
        selections, plans = materialize_workflow_asset_aliases(
            session,
            [_reference(expected)],
            [
                WorkflowAssetPlanSelection(
                    expected,
                    source.id,
                    "provider-model.safetensors",
                )
            ],
            {source.id: source},
        )
        (selection,) = selections
        derived = plans[selection.install_plan_id]
        first_id = derived.id
        assert selection.artifact_path == expected
        assert derived.artifacts_json == [
            {
                "path": expected,
                "kind": "checkpoint",
                "target_folder": "checkpoints",
                "size_bytes": 19,
                "sha256": DIGEST,
                "required": True,
                "reuse": "download",
                "source_remote_id": None,
                "source_revision": None,
                "source_path": None,
                "source_version_id": "202",
                "source_file_id": "301",
            }
        ]
        runtime = derived.runtime_contract_json
        assert runtime["workflow_asset_kind"] == "checkpoint"
        assert runtime["workflow_component_folders"] == {expected: "checkpoints"}
        assert runtime["workflow_asset_alias"] == {
            "version": 1,
            "source_plan_hash": source.plan_hash,
            "source_artifact_path": "provider-model.safetensors",
            "destination_path": expected,
        }
        request = install_plan_download_request(derived)
        assert request.allow_patterns == [expected]
        assert request.file_sources == {}
        assert request.workflow_asset_kind == "checkpoint"

        repeated, repeated_plans = materialize_workflow_asset_aliases(
            session,
            [_reference(expected)],
            [
                WorkflowAssetPlanSelection(
                    expected,
                    source.id,
                    "provider-model.safetensors",
                )
            ],
            {source.id: source},
        )
        assert repeated[0].install_plan_id == first_id
        assert repeated_plans[first_id].plan_hash == derived.plan_hash


def test_hugging_face_alias_maps_destination_to_exact_provider_source() -> None:
    source = _plan(plan_id="plan_hf", provider="huggingface")
    expected = "models/workflow-name.safetensors"
    with SessionLocal() as session:
        session.add(source)
        session.flush()
        selections, plans = materialize_workflow_asset_aliases(
            session,
            [_reference(expected)],
            [
                WorkflowAssetPlanSelection(
                    expected,
                    source.id,
                    "provider-model.safetensors",
                )
            ],
            {source.id: source},
        )
        request = install_plan_download_request(plans[selections[0].install_plan_id])

    mapped = request.file_sources[expected]
    assert mapped.remote_id == "author/model"
    assert mapped.revision == "b" * 40
    assert mapped.filename == "provider-model.safetensors"
    assert mapped.sha256 == DIGEST


def test_exact_filename_keeps_the_original_plan() -> None:
    expected = "provider-model.safetensors"
    source = _plan(plan_id="plan_exact")
    selection = WorkflowAssetPlanSelection(expected, source.id, expected)
    with SessionLocal() as session:
        session.add(source)
        session.flush()
        selections, plans = materialize_workflow_asset_aliases(
            session,
            [_reference(expected)],
            [selection],
            {source.id: source},
        )

    assert selections == (selection,)
    assert set(plans) == {source.id}


def test_one_provider_artifact_cannot_back_two_runtime_names() -> None:
    source = _plan(plan_id="plan_duplicate")
    with SessionLocal() as session:
        session.add(source)
        session.flush()
        with pytest.raises(WorkflowAssetAliasError) as caught:
            materialize_workflow_asset_aliases(
                session,
                [_reference("first.safetensors"), _reference("second.safetensors")],
                [
                    WorkflowAssetPlanSelection(
                        "first.safetensors", source.id, "provider-model.safetensors"
                    ),
                    WorkflowAssetPlanSelection(
                        "second.safetensors", source.id, "provider-model.safetensors"
                    ),
                ],
                {source.id: source},
            )

    assert caught.value.code == "duplicate_alias_source"


def test_incomplete_legacy_source_plan_refuses_before_deriving() -> None:
    source = _plan(
        plan_id="plan_legacy",
        artifact_overrides={"source_file_id": None},
    )
    with SessionLocal() as session:
        session.add(source)
        session.flush()
        with pytest.raises(WorkflowAssetAliasError) as caught:
            materialize_workflow_asset_aliases(
                session,
                [_reference("workflow-name.safetensors")],
                [
                    WorkflowAssetPlanSelection(
                        "workflow-name.safetensors",
                        source.id,
                        "provider-model.safetensors",
                    )
                ],
                {source.id: source},
            )

    assert caught.value.code == "incomplete_civitai_provenance"


def test_malformed_runtime_contract_refuses_before_deriving() -> None:
    source = _plan(
        plan_id="plan_bad_runtime",
        plan_overrides={"runtime_contract_json": ["not", "a", "mapping"]},
    )
    with SessionLocal() as session:
        session.add(source)
        session.flush()
        with pytest.raises(WorkflowAssetAliasError) as caught:
            materialize_workflow_asset_aliases(
                session,
                [_reference("workflow-name.safetensors")],
                [
                    WorkflowAssetPlanSelection(
                        "workflow-name.safetensors",
                        source.id,
                        "provider-model.safetensors",
                    )
                ],
                {source.id: source},
            )

    assert caught.value.code == "invalid_install_plan"
