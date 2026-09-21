from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.orm import Session
from test_comfy_registry_mixed_wheel_metadata import _mixed
from test_comfy_registry_source_artifacts import source_review_context as source_review_context
from test_comfy_registry_wheel_closure import _project
from test_comfy_registry_wheel_metadata import _environment

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_wheel_closure import ComfyRegistryWheelClosureError
from local_lm.comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputManifest,
    build_comfy_registry_wheel_input_manifest,
)
from local_lm.comfy_registry_wheel_selection import select_comfy_registry_wheel_versions

if TYPE_CHECKING:
    from local_lm.comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure


def _start(
    manifest: ComfyRegistryWheelInputManifest,
    documents: dict[str, bytes],
    *,
    runtime: dict[str, str] | None = None,
) -> ComfyRegistryMixedWheelClosure:
    from local_lm.comfy_registry_mixed_wheel_closure import plan_comfy_registry_mixed_wheel_closure

    return plan_comfy_registry_mixed_wheel_closure(
        manifest,
        documents,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions=runtime or {},
    )


def test_transitive_rounds_keep_source_review_and_activate_local_extras(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import (
        advance_comfy_registry_mixed_wheel_closure,
        validate_comfy_registry_mixed_wheel_closure,
    )

    manifest, documents = _mixed(
        source_review_context, ["beta>=1", 'alpha>=3; extra == "deep"'], []
    )
    first = _start(manifest, documents)
    assert first.pending_projects == ("beta",) and first.round_number == 0
    beta, beta_documents = _project("beta", [("2.0", ["example-pkg[deep]>=1"])])
    selection = select_comfy_registry_wheel_versions(
        first.metadata_plan,
        {"beta": beta},
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    documents.update(beta_documents)
    second = advance_comfy_registry_mixed_wheel_closure(
        first,
        selection,
        documents,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    assert second.pending_projects == ("alpha",) and second.round_number == 1
    alpha, alpha_documents = _project("alpha", [("3.0", [])])
    selection = select_comfy_registry_wheel_versions(
        second.metadata_plan,
        {"alpha": alpha},
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    documents.update(alpha_documents)
    final = advance_comfy_registry_mixed_wheel_closure(
        second,
        selection,
        documents,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    assert final.complete and final.pending_projects == () and final.round_number == 2
    assert final.manifest.reviewed_local == manifest.reviewed_local
    assert final.manifest.declaration_sha256 == manifest.declaration_sha256
    assert final.manifest.remote.declaration_sha256 == manifest.remote.declaration_sha256
    assert [item.name for item in final.manifest.remote.artifacts] == ["alpha", "beta"]
    assert final.manifest_history == tuple(
        item.manifest.manifest_sha256 for item in (first, second, final)
    )
    assert len({item.closure_sha256 for item in (first, second, final)}) == 3
    assert validate_comfy_registry_mixed_wheel_closure(final) == final.manifest


@pytest.mark.parametrize("runtime", [{"helper": "2.0"}, {"helper": "2.0", "unrelated": "1.0"}])
def test_runtime_baseline_completes_a_local_dependency_and_binds_identity(
    source_review_context: tuple[Session, ArtifactStore],
    runtime: dict[str, str],
) -> None:
    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    first = _start(manifest, documents, runtime=runtime)
    second = _start(manifest, documents, runtime={**runtime, "other": "1.0"})
    assert first.complete and second.complete
    assert first.manifest == second.manifest
    assert first.closure_sha256 != second.closure_sha256


@pytest.mark.parametrize("problem", ["conflict", "unavailable"])
def test_incomplete_or_conflicting_metadata_cannot_form_a_closure(
    source_review_context: tuple[Session, ArtifactStore],
    problem: str,
) -> None:
    manifest, documents = _mixed(
        source_review_context,
        ["helper>=3"],
        [
            ("helper", "2.0", "helper==2.0", [], problem != "unavailable"),
        ],
    )
    with pytest.raises(ComfyRegistryWheelClosureError) as caught:
        _start(manifest, documents)
    assert caught.value.code == (
        "dependency_conflict" if problem == "conflict" else "metadata_unavailable"
    )


@pytest.mark.parametrize("change", ["review", "selection_target", "planning_target"])
def test_advance_rejects_selection_or_target_from_another_context(
    source_review_context: tuple[Session, ArtifactStore],
    change: str,
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import (
        advance_comfy_registry_mixed_wheel_closure,
    )

    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    first = _start(manifest, documents)
    project, extra = _project("helper", [("2.0", [])])
    other_environment = _environment(python_version="3.13", python_full_version="3.13.1")
    selection = select_comfy_registry_wheel_versions(
        first.metadata_plan,
        {"helper": project},
        marker_environment=other_environment if change == "selection_target" else _environment(),
        supported_tags=("py3-none-any",),
    )
    if change == "review":
        local = replace(manifest.reviewed_local[0], review_sha256="e" * 64)
        changed = build_comfy_registry_wheel_input_manifest("f" * 64, manifest.remote, [local])
        first = _start(changed, documents)
    with pytest.raises(ComfyRegistryWheelClosureError) as caught:
        advance_comfy_registry_mixed_wheel_closure(
            first,
            selection,
            {**documents, **extra},
            marker_environment=other_environment if change == "planning_target" else _environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == (
        "wheel_input_target_mismatch"
        if change == "planning_target"
        else "selection_source_mismatch"
    )


def test_advancing_rechecks_existing_local_metadata(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import (
        advance_comfy_registry_mixed_wheel_closure,
    )

    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    first = _start(manifest, documents)
    project, extra = _project("helper", [("2.0", [])])
    selection = select_comfy_registry_wheel_versions(
        first.metadata_plan,
        {"helper": project},
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    documents[manifest.reviewed_local[0].filename] += b"changed"
    with pytest.raises(ComfyRegistryWheelClosureError) as caught:
        advance_comfy_registry_mixed_wheel_closure(
            first,
            selection,
            {**documents, **extra},
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == "core_metadata_hash_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("round_number", True),
        ("round_number", 2),
        ("complete", True),
        ("complete", 0),
        ("pending_projects", ()),
        ("manifest_history", ()),
        ("manifest_history", ("e" * 64, "e" * 64)),
        ("closure_sha256", "0" * 64),
        ("runtime_distributions", []),
    ],
)
def test_closure_validation_refuses_inconsistent_or_replaced_state(
    source_review_context: tuple[Session, ArtifactStore],
    field: str,
    value: object,
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import (
        validate_comfy_registry_mixed_wheel_closure,
    )

    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    closure = _start(manifest, documents)
    changed = replace(closure)
    object.__setattr__(changed, field, value)
    with pytest.raises(ComfyRegistryWheelClosureError):
        validate_comfy_registry_mixed_wheel_closure(changed)


def test_completed_closure_cannot_advance(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import (
        advance_comfy_registry_mixed_wheel_closure,
    )

    manifest, documents = _mixed(source_review_context, [], [])
    closure = _start(manifest, documents)
    selection = select_comfy_registry_wheel_versions(
        closure.metadata_plan,
        {},
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    with pytest.raises(ComfyRegistryWheelClosureError) as caught:
        advance_comfy_registry_mixed_wheel_closure(
            closure,
            selection,
            documents,
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == "closure_already_complete"


def test_round_limit_stops_a_growing_dependency_chain(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_mixed_wheel_closure as module

    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    closure = _start(manifest, documents)
    project, extra = _project("helper", [("2.0", [])])
    selection = select_comfy_registry_wheel_versions(
        closure.metadata_plan,
        {"helper": project},
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    monkeypatch.setattr(module, "MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS", 0)
    with pytest.raises(ComfyRegistryWheelClosureError) as caught:
        module.advance_comfy_registry_mixed_wheel_closure(
            closure,
            selection,
            {**documents, **extra},
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == "closure_round_limit"


@pytest.mark.parametrize("selected_names", [(), ("beta",)])
def test_advance_requires_the_entire_exact_pending_frontier(
    source_review_context: tuple[Session, ArtifactStore],
    selected_names: tuple[str, ...],
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import (
        advance_comfy_registry_mixed_wheel_closure,
    )
    from local_lm.comfy_registry_wheel_selection import validate_comfy_registry_wheel_selection

    manifest, documents = _mixed(source_review_context, ["alpha>=1", "beta>=2"], [])
    closure = _start(manifest, documents)
    alpha, alpha_metadata = _project("alpha", [("1.0", [])])
    beta, beta_metadata = _project("beta", [("2.0", [])])
    selection = select_comfy_registry_wheel_versions(
        closure.metadata_plan,
        {"alpha": alpha, "beta": beta},
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    selection = replace(
        selection,
        artifacts=tuple(item for item in selection.artifacts if item.name in selected_names),
    )
    payload = {
        "version": 1,
        "artifact_manifest_sha256": selection.artifact_manifest_sha256,
        "metadata_plan_sha256": selection.metadata_plan_sha256,
        "target_sha256": selection.target_sha256,
        "artifacts": [asdict(item) for item in selection.artifacts],
    }
    selection = replace(
        selection,
        selection_sha256=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    )
    assert validate_comfy_registry_wheel_selection(selection) == selection.artifacts
    with pytest.raises(ComfyRegistryWheelClosureError) as caught:
        advance_comfy_registry_mixed_wheel_closure(
            closure,
            selection,
            {**documents, **alpha_metadata, **beta_metadata},
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == "selection_frontier_mismatch"
