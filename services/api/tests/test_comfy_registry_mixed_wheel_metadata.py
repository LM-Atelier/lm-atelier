from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from sqlalchemy.orm import Session
from test_comfy_registry_reviewed_wheel_metadata import _HEADERS, _review_metadata
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_source_artifacts import source_review_context as source_review_context
from test_comfy_registry_wheel_metadata import _environment, _inputs

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    build_comfy_registry_wheel_input_manifest,
    reviewed_wheel_input,
)
from local_lm.comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataError,
    ComfyRegistryWheelMetadataPlan,
    plan_comfy_registry_wheel_metadata,
)
from local_lm.comfy_registry_wheel_selection import validate_comfy_registry_wheel_metadata_plan


def _mixed(
    context: tuple[Session, ArtifactStore],
    local_dependencies: list[str],
    remote_entries: list[tuple[str, str, str, list[str], bool]],
) -> tuple[ComfyRegistryWheelInputManifest, dict[str, bytes]]:
    session, store = context
    content = (
        _HEADERS
        + b"".join(f"Requires-Dist: {requirement}\n".encode() for requirement in local_dependencies)
        + b"\n"
    )
    _review_metadata(session, store, content)
    local = reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    remote, documents = _inputs(remote_entries)
    documents[local.filename] = content
    return build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local]), documents


def _plan(
    manifest: ComfyRegistryWheelInputManifest,
    documents: dict[str, bytes],
    *,
    environment: dict[str, str] | None = None,
    tags: tuple[str, ...] = ("py3-none-any",),
    runtime: dict[str, str] | None = None,
) -> ComfyRegistryWheelMetadataPlan:
    from local_lm.comfy_registry_mixed_wheel_metadata import (
        plan_comfy_registry_mixed_wheel_metadata,
    )

    return plan_comfy_registry_mixed_wheel_metadata(
        manifest,
        documents,
        marker_environment=environment or _environment(),
        supported_tags=tags,
        runtime_distributions=runtime or {},
    )


def test_cross_source_extras_reach_a_fixed_point(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, documents = _mixed(
        source_review_context,
        ["helper[fast]>=2", 'leaf>=3; extra == "deep"', 'unused; python_version < "3"'],
        [("helper", "2.0", "helper==2.0", ['example-pkg[deep]>=1; extra == "fast"'], True)],
    )
    plan = _plan(manifest, documents)
    frontier = {item.name: item for item in plan.frontier}
    assert set(frontier) == {"example-pkg", "helper", "leaf"}
    assert frontier["helper"].status == frontier["example-pkg"].status == "satisfied"
    assert frontier["helper"].requested_extras == ("fast",)
    assert frontier["example-pkg"].requested_extras == ("deep",)
    assert frontier["leaf"].status == "resolve"
    assert plan.resolution_required and not plan.conflicts
    assert plan.artifact_manifest_sha256 == manifest.manifest_sha256
    assert validate_comfy_registry_wheel_metadata_plan(plan) == plan.plan_sha256


@pytest.mark.parametrize("dependency", ["helper<2", "helper>=3"])
def test_local_requirements_conflict_with_a_remote_lock(
    source_review_context: tuple[Session, ArtifactStore],
    dependency: str,
) -> None:
    manifest, documents = _mixed(
        source_review_context,
        [dependency],
        [
            ("helper", "2.0", "helper==2.0", [], True),
        ],
    )
    plan = _plan(manifest, documents)
    assert plan.conflicts == ("helper",)
    assert plan.frontier[0].locked_version == "2.0"
    assert plan.frontier[0].status == "conflict"


def test_remote_requirement_conflicts_with_a_reviewed_local_lock(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, documents = _mixed(
        source_review_context,
        [],
        [
            ("helper", "2.0", "helper==2.0", ["example-pkg>=2"], True),
        ],
    )
    plan = _plan(manifest, documents)
    assert plan.conflicts == ("example-pkg",)
    assert plan.frontier[0].locked_version == "1.2.3"


def test_runtime_can_satisfy_a_local_dependency(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    plan = _plan(manifest, documents, runtime={"helper": "2.1"})
    assert not plan.resolution_required
    assert plan.frontier[0].status == "satisfied"
    assert plan.frontier[0].locked_version == "2.1"


@pytest.mark.parametrize("change", ["missing", "extra", "changed"])
def test_metadata_documents_must_match_every_selected_input(
    source_review_context: tuple[Session, ArtifactStore],
    change: str,
) -> None:
    manifest, documents = _mixed(source_review_context, [], [])
    filename = manifest.reviewed_local[0].filename
    code = "core_metadata_hash_mismatch"
    if change == "missing":
        documents.pop(filename)
        code = "missing_core_metadata"
    elif change == "extra":
        documents["unselected.whl"] = b"unselected"
        code = "unexpected_core_metadata"
    else:
        documents[filename] += b"changed"
    with pytest.raises(ComfyRegistryWheelMetadataError) as caught:
        _plan(manifest, documents)
    assert caught.value.code == code


@pytest.mark.parametrize("change", ["environment", "tags"])
def test_mixed_planning_refuses_a_different_target(
    source_review_context: tuple[Session, ArtifactStore],
    change: str,
) -> None:
    manifest, documents = _mixed(source_review_context, [], [])
    environment = _environment(python_full_version="3.13.1", python_version="3.13")
    with pytest.raises(ComfyRegistryWheelInputError) as caught:
        _plan(
            manifest,
            documents,
            environment=environment if change == "environment" else None,
            tags=("py2-none-any",) if change == "tags" else ("py3-none-any",),
        )
    assert caught.value.code == "wheel_input_target_mismatch"


@pytest.mark.parametrize("change", ["review", "declaration", "root_extras"])
def test_equal_metadata_keeps_distinct_review_and_declaration_identity(
    source_review_context: tuple[Session, ArtifactStore],
    change: str,
) -> None:
    manifest, documents = _mixed(source_review_context, ['leaf; extra == "extra"'], [])
    local = manifest.reviewed_local[0]
    first = _plan(manifest, documents)
    if change == "review":
        local = replace(local, review_sha256="e" * 64)
    else:
        declaration = local.declaration.replace(local.commit, "d" * 40)
        commit = "d" * 40
        if change == "root_extras":
            declaration = local.declaration.replace("example-pkg @", "example-pkg[extra] @")
            commit = local.commit
        local = replace(
            local,
            declaration=declaration,
            commit=commit,
            declaration_sha256=hashlib.sha256(declaration.encode()).hexdigest(),
        )
    changed = build_comfy_registry_wheel_input_manifest("f" * 64, manifest.remote, [local])
    second = _plan(changed, documents)
    assert first.plan_sha256 != second.plan_sha256
    assert second.artifact_manifest_sha256 == changed.manifest_sha256
    if change == "root_extras":
        assert not first.frontier
        assert second.frontier[0].name == "leaf"
    else:
        assert first.requirements == second.requirements
        assert first.frontier == second.frontier


def test_remote_only_adapter_preserves_frontier_and_changes_only_source_binding() -> None:
    remote, documents = _inputs(
        [
            ("helper", "2.0", "helper==2.0", ["leaf>=1"], True),
        ]
    )
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, remote, [])
    existing = plan_comfy_registry_wheel_metadata(
        remote, documents, marker_environment=_environment()
    )
    mixed = _plan(manifest, documents)
    assert mixed == replace(
        existing, artifact_manifest_sha256=manifest.manifest_sha256, plan_sha256=mixed.plan_sha256
    )
    assert mixed.plan_sha256 != existing.plan_sha256
    assert validate_comfy_registry_wheel_metadata_plan(mixed) == mixed.plan_sha256


def test_missing_remote_metadata_remains_unresolved_with_local_inputs(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, documents = _mixed(
        source_review_context,
        [],
        [
            ("helper", "2.0", "helper==2.0", [], False),
        ],
    )
    plan = _plan(manifest, documents)
    assert plan.unavailable_metadata == ("helper-2.0-py3-none-any.whl",)
    assert plan.resolution_required


def test_remote_planning_preserves_a_valid_manifest_digest_spelling() -> None:
    manifest, documents = _inputs()
    manifest = replace(manifest, manifest_sha256=manifest.manifest_sha256.upper())
    plan = plan_comfy_registry_wheel_metadata(
        manifest, documents, marker_environment=_environment()
    )
    assert plan.artifact_manifest_sha256 == manifest.manifest_sha256
