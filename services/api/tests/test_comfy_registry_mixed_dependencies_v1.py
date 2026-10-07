from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from sqlalchemy.orm import Session
from test_comfy_registry_source_artifacts import DECLARATION, _artifact
from test_comfy_registry_source_artifacts import source_review_context as source_review_context

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_dependencies import (
    ComfyRegistryDependencyError,
    plan_comfy_registry_dependencies,
)
from local_lm.comfy_registry_source_artifacts import (
    record_local_source_artifact_review,
    verified_reviewed_source_wheel,
)

SOURCE = "Local_Package[SECOND,First] @ git+https://github.com/example/package@" + "a" * 40


@pytest.mark.parametrize(
    "declarations",
    [
        [],
        ["# comment", "  "],
        ["Remote_Name[EXTRA]>=1; python_version >= '3.12'"],
        ["zeta==1", "alpha~=2.0"],
        ["sample===vendor-build-7"],
    ],
)
def test_remote_only_plans_preserve_existing_requirements_and_hashes(
    declarations: list[str],
) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    old = plan_comfy_registry_dependencies(declarations)
    plan = plan_comfy_registry_mixed_dependencies(declarations)
    assert plan.remote == old
    assert plan.declaration_sha256 == old.declaration_sha256
    assert plan.sources == ()


def test_source_declaration_matches_the_existing_review_reader(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    session, store = source_review_context
    artifact = _artifact(session, store)
    record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    wheel = verified_reviewed_source_wheel(session, store, declaration=DECLARATION)
    plan = plan_comfy_registry_mixed_dependencies(["  " + DECLARATION + "  # reviewed wheel"])
    source = plan.sources[0]
    assert source.declaration == wheel.declaration
    assert source.repository == wheel.repository
    assert source.commit == wheel.commit
    assert source.name == wheel.distribution
    assert source.declaration_sha256 == hashlib.sha256(wheel.declaration.encode()).hexdigest()


def test_mixed_identity_preserves_source_spelling_and_canonical_target() -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    marker = ' ; sys_platform == "win32"'
    plan = plan_comfy_registry_mixed_dependencies([SOURCE + marker, "remote>=2"])
    source = plan.sources[0]
    assert source.name == "local-package"
    assert source.extras == ("first", "second")
    assert source.marker == 'sys_platform == "win32"'
    assert source.repository == "example/package"
    assert source.commit == "a" * 40
    assert source.declaration.startswith("Local_Package[First,SECOND] @ ")
    assert plan.remote == plan_comfy_registry_dependencies(["remote>=2"])
    assert plan.declaration_sha256 != plan.remote.declaration_sha256


def test_order_and_requirement_comments_do_not_change_the_mixed_identity() -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    other = SOURCE.replace("Local_Package", "Other")
    first = plan_comfy_registry_mixed_dependencies([SOURCE, "remote==2", other])
    second = plan_comfy_registry_mixed_dependencies(
        ["# requirements", other + " # local", "remote==2 # remote", SOURCE, ""]
    )
    assert first == second
    assert [source.name for source in first.sources] == ["local-package", "other"]


@pytest.mark.parametrize(
    "changed",
    [
        SOURCE.replace("a" * 40, "b" * 40),
        SOURCE.replace("example/package", "example/other"),
        SOURCE.replace("SECOND,First", "First"),
        SOURCE + ' ; sys_platform == "linux"',
        SOURCE.replace("Local_Package", "local-package"),
    ],
)
def test_changed_source_identity_cannot_reuse_the_declaration_hash(changed: str) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    assert (
        plan_comfy_registry_mixed_dependencies([SOURCE]).declaration_sha256
        != plan_comfy_registry_mixed_dependencies([changed]).declaration_sha256
    )


@pytest.mark.parametrize("other", ["local-package==2", SOURCE, SOURCE.replace("a" * 40, "b" * 40)])
def test_duplicate_targets_are_refused_across_both_origins(other: str) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    with pytest.raises(ComfyRegistryDependencyError) as error:
        plan_comfy_registry_mixed_dependencies([SOURCE, other])
    assert error.value.code == "ambiguous_dependency"


def test_distinct_environment_targets_remain_bound_even_when_inactive() -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    declarations = [
        SOURCE + ' ; sys_platform == "win32"',
        'local-package==2; sys_platform == "linux"',
    ]
    plan = plan_comfy_registry_mixed_dependencies(declarations)
    assert len(plan.sources) == len(plan.remote.dependencies) == 1
    assert (
        plan.declaration_sha256
        != plan_comfy_registry_mixed_dependencies(declarations[:1]).declaration_sha256
    )


@pytest.mark.parametrize(
    ("declaration", "code"),
    [
        (SOURCE.replace("a" * 40, "main"), "unpinned_source_dependency"),
        (SOURCE.replace("a" * 40, "a" * 12), "unpinned_source_dependency"),
        (SOURCE.replace("github.com", "example.com"), "direct_dependency_url"),
        ("local @ file:///tmp/local.whl", "direct_dependency_url"),
        ("local @ https://example.com/local.whl", "direct_dependency_url"),
        (SOURCE.split(" @ ")[1], "unresolved_source_dependency"),
        ("--index-url https://example.com", "dependency_option_unsupported"),
        ("-r requirements.txt", "dependency_option_unsupported"),
        (SOURCE + "\nremote==2", "invalid_dependency"),
        (SOURCE + "\x7f", "invalid_dependency"),
        ("not a requirement!", "invalid_dependency"),
    ],
)
def test_unsupported_requirements_keep_precise_refusals(declaration: str, code: str) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    with pytest.raises(ComfyRegistryDependencyError) as error:
        plan_comfy_registry_mixed_dependencies([declaration])
    assert error.value.code == code


def test_source_planning_does_not_enable_the_installers_remote_planner() -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    assert len(plan_comfy_registry_mixed_dependencies([SOURCE]).sources) == 1
    with pytest.raises(ComfyRegistryDependencyError) as error:
        plan_comfy_registry_dependencies([SOURCE])
    assert error.value.code == "unresolved_source_dependency"


def test_combined_declaration_count_includes_both_origins_and_comments() -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    with pytest.raises(ComfyRegistryDependencyError) as error:
        plan_comfy_registry_mixed_dependencies([SOURCE, "remote==1"] + ["# comment"] * 255)
    assert error.value.code == "too_many_dependencies"


def test_source_extras_are_bounded() -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    declaration = SOURCE.replace("SECOND,First", ",".join(f"extra{i}" for i in range(33)))
    with pytest.raises(ComfyRegistryDependencyError) as error:
        plan_comfy_registry_mixed_dependencies([declaration])
    assert error.value.code == "too_many_dependency_extras"


def test_complete_source_identity_is_subject_to_the_plan_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_mixed_dependencies_v1 as mixed

    monkeypatch.setattr(mixed, "MAX_REGISTRY_DEPENDENCY_PLAN_BYTES", 100)
    with pytest.raises(ComfyRegistryDependencyError) as error:
        mixed.plan_comfy_registry_mixed_dependencies([SOURCE])
    assert error.value.code == "dependency_plan_too_large"


@pytest.mark.parametrize(
    "field", ["commit", "declaration_sha256", "repository", "name", "marker", "extras"]
)
def test_validation_rejects_changed_source_evidence(field: str) -> None:
    from local_lm import comfy_registry_mixed_dependencies_v1 as mixed

    plan = mixed.plan_comfy_registry_mixed_dependencies([SOURCE, "remote==2"])
    source = replace(plan.sources[0])
    object.__setattr__(source, field, ("changed",) if field == "extras" else "changed")
    with pytest.raises(ComfyRegistryDependencyError) as error:
        mixed.validate_comfy_registry_mixed_dependency_plan(replace(plan, sources=(source,)))
    assert error.value.code == "invalid_mixed_dependency_plan"


def test_validation_rejects_changed_remote_evidence_and_missing_sources() -> None:
    from local_lm import comfy_registry_mixed_dependencies_v1 as mixed

    plan = mixed.plan_comfy_registry_mixed_dependencies([SOURCE, "remote==2"])
    assert mixed.validate_comfy_registry_mixed_dependency_plan(plan) == plan
    for changed in (
        replace(plan, remote=plan_comfy_registry_dependencies(["remote==3"])),
        replace(plan, sources=()),
        replace(plan, declaration_sha256="0" * 64),
    ):
        with pytest.raises(ComfyRegistryDependencyError) as error:
            mixed.validate_comfy_registry_mixed_dependency_plan(changed)
        assert error.value.code == "invalid_mixed_dependency_plan"
