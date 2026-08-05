from __future__ import annotations

import pytest

import local_lm.comfy_registry_dependencies as dependency_module
from local_lm.comfy_registry_dependencies import (
    MAX_REGISTRY_PIP_DEPENDENCIES,
    ComfyRegistryDependencyError,
    plan_comfy_registry_dependencies,
)


def _assert_error(code: str, declarations: object) -> None:
    with pytest.raises(ComfyRegistryDependencyError) as raised:
        plan_comfy_registry_dependencies(declarations)  # type: ignore[arg-type]
    assert raised.value.code == code


def test_empty_dependency_plan_is_stable_and_needs_no_resolution() -> None:
    first = plan_comfy_registry_dependencies([])
    second = plan_comfy_registry_dependencies(())

    assert first == second
    assert first.dependencies == ()
    assert len(first.declaration_sha256) == 64
    assert first.version_resolution_required is False
    assert first.artifact_resolution_required is False


def test_canonicalizes_exact_pins_extras_and_markers() -> None:
    plan = plan_comfy_registry_dependencies(
        ["Example_Package[Second,First]==1.2.3 ; sys_platform == 'win32'"]
    )

    assert plan.version_resolution_required is False
    assert plan.artifact_resolution_required is True
    assert plan.dependencies == (
        dependency_module.ComfyRegistryDependency(
            name="example-package",
            requirement='example-package[first,second]==1.2.3; sys_platform == "win32"',
            marker='sys_platform == "win32"',
            extras=("first", "second"),
            pinned_version="1.2.3",
            version_resolution_required=False,
        ),
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "example",
        "example>=1.0",
        "example>=1,<3",
        "example~=2.1",
        "example==2.*",
        "example!=2.0",
    ],
)
def test_unresolved_constraints_remain_inert(declaration: str) -> None:
    plan = plan_comfy_registry_dependencies([declaration])

    assert plan.version_resolution_required is True
    assert plan.dependencies[0].pinned_version is None
    assert plan.dependencies[0].version_resolution_required is True


def test_arbitrary_exact_equality_is_classified_as_pinned() -> None:
    plan = plan_comfy_registry_dependencies(["example===vendor-build-7"])

    assert plan.dependencies[0].pinned_version == "vendor-build-7"
    assert plan.version_resolution_required is False


def test_input_order_does_not_change_manifest_identity() -> None:
    first = plan_comfy_registry_dependencies(["zeta==2", "alpha>=1"])
    second = plan_comfy_registry_dependencies(["alpha>=1", "zeta==2"])

    assert first == second
    assert [item.name for item in first.dependencies] == ["alpha", "zeta"]


def test_same_package_can_have_distinct_environment_targets() -> None:
    plan = plan_comfy_registry_dependencies(
        [
            "runtime==1; sys_platform == 'win32'",
            "runtime==2; sys_platform == 'linux'",
        ]
    )

    assert len(plan.dependencies) == 2
    assert plan.version_resolution_required is False


def test_duplicate_environment_target_is_ambiguous() -> None:
    _assert_error("ambiguous_dependency", ["runtime>=1", "Runtime<3"])


@pytest.mark.parametrize(
    "declaration",
    [
        "example @ https://example.com/example.whl",
        "example @ file:///tmp/example.whl",
        "example @ git+https://gitlab.com/example/example.git@main",
    ],
)
def test_rejects_direct_local_and_vcs_urls(declaration: str) -> None:
    _assert_error("direct_dependency_url", [declaration])


@pytest.mark.parametrize(
    ("declaration", "code"),
    [
        ("example @ git+https://github.com/example/example.git@main", "unpinned_source_dependency"),
        ("example @ git+https://github.com/example/example", "unpinned_source_dependency"),
        (
            "example @ git+https://github.com/example/example@" + "a" * 40,
            "unresolved_source_dependency",
        ),
    ],
)
def test_an_allowed_source_host_is_refused_by_what_would_fix_it(
    declaration: str, code: str
) -> None:
    """Still refused, every one. The code is what tells them apart.

    A branch cannot be made exact by any later machinery, so the package has
    to pin it; an exact commit is refused only until its source can be bound
    to a reviewed artifact. One blanket code said neither.
    """
    _assert_error(code, [declaration])


@pytest.mark.parametrize(
    "declarations",
    [
        "example==1",
        ["example==1\nother==2"],
        [123],
    ],
)
def test_rejects_invalid_dependency_declarations(declarations: object) -> None:
    expected = "invalid_dependency_list" if isinstance(declarations, str) else "invalid_dependency"
    _assert_error(expected, declarations)


@pytest.mark.parametrize(
    "declaration",
    ["-r requirements.txt", "-e .", "--index-url https://example.com/simple"],
)
def test_rejects_installer_options_with_their_own_code(declaration: str) -> None:
    """Dropping one of these would install something other than what was declared."""
    _assert_error("dependency_option_unsupported", [declaration])


@pytest.mark.parametrize(
    "declarations",
    [
        [""],
        ["   "],
        ["# pinned for the CUDA build"],
        ["  # indented comment"],
    ],
)
def test_ignores_what_a_requirements_file_carries_but_does_not_declare(
    declarations: list[str],
) -> None:
    """Publishers dump a requirements file here, furniture and all."""
    plan = plan_comfy_registry_dependencies(declarations)

    assert plan.dependencies == ()
    assert plan.artifact_resolution_required is False


def test_an_inline_comment_does_not_make_a_requirement_invalid() -> None:
    plan = plan_comfy_registry_dependencies(["numpy>=1.26  # needed by the sampler", ""])

    assert [dependency.requirement for dependency in plan.dependencies] == ["numpy>=1.26"]


def test_rejects_too_many_dependencies() -> None:
    declarations = [f"package-{index}==1" for index in range(MAX_REGISTRY_PIP_DEPENDENCIES + 1)]

    _assert_error("too_many_dependencies", declarations)


def test_rejects_too_many_extras() -> None:
    extras = ",".join(f"extra{index}" for index in range(33))

    _assert_error("too_many_dependency_extras", [f"example[{extras}]==1"])


def test_rejects_oversized_canonical_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dependency_module, "MAX_REGISTRY_DEPENDENCY_PLAN_BYTES", 10)

    _assert_error("dependency_plan_too_large", ["example==1"])
