"""Completeness guard for modules with reviewed numeric parser ceilings.

The companion tests bound individual constants. This file closes the other
half of that contract: adding a numeric ``MAX_*`` constant to one of the same
modules must fail until a reviewer assigns it a ceiling in the bounds file.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest

from local_lm import (
    artifact_library_schema,
    comfy_registry_archives,
    comfy_registry_wheel_environments,
    comfy_workflow_packages,
    model_manifests,
    workflow_dependencies,
)

TESTS_DIR = Path(__file__).resolve().parent

BOUNDS_FILES: tuple[tuple[str, tuple[ModuleType, ...]], ...] = (
    (
        "test_comfy_registry_archive_bounds.py",
        (comfy_registry_archives,),
    ),
    (
        "test_comfy_registry_wheel_environment_bounds.py",
        (comfy_registry_wheel_environments,),
    ),
    (
        "test_parser_ceiling_bounds.py",
        (
            comfy_workflow_packages,
            workflow_dependencies,
            model_manifests,
            artifact_library_schema,
        ),
    ),
)


def _reviewed_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("MAX_"):
                names.add(node.value)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("MAX_"):
            names.add(node.attr)
    return names


def _numeric_policy_names(module: ModuleType) -> set[str]:
    return {
        name
        for name, value in vars(module).items()
        if name.startswith("MAX_") and type(value) in {int, float}
    }


def test_the_bounds_file_registry_is_present_and_nonempty() -> None:
    assert BOUNDS_FILES, "no ceiling bounds files are registered"
    for filename, modules in BOUNDS_FILES:
        assert modules, f"{filename} has no production module"
        assert (TESTS_DIR / filename).is_file(), f"{filename} is missing"


@pytest.mark.parametrize(
    "filename,modules",
    BOUNDS_FILES,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_numeric_max_constant_is_named_by_its_bounds_file(
    filename: str,
    modules: tuple[ModuleType, ...],
) -> None:
    path = TESTS_DIR / filename
    reviewed = _reviewed_names(path)
    production = set().union(*(_numeric_policy_names(module) for module in modules))

    assert production, f"{filename} covers no numeric MAX_* constants"
    assert reviewed == production, (
        f"{filename} must name every numeric MAX_* constant in its production "
        f"module set; missing={sorted(production - reviewed)!r}, "
        f"stale={sorted(reviewed - production)!r}"
    )
