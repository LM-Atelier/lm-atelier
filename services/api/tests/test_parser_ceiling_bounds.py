"""Twenty-five parser ceilings that nothing would notice moving.

Fourth and last module group in the sweep that began with the repository hygiene
size ceiling and the six resource ceilings in `comfy_registry_source_artifacts`.
Every constant below bounds bytes this application did not write: workflow graph
JSON from a package, a dependency declaration, a downloaded model file's header,
and artifact JSON. Each was multiplied by a thousand against every test module
that imports it, and only `MAX_UI_GRAPH_NODES` failed anything.

That is the defect: not the values, which are sound, but that a later edit
loosening one would go through with the suite green. A depth or node ceiling in
particular is the whole defence against a small hostile document that costs a
great deal to parse, and it is the kind of number somebody raises to make an
awkward input work without registering what it was for.

Bounds rather than pins, as in the sibling files. `<=` lets any of these be
tightened without touching this file, and refuses only the direction that admits
more. The grouping is by module so a failure names the parser it belongs to.
"""

from __future__ import annotations

import pytest

from local_lm import (
    artifact_library_schema,
    comfy_workflow_packages,
    model_manifests,
    workflow_dependencies,
)

MIB = 1024 * 1024

UI_GRAPH = [
    ("MAX_UI_GRAPH_BYTES", 1 * MIB),
    ("MAX_UI_GRAPH_NODES", 4_096),
    ("MAX_UI_GRAPH_LINKS", 16_384),
    ("MAX_UI_GRAPH_SUBGRAPHS", 256),
    ("MAX_UI_GRAPH_VALUES", 100_000),
    ("MAX_UI_GRAPH_DEPTH", 32),
    ("MAX_UI_GRAPH_KEY_CHARACTERS", 200),
    ("MAX_UI_GRAPH_STRING_CHARACTERS", 65_536),
    ("MAX_ASSET_REFERENCE_CHARACTERS", 1_000),
]

DEPENDENCIES = [
    ("MAX_WORKFLOW_DEPENDENCY_SLOTS", 64),
    ("MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS_PER_SLOT", 64),
    ("MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS", 512),
    ("MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES", 256 * 1024),
    ("MAX_WORKFLOW_DEPENDENCY_JSON_NODES", 8_192),
    ("MAX_WORKFLOW_DEPENDENCY_JSON_DEPTH", 12),
    ("MAX_WORKFLOW_DEPENDENCY_JSON_ARRAY_ITEMS", 4_096),
]

MANIFESTS = [
    ("MAX_METADATA_BYTES", 1 * MIB),
    ("MAX_WEIGHT_HEADER_BYTES", 16 * MIB),
    ("MAX_METADATA_NODES", 100_000),
    ("MAX_GGUF_FIELDS", 4_096),
    ("MAX_GGUF_NESTING", 8),
]

LIBRARY_JSON = [
    ("MAX_JSON_BYTES", 1_048_576),
    ("MAX_JSON_NODES", 100_000),
    ("MAX_JSON_DEPTH", 16),
    ("MAX_JSON_MEMBERS", 4_096),
    ("MAX_JSON_TEXT_BYTES", 1_000_000),
]


def _assert_bounded(module: object, name: str, ceiling: int) -> None:
    actual = getattr(module, name)
    assert isinstance(actual, int), f"{name} is {type(actual).__name__}, not an int"
    assert actual > 0, f"{name} is {actual}, which disables the bound it exists to impose"
    assert actual <= ceiling, (
        f"{name} is {actual}, above the reviewed ceiling of {ceiling}. This bounds input "
        "this application did not write. Tightening needs no change here; widening should "
        "be argued for in the same commit so it is read in review rather than inferred."
    )


@pytest.mark.parametrize("name,ceiling", UI_GRAPH)
def test_workflow_graph_ceilings_are_bounded(name: str, ceiling: int) -> None:
    _assert_bounded(comfy_workflow_packages, name, ceiling)


@pytest.mark.parametrize("name,ceiling", DEPENDENCIES)
def test_dependency_declaration_ceilings_are_bounded(name: str, ceiling: int) -> None:
    _assert_bounded(workflow_dependencies, name, ceiling)


@pytest.mark.parametrize("name,ceiling", MANIFESTS)
def test_model_manifest_ceilings_are_bounded(name: str, ceiling: int) -> None:
    _assert_bounded(model_manifests, name, ceiling)


@pytest.mark.parametrize("name,ceiling", LIBRARY_JSON)
def test_artifact_library_json_ceilings_are_bounded(name: str, ceiling: int) -> None:
    _assert_bounded(artifact_library_schema, name, ceiling)


def test_a_per_slot_budget_cannot_exceed_the_whole_budget() -> None:
    """Per-slot and total requirement budgets have to agree about which binds.

    If the per-slot ceiling rose above the total, the total would be unreachable
    through that path and would read as protection while never binding.
    """

    assert (
        workflow_dependencies.MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS_PER_SLOT
        <= workflow_dependencies.MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS
    )


def test_a_single_string_cannot_exceed_the_document_that_holds_it() -> None:
    """Character ceilings have to stay inside the byte ceiling of their document.

    A string ceiling larger than the whole-graph byte ceiling can never be the
    binding check, so raising it would look like a loosened limit while changing
    nothing - and lowering the byte ceiling later would silently make it dead.
    """

    assert comfy_workflow_packages.MAX_UI_GRAPH_STRING_CHARACTERS <= (
        comfy_workflow_packages.MAX_UI_GRAPH_BYTES
    )
    assert comfy_workflow_packages.MAX_UI_GRAPH_KEY_CHARACTERS <= (
        comfy_workflow_packages.MAX_UI_GRAPH_STRING_CHARACTERS
    )


def test_the_json_text_ceiling_stays_inside_the_json_byte_ceiling() -> None:
    """One text value may not be allowed more than the whole document."""

    assert artifact_library_schema.MAX_JSON_TEXT_BYTES <= artifact_library_schema.MAX_JSON_BYTES
