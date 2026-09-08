from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from local_lm.model_planner import workflow_artifact_contract
from local_lm.workflow_output_geometry import (
    WORKFLOW_OUTPUT_GEOMETRY_UNAVAILABLE,
    WorkflowOutputGeometryError,
    WorkflowOutputGeometryProof,
    prove_workflow_output_geometry,
    workflow_output_geometry_payload,
)


def _graph() -> dict[str, object]:
    return {
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": "${width}",
                "height": "${height}",
                "batch_size": 1,
            },
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"latent_image": ["latent", 0]},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0]},
        },
    }


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "width": {
                "type": "integer",
                "default": 1024,
                "minimum": 128,
                "maximum": 2048,
                "multipleOf": 64,
            },
            "height": {
                "type": "integer",
                "default": 768,
                "minimum": 128,
                "maximum": 2048,
                "multipleOf": 64,
            },
        },
    }


def _arguments() -> dict[str, object]:
    graph = _graph()
    schema = _schema()
    dependencies: dict[str, object] = {}
    return {
        "workflow_id": "workflow-1",
        "revision_id": "revision-1",
        "operation": "text_to_image",
        "engine": "comfyui",
        "api_graph": graph,
        "input_schema": schema,
        "dependencies": dependencies,
        "artifact_sha256": workflow_artifact_contract(
            operation="text_to_image",
            engine="comfyui",
            api_graph=graph,
            input_schema=schema,
            dependencies=dependencies,
        ),
        "trusted": True,
    }


def _prove(arguments: dict[str, object] | None = None):  # type: ignore[no-untyped-def]
    return prove_workflow_output_geometry(**(arguments or _arguments()))


def _rehash(arguments: dict[str, object]) -> None:
    arguments["artifact_sha256"] = workflow_artifact_contract(
        operation=str(arguments["operation"]),
        engine=str(arguments["engine"]),
        api_graph=arguments["api_graph"],  # type: ignore[arg-type]
        input_schema=arguments["input_schema"],  # type: ignore[arg-type]
        dependencies=arguments["dependencies"],  # type: ignore[arg-type]
    )


def _insert_downstream_transform(graph: dict[str, object], class_type: str) -> None:
    graph["transform"] = {
        "class_type": class_type,
        "inputs": {"image": ["decode", 0]},
    }
    graph["save"]["inputs"]["images"] = ["transform", 0]  # type: ignore[index]


def test_proof_binds_exact_revision_graph_and_schema_limits() -> None:
    result = _prove()

    assert result.available is True
    assert result.reason is None
    assert result.proof is not None
    proof = result.proof
    assert proof.workflow_id == "workflow-1"
    assert proof.revision_id == "revision-1"
    assert proof.latent_node_id == "latent"
    assert proof.sampler_node_ids == ("sampler",)
    assert proof.decode_node_ids == ("decode",)
    assert proof.save_node_ids == ("save",)
    assert (proof.width.minimum, proof.width.default, proof.width.maximum) == (128, 1024, 2048)
    assert (proof.height.minimum, proof.height.default, proof.height.maximum) == (128, 768, 2048)
    assert proof.width.multiple_of == proof.height.multiple_of == 64
    assert proof.graph_binding_verified is True
    assert proof.request_authorized is False
    assert proof.capability.graph_binding_verified is False
    assert proof.capability.request_authorized is False

    payload = workflow_output_geometry_payload(result)
    assert payload["available"] is True
    assert payload["size_modes"] == ["exact"]
    assert payload["width"] == {
        "key": "width",
        "node_id": "latent",
        "input_name": "width",
        "default": 1024,
        "minimum": 128,
        "maximum": 2048,
        "multiple_of": 64,
    }
    assert payload["request_authorized"] is False


def test_proof_is_sealed_frozen_and_detached() -> None:
    arguments = _arguments()
    result = _prove(arguments)
    assert result.proof is not None
    with pytest.raises(WorkflowOutputGeometryError):
        WorkflowOutputGeometryProof(object())
    with pytest.raises(FrozenInstanceError):
        result.proof.revision_id = "other"  # type: ignore[misc]

    cast_graph = arguments["api_graph"]
    assert isinstance(cast_graph, dict)
    cast_graph["save"] = {"class_type": "ImageScale", "inputs": {}}
    assert result.proof.save_node_ids == ("save",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", "image_to_image"),
        ("operation", "text_to_video"),
        ("engine", "mock"),
        ("trusted", False),
        ("artifact_sha256", "A" * 64),
        ("artifact_sha256", "a" * 63),
    ],
)
def test_revision_authority_mutations_fail_closed(field: str, value: object) -> None:
    arguments = _arguments()
    arguments[field] = value

    result = _prove(arguments)

    assert result.available is False
    assert result.reason == WORKFLOW_OUTPUT_GEOMETRY_UNAVAILABLE
    assert result.proof is None


def test_artifact_drift_fails_closed() -> None:
    arguments = _arguments()
    graph = arguments["api_graph"]
    assert isinstance(graph, dict)
    graph["latent"]["inputs"]["width"] = 512  # type: ignore[index]

    assert _prove(arguments).available is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda graph: graph["latent"]["inputs"].__setitem__("width", 1024),
        lambda graph: graph["latent"]["inputs"].__setitem__("height", "${width}"),
        lambda graph: graph["latent"].__setitem__("class_type", "LoadImage"),
        lambda graph: graph["sampler"].__setitem__("class_type", "KSamplerAdvanced"),
        lambda graph: graph["sampler"]["inputs"].__setitem__("latent_image", ["latent", 1]),
        lambda graph: graph["decode"].__setitem__("class_type", "VAEDecodeTiled"),
        lambda graph: graph["decode"]["inputs"].__setitem__("samples", ["sampler", 1]),
        lambda graph: graph["save"].__setitem__("class_type", "PreviewImage"),
        lambda graph: graph["save"]["inputs"].__setitem__("images", ["decode", 1]),
        lambda graph: _insert_downstream_transform(graph, "ImageScale"),
        lambda graph: _insert_downstream_transform(graph, "ImageCrop"),
        lambda graph: _insert_downstream_transform(graph, "ImagePadForOutpaint"),
        lambda graph: _insert_downstream_transform(graph, "UnknownGeometryNode"),
        lambda graph: graph.__setitem__(
            "second-save",
            {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}},
        ),
    ],
)
def test_unknown_or_geometry_changing_paths_fail_closed(mutation) -> None:  # type: ignore[no-untyped-def]
    arguments = _arguments()
    graph = arguments["api_graph"]
    assert isinstance(graph, dict)
    mutation(graph)
    _rehash(arguments)

    assert _prove(arguments).available is False


def test_every_save_must_share_the_same_proven_root() -> None:
    arguments = _arguments()
    graph = arguments["api_graph"]
    assert isinstance(graph, dict)
    graph.update(
        {
            "latent-2": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": "${width}", "height": "${height}"},
            },
            "sampler-2": {
                "class_type": "KSampler",
                "inputs": {"latent_image": ["latent-2", 0]},
            },
            "decode-2": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sampler-2", 0]},
            },
            "save-2": {
                "class_type": "SaveImage",
                "inputs": {"images": ["decode-2", 0]},
            },
        }
    )
    _rehash(arguments)

    assert _prove(arguments).available is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda schema: schema.__setitem__("type", "array"),
        lambda schema: schema["properties"].__delitem__("width"),
        lambda schema: schema["properties"]["width"].__setitem__("type", "number"),
        lambda schema: schema["properties"]["width"].__setitem__("readOnly", True),
        lambda schema: schema["properties"]["width"].__setitem__("const", 1024),
        lambda schema: schema["properties"]["height"].__setitem__("enum", [768]),
        lambda schema: schema["properties"]["width"].__setitem__("default", 1000),
        lambda schema: schema["properties"]["height"].__setitem__("maximum", 8192),
        lambda schema: schema["properties"]["width"].__delitem__("multipleOf"),
    ],
)
def test_schema_mutations_fail_closed(mutation) -> None:  # type: ignore[no-untyped-def]
    arguments = _arguments()
    schema = arguments["input_schema"]
    assert isinstance(schema, dict)
    mutation(schema)
    _rehash(arguments)

    assert _prove(arguments).available is False


def test_supported_multiple_saves_are_all_bound() -> None:
    arguments = _arguments()
    graph = arguments["api_graph"]
    assert isinstance(graph, dict)
    graph["save-2"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0]},
    }
    _rehash(arguments)

    result = _prove(arguments)

    assert result.available is True
    assert result.proof is not None
    assert result.proof.save_node_ids == ("save", "save-2")


def test_split_dimension_bindings_and_lone_outpaint_node_fail_closed() -> None:
    split = _arguments()
    split_graph = split["api_graph"]
    assert isinstance(split_graph, dict)
    split_graph["latent"]["inputs"]["height"] = 768  # type: ignore[index]
    split_graph["unrelated"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": "${height}"},
    }
    _rehash(split)
    assert _prove(split).available is False

    outpaint = _arguments()
    outpaint["api_graph"] = {"outpaint": {"class_type": "ImagePadForOutpaint", "inputs": {}}}
    _rehash(outpaint)
    assert _prove(outpaint).available is False


def test_refusal_payload_is_fixed_and_non_echoing() -> None:
    arguments = _arguments()
    arguments["revision_id"] = "private-user-controlled-id"
    arguments["engine"] = "private-engine-name"

    payload = workflow_output_geometry_payload(_prove(arguments))

    assert payload["available"] is False
    assert payload["reason"] == WORKFLOW_OUTPUT_GEOMETRY_UNAVAILABLE
    assert payload["revision_id"] is None
    assert payload["engine"] is None
    assert "private" not in str(payload)


def test_mapping_subclasses_and_hostile_nesting_fail_closed() -> None:
    class MappingSubclass(dict[str, object]):
        pass

    subclass = _arguments()
    subclass["api_graph"] = MappingSubclass(_graph())
    assert _prove(subclass).available is False

    nested = _arguments()
    value: list[object] = []
    root = value
    for _ in range(30):
        child: list[object] = []
        value.append(child)
        value = child
    nested["dependencies"] = {"nested": root}
    assert _prove(nested).available is False


def test_nan_and_boolean_dimensions_fail_closed() -> None:
    for invalid in (float("nan"), True):
        arguments = deepcopy(_arguments())
        schema = arguments["input_schema"]
        assert isinstance(schema, dict)
        schema["properties"]["width"]["default"] = invalid  # type: ignore[index]
        arguments["artifact_sha256"] = "a" * 64
        assert _prove(arguments).available is False


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"exclusiveMaximum": 1536}, id="exclusive-maximum"),
        pytest.param({"exclusiveMinimum": 128}, id="exclusive-minimum"),
        pytest.param({"$ref": "#/$defs/width"}, id="field-ref"),
        pytest.param({"allOf": [{"maximum": 1024}]}, id="field-all-of"),
        pytest.param({"not": {"const": 2048}}, id="field-not"),
    ],
)
def test_a_dimension_constraint_the_proof_cannot_read_is_refused(
    mutation: dict[str, object],
) -> None:
    """An advertised range must be a subset of what the workflow accepts.

    The proof reads type, default, minimum, maximum and multipleOf. Anything
    else that ASSERTS can only narrow the accepted range, so describing the
    schema from the part that is understood would offer a size the workflow
    then refuses. exclusiveMaximum 1536 beside maximum 2048 is the plain case:
    the old code advertised 2048 and a request for it was legal here and
    invalid there.
    """

    arguments = _arguments()
    schema = cast(dict[str, Any], arguments["input_schema"])
    cast(dict[str, Any], schema["properties"])["width"].update(mutation)
    _rehash(arguments)
    result = _prove(arguments)
    assert result.available is False
    assert result.proof is None


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("allOf", [{"properties": {"width": {"maximum": 1024}}}], id="all-of"),
        pytest.param("anyOf", [{"properties": {"width": {"maximum": 1024}}}], id="any-of"),
        pytest.param("oneOf", [{"properties": {"width": {"maximum": 1024}}}], id="one-of"),
        pytest.param("not", {"properties": {"width": {"minimum": 2048}}}, id="not"),
        pytest.param("if", {"properties": {"width": {"maximum": 1024}}}, id="if"),
        pytest.param("$ref", "#/$defs/whole", id="ref"),
        pytest.param("dependentSchemas", {"width": {"maximum": 1024}}, id="dependent-schemas"),
        pytest.param("const", {"width": 1024, "height": 768}, id="const"),
        pytest.param("enum", [{"width": 1024, "height": 768}], id="enum"),
        pytest.param("unevaluatedProperties", False, id="unevaluated-properties"),
        pytest.param("patternProperties", {"^w": {"maximum": 1024}}, id="pattern-properties"),
    ],
)
def test_a_whole_schema_constraint_is_refused_even_when_the_dimensions_look_plain(
    keyword: str, value: object
) -> None:
    """A top-level constraint narrows a dimension without touching its bounds.

    allOf [{properties: {width: {maximum: 1024}}}] leaves the width subschema
    reading maximum 2048, so every per-field check passes and the capability
    would still advertise 2048. Whole-object const and enum do the same thing
    more sharply: they permit exactly one pair while the dimensions still
    declare a range.
    """

    arguments = _arguments()
    schema = cast(dict[str, Any], arguments["input_schema"])
    schema[keyword] = value
    _rehash(arguments)
    result = _prove(arguments)
    assert result.available is False
    assert result.proof is None


def test_an_unrecognised_top_level_keyword_is_refused_without_being_named() -> None:
    """The rule is an allowlist, so a keyword nobody listed still refuses.

    This is the property the earlier repair failed to establish. Naming the
    composition keywords left whole-object const and enum accepted, and any
    future assertion keyword would have been accepted too. Refusing what is not
    understood does not depend on having anticipated it, so this test passes
    with a keyword invented here that no list mentions.
    """

    arguments = _arguments()
    schema = cast(dict[str, Any], arguments["input_schema"])
    schema["x-constraint-nobody-enumerated"] = {"width": {"maximum": 1024}}
    _rehash(arguments)
    result = _prove(arguments)
    assert result.available is False
    assert result.proof is None


def test_annotation_keywords_do_not_stop_a_supported_schema() -> None:
    """Describing a dimension is not constraining it."""

    arguments = _arguments()
    schema = cast(dict[str, Any], arguments["input_schema"])
    cast(dict[str, Any], schema["properties"])["width"].update(
        {"title": "Width", "description": "Output width in pixels", "examples": [1024]}
    )
    _rehash(arguments)
    result = _prove(arguments)
    assert result.available is True
    assert result.proof is not None
    assert result.proof.width.maximum == 2048
