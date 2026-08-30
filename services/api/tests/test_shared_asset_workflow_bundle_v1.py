from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import local_lm.shared_asset_workflow_bundle_v1 as bundle
from local_lm.shared_asset_package_v1 import (
    SCHEMA_ID_V2 as PACKAGE_SCHEMA_ID_V2,
)
from local_lm.shared_asset_package_v1 import (
    SCHEMA_VERSION_V2 as PACKAGE_SCHEMA_VERSION_V2,
)
from local_lm.shared_asset_package_v1 import (
    load_package,
    publish_package,
)
from local_lm.shared_asset_store_v1 import object_path, publish_file
from local_lm.shared_asset_view_v1 import INVALID_VIEW, SharedAssetViewError, open_package_view
from local_lm.shared_asset_workflow_bundle_v1 import (
    DECLARED_COLOR_SPACES,
    DECLARED_COMPUTE_PRECISIONS,
    DECLARED_OUTPUT_FORMATS,
    INVALID_BUNDLE,
    MAX_DECLARED_OUTPUT_COMPONENT,
    MAX_ENGINE_VERSION_CHARS,
    MAX_JSON_KEY_CHARS,
    MAX_JSON_STRING_CHARS,
    MAX_JSON_TEXT_CHARS,
    MAX_WORKFLOW_BUNDLE_BYTES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    WORKFLOW_ENGINES,
    WORKFLOW_OPERATIONS,
    SharedAssetWorkflowBundleError,
    decode_workflow_bundle,
    encode_workflow_bundle,
    workflow_bundle_sha256,
)


def _dependency_contract() -> dict[str, Any]:
    return {
        "version": 1,
        "slots": [
            {
                "name": "runtime",
                "resource_kind": "runtime",
                "required": True,
                "satisfaction": "all_of",
                "requirements": [
                    {
                        "key": "engine",
                        "constraints": {
                            "engine": "comfyui",
                            "kind": "runtime",
                            "node_types": ["Sampler", "Loader"],
                        },
                    }
                ],
            }
        ],
    }


def _declared_metadata() -> dict[str, Any]:
    return {
        "declared_compute_precision": "fp16",
        "declared_output": {
            "channels": 4,
            "color_space": "srgb",
            "format": "image/png",
            "frame_count": 1,
            "frame_rate": {"denominator": 1, "numerator": 24},
            "height": 1024,
            "width": 1024,
        },
    }


def _bundle_fields() -> dict[str, Any]:
    return {
        "operation": "text_to_image",
        "engine": "comfyui",
        "engine_version": "0.3.50",
        "ui_graph": {
            "nodes": [{"id": 1, "type": "Sampler"}],
            "links": [],
        },
        "api_graph": {
            "1": {"class_type": "Sampler", "inputs": {"denoise": 0.5, "steps": 24}},
        },
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
        "dependency_contract": _dependency_contract(),
        "technical_metadata": _declared_metadata(),
    }


def _encode(fields: dict[str, Any] | None = None) -> bytes:
    return encode_workflow_bundle(**(_bundle_fields() if fields is None else fields))


def _assert_invalid(call: Callable[[], object]) -> None:
    with pytest.raises(
        SharedAssetWorkflowBundleError, match=f"^{re.escape(INVALID_BUNDLE)}$"
    ) as caught:
        call()
    assert type(caught.value) is SharedAssetWorkflowBundleError
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_bundle_is_canonical_neutral_and_digest_addressed() -> None:
    payload = _encode()
    decoded = decode_workflow_bundle(payload)

    assert SCHEMA_ID == "lm-atelier-shared-asset-workflow-bundle-v1"
    assert SCHEMA_VERSION == 1
    assert INVALID_BUNDLE == "shared asset workflow bundle is invalid"
    assert decoded.operation == "text_to_image"
    assert decoded.engine == "comfyui"
    assert decoded.ui_graph["nodes"] == [{"id": 1, "type": "Sampler"}]
    assert decoded.dependency_contract["version"] == 1
    assert decoded.technical_metadata == _declared_metadata()
    assert workflow_bundle_sha256(payload) == hashlib.sha256(payload).hexdigest()
    assert payload == json.dumps(
        json.loads(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def test_bundle_v1_operation_vocabulary_is_explicit_and_pinned() -> None:
    assert (
        frozenset(
            {
                "image_to_image",
                "image_to_video",
                "text",
                "text_to_image",
                "text_to_video",
            }
        )
        == WORKFLOW_OPERATIONS
    )
    for operation in WORKFLOW_OPERATIONS:
        fields = _bundle_fields()
        fields["operation"] = operation
        assert decode_workflow_bundle(_encode(fields)).operation == operation

    fields = _bundle_fields()
    fields["operation"] = "future_operation"
    _assert_invalid(lambda: _encode(fields))


def test_bundle_v1_engine_vocabulary_and_version_are_pinned() -> None:
    assert frozenset({"comfyui"}) == WORKFLOW_ENGINES
    for engine, version in (("other_engine", "0.3.50"), ("comfyui", "local-label")):
        fields = _bundle_fields()
        fields["engine"] = engine
        fields["engine_version"] = version
        _assert_invalid(partial(_encode, fields))


def test_generic_graph_identity_is_syntactic() -> None:
    integer = _bundle_fields()
    floating = copy.deepcopy(integer)
    floating["api_graph"]["1"]["inputs"]["steps"] = 24.0
    reordered_list = copy.deepcopy(integer)
    reordered_list["ui_graph"]["links"] = [2, 1]
    ordered_list = copy.deepcopy(integer)
    ordered_list["ui_graph"]["links"] = [1, 2]

    assert _encode(integer) != _encode(floating)
    assert _encode(ordered_list) != _encode(reordered_list)


def test_mapping_and_dependency_order_do_not_change_bundle_identity() -> None:
    fields = _bundle_fields()
    reordered = copy.deepcopy(fields)
    reordered["ui_graph"] = {
        "links": [],
        "nodes": [{"type": "Sampler", "id": 1}],
    }
    reordered["api_graph"] = {
        "1": {"inputs": {"steps": 24, "denoise": 0.5}, "class_type": "Sampler"}
    }
    contract = reordered["dependency_contract"]
    contract["slots"][0]["requirements"][0]["constraints"]["node_types"] = [
        "Loader",
        "Sampler",
    ]
    metadata = reordered["technical_metadata"]
    metadata["declared_output"] = dict(reversed(list(metadata["declared_output"].items())))

    assert _encode(reordered) == _encode(fields)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(operation="image_to_image"),
        lambda value: value.update(engine_version="0.3.51"),
        lambda value: value["ui_graph"].update(extra=True),
        lambda value: value["api_graph"]["1"]["inputs"].update(steps=25),
        lambda value: value["input_schema"].update(additionalProperties=False),
        lambda value: value["dependency_contract"]["slots"][0].update(required=False),
        lambda value: value["technical_metadata"].update(declared_compute_precision="fp32"),
        lambda value: value["technical_metadata"]["declared_output"].update(channels=3),
        lambda value: value["technical_metadata"]["declared_output"].update(
            color_space="display-p3"
        ),
        lambda value: value["technical_metadata"]["declared_output"].update(format="image/webp"),
        lambda value: value["technical_metadata"]["declared_output"].update(frame_count=2),
        lambda value: value["technical_metadata"]["declared_output"].update(
            frame_rate={"denominator": 1001, "numerator": 24000}
        ),
        lambda value: value["technical_metadata"]["declared_output"].update(height=768),
        lambda value: value["technical_metadata"]["declared_output"].update(width=768),
    ],
)
def test_each_exact_declared_fact_changes_digest(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    baseline = _bundle_fields()
    changed = copy.deepcopy(baseline)
    mutate(changed)

    assert workflow_bundle_sha256(_encode(changed)) != workflow_bundle_sha256(_encode(baseline))


def test_complete_metadata_has_one_explicit_null_representation() -> None:
    fields = _bundle_fields()
    fields["technical_metadata"] = {
        "declared_compute_precision": None,
        "declared_output": {
            "channels": None,
            "color_space": None,
            "format": None,
            "frame_count": None,
            "frame_rate": None,
            "height": None,
            "width": None,
        },
    }

    assert (
        decode_workflow_bundle(_encode(fields)).technical_metadata == fields["technical_metadata"]
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"declared_compute_precision": None},
        {"declared_output": _declared_metadata()["declared_output"]},
        {
            "declared_compute_precision": "fp16",
            "declared_output": {},
        },
        {
            "declared_compute_precision": "fp16",
            "declared_output": {
                **_declared_metadata()["declared_output"],
                "local_path": "C:/local/output",
            },
        },
        {
            "declared_compute_precision": "profile-local-label",
            "declared_output": _declared_metadata()["declared_output"],
        },
        {
            "declared_compute_precision": "fp16",
            "declared_output": {
                **_declared_metadata()["declared_output"],
                "format": "profile-local/format",
            },
        },
        {
            "declared_compute_precision": "fp16",
            "declared_output": {
                **_declared_metadata()["declared_output"],
                "color_space": "trusted",
            },
        },
        {"output": _declared_metadata()["declared_output"], "precision": "fp16"},
    ],
)
def test_bundle_refuses_incomplete_open_or_profile_local_metadata(
    metadata: dict[str, Any],
) -> None:
    fields = _bundle_fields()
    fields["technical_metadata"] = metadata

    _assert_invalid(lambda: _encode(fields))


@pytest.mark.parametrize(
    "frame_rate",
    [
        24,
        24.0,
        {},
        {"numerator": 24},
        {"denominator": 1},
        {"denominator": 1, "numerator": 24, "extra": 1},
        {"denominator": 2, "numerator": 48},
        {"denominator": 0, "numerator": 24},
        {"denominator": 1, "numerator": True},
        {"denominator": 1.0, "numerator": 24},
        {"denominator": 1, "numerator": 10**1000},
    ],
)
def test_invalid_frame_rates_are_fixed_refusals(frame_rate: object) -> None:
    fields = _bundle_fields()
    fields["technical_metadata"]["declared_output"]["frame_rate"] = frame_rate

    _assert_invalid(lambda: _encode(fields))

    value = json.loads(_encode())
    value["technical_metadata"]["declared_output"]["frame_rate"] = frame_rate
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    _assert_invalid(lambda: decode_workflow_bundle(raw))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("channels", 0),
        ("channels", True),
        ("frame_count", MAX_DECLARED_OUTPUT_COMPONENT + 1),
        ("height", -1),
        ("width", 1.0),
        ("format", "PNG"),
        ("color_space", "sRGB"),
    ],
)
def test_declared_output_values_are_finite_and_bounded(key: str, value: object) -> None:
    fields = _bundle_fields()
    fields["technical_metadata"]["declared_output"][key] = value
    _assert_invalid(lambda: _encode(fields))


def test_declared_value_vocabularies_are_finite_and_versioned() -> None:
    assert (
        frozenset({"bf16", "fp16", "fp32", "fp64", "fp8", "int4", "int8", "tf32"})
        == DECLARED_COMPUTE_PRECISIONS
    )
    assert (
        frozenset(
            {
                "image/gif",
                "image/jpeg",
                "image/png",
                "image/tiff",
                "image/webp",
                "video/h264-mp4",
                "video/mkv",
                "video/webm",
            }
        )
        == DECLARED_OUTPUT_FORMATS
    )
    assert (
        frozenset({"display-p3", "linear-srgb", "rec2020", "rec709", "srgb"})
        == DECLARED_COLOR_SPACES
    )


def test_dependency_contract_refuses_local_paths_without_echoing_them() -> None:
    fields = _bundle_fields()
    contract = fields["dependency_contract"]
    contract["slots"][0]["requirements"][0]["constraints"]["local_path"] = "C:/local/model.bin"

    _assert_invalid(lambda: _encode(fields))


def test_dependency_contract_refuses_dict_subclasses_before_using_them() -> None:
    class ExplodingDict(dict[str, Any]):
        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("must not run")

    fields = _bundle_fields()
    fields["dependency_contract"] = ExplodingDict(_dependency_contract())
    _assert_invalid(lambda: _encode(fields))


def test_decoder_refuses_noncanonical_unknown_and_oversized_documents() -> None:
    payload = _encode()
    value = json.loads(payload)
    value["unexpected"] = True

    _assert_invalid(
        lambda: decode_workflow_bundle(payload.replace(b'"api_graph"', b'"api_graph" ', 1))
    )
    _assert_invalid(
        lambda: decode_workflow_bundle(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    )
    _assert_invalid(lambda: decode_workflow_bundle(b" " * (MAX_WORKFLOW_BUNDLE_BYTES + 1)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", "made_up"),
        ("engine", "Comfy UI"),
        ("engine_version", ""),
        ("ui_graph", []),
        ("api_graph", {1: {"class_type": "Sampler"}}),
        ("input_schema", {"minimum": float("nan")}),
        ("technical_metadata", {"declared_output": object()}),
    ],
)
def test_encoder_refuses_invalid_values_with_one_fixed_error(field: str, value: object) -> None:
    fields = _bundle_fields()
    fields[field] = value

    _assert_invalid(lambda: _encode(fields))


@pytest.mark.parametrize("kind", ["key", "value", "cumulative", "engine_version"])
def test_oversized_text_is_refused_before_canonical_encoding(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = _bundle_fields()
    if kind == "key":
        fields["api_graph"] = {"k" * (MAX_JSON_KEY_CHARS + 1): 1}
    elif kind == "value":
        fields["api_graph"] = {"value": "x" * (MAX_JSON_STRING_CHARS + 1)}
    elif kind == "cumulative":
        fields["api_graph"] = {
            "a": "x" * MAX_JSON_STRING_CHARS,
            "b": "x" * MAX_JSON_STRING_CHARS,
            "c": "x" * (MAX_JSON_TEXT_CHARS - 2 * MAX_JSON_STRING_CHARS - 3 - len("0.3.50") + 1),
        }
        fields["ui_graph"] = {}
        fields["input_schema"] = {}
    else:
        fields["api_graph"] = {}
        fields["engine_version"] = "1" * (MAX_ENGINE_VERSION_CHARS + 1) + ".0.0"

    def encoding_was_reached(_value: object) -> bytes:
        raise AssertionError("oversized graph reached canonical JSON encoding")

    monkeypatch.setattr(bundle, "_canonical_bytes", encoding_was_reached)
    _assert_invalid(lambda: _encode(fields))


def _bypass_dependency_for_graph_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def canonical_dependency(_value: object) -> dict[str, object]:
        return {"slots": [], "version": 1}

    monkeypatch.setattr(bundle, "_dependency_contract", canonical_dependency)


@pytest.mark.parametrize(
    ("constant", "limit", "graph"),
    [
        ("MAX_JSON_DEPTH", 1, {"a": {"b": 1}}),
        ("MAX_JSON_NODES", 2, {"a": 1, "b": 2}),
        ("MAX_JSON_OBJECT_MEMBERS", 1, {"a": 1, "b": 2}),
        ("MAX_JSON_ARRAY_ITEMS", 1, {"a": [1, 2]}),
    ],
)
def test_structural_parser_ceilings_bind(
    constant: str,
    limit: int,
    graph: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bundle, constant, limit)
    _bypass_dependency_for_graph_bound(monkeypatch)
    fields = _bundle_fields()
    fields["api_graph"] = graph
    _assert_invalid(lambda: _encode(fields))


@pytest.mark.parametrize(
    ("constant", "limit", "graph"),
    [
        ("MAX_JSON_DEPTH", 1, {"a": 1}),
        ("MAX_JSON_NODES", 4, {"a": 1}),
        ("MAX_JSON_OBJECT_MEMBERS", 1, {"a": 1}),
        ("MAX_JSON_ARRAY_ITEMS", 1, {"a": [1]}),
    ],
)
def test_structural_parser_ceilings_accept_the_exact_limit(
    constant: str,
    limit: int,
    graph: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bundle, constant, limit)
    _bypass_dependency_for_graph_bound(monkeypatch)
    fields = _bundle_fields()
    fields["api_graph"] = graph
    fields["ui_graph"] = {}
    fields["input_schema"] = {}
    assert decode_workflow_bundle(_encode(fields)).api_graph == graph


def test_text_and_declared_component_ceilings_accept_the_exact_limit() -> None:
    fields = _bundle_fields()
    key = "k" * MAX_JSON_KEY_CHARS
    value = "x" * MAX_JSON_STRING_CHARS
    fields["api_graph"] = {key: value}
    fields["ui_graph"] = {}
    fields["input_schema"] = {}
    engine_version = "1" + "0" * (MAX_ENGINE_VERSION_CHARS - 5) + ".0.0"
    fields["engine_version"] = engine_version
    fields["technical_metadata"]["declared_output"]["width"] = MAX_DECLARED_OUTPUT_COMPONENT

    decoded = decode_workflow_bundle(_encode(fields))
    assert decoded.api_graph == {key: value}
    assert decoded.engine_version == engine_version
    assert decoded.technical_metadata["declared_output"]["width"] == MAX_DECLARED_OUTPUT_COMPONENT

    cumulative = _bundle_fields()
    cumulative["api_graph"] = {
        "a": "x" * MAX_JSON_STRING_CHARS,
        "b": "x" * MAX_JSON_STRING_CHARS,
        "c": "x" * (MAX_JSON_TEXT_CHARS - 2 * MAX_JSON_STRING_CHARS - 3 - len("0.3.50")),
    }
    cumulative["ui_graph"] = {}
    cumulative["input_schema"] = {}
    assert decode_workflow_bundle(_encode(cumulative)).api_graph == cumulative["api_graph"]


def test_workflow_bundle_parser_ceilings_are_complete_and_bounded() -> None:
    reviewed = {
        "MAX_DECLARED_OUTPUT_COMPONENT": 1_000_000,
        "MAX_ENGINE_VERSION_CHARS": 64,
        "MAX_JSON_ARRAY_ITEMS": 100_000,
        "MAX_JSON_DEPTH": 32,
        "MAX_JSON_KEY_CHARS": 256,
        "MAX_JSON_NODES": 100_000,
        "MAX_JSON_OBJECT_MEMBERS": 10_000,
        "MAX_JSON_STRING_CHARS": 100_000,
        "MAX_JSON_TEXT_CHARS": 250_000,
        "MAX_WORKFLOW_BUNDLE_BYTES": 4 * 1024 * 1024,
    }
    actual = {
        name: value
        for name, value in vars(bundle).items()
        if name.startswith("MAX_") and type(value) is int
    }

    assert set(actual) == set(reviewed)
    for name, ceiling in reviewed.items():
        assert 0 < actual[name] <= ceiling


def test_published_bundle_uses_the_package_v2_workflow_envelope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    source = tmp_path / "workflow-bundle.json"
    payload = _encode()
    source.write_bytes(payload)

    bundle_digest = publish_file(root=root, source=source)
    package_digest = publish_package(root=root, members={"workflow": bundle_digest})
    descriptor = json.loads(
        object_path(root=root, digest=package_digest).read_text(encoding="ascii")
    )

    assert bundle_digest == workflow_bundle_sha256(payload)
    assert descriptor == {
        "members": {"workflow": bundle_digest},
        "schema": PACKAGE_SCHEMA_ID_V2,
        "version": PACKAGE_SCHEMA_VERSION_V2,
    }
    assert load_package(root=root, digest=package_digest) == (("workflow", bundle_digest),)


def test_v1_role_view_refuses_a_v2_workflow_package(tmp_path: Path) -> None:
    root = tmp_path / "library"
    source = tmp_path / "workflow-bundle.json"
    source.write_bytes(_encode())
    workflow = publish_file(root=root, source=source)
    package_digest = publish_package(root=root, members={"workflow": workflow})

    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        open_package_view(root=root, digest=package_digest)
