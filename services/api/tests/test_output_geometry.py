from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from local_lm.output_geometry import (
    MAX_DIMENSION,
    MAX_GEOMETRY_BUCKETS,
    MAX_PIXELS,
    OutputGeometryCapability,
    OutputGeometryError,
    declare_output_geometry,
    output_geometry_capability_payload,
    resolve_output_geometry,
)


def _combination(
    mode: str,
    size_mode: str,
    *,
    default: tuple[int, int] = (512, 512),
    buckets: list[list[int]] | None = None,
) -> dict[str, object]:
    return {
        "mode": mode,
        "size_mode": size_mode,
        "min_width": 64,
        "max_width": 2048,
        "min_height": 64,
        "max_height": 2048,
        "width_multiple": 64,
        "height_multiple": 64,
        "max_pixels": 2048 * 2048,
        "min_aspect": [1, 4],
        "max_aspect": [4, 1],
        "default_width": default[0],
        "default_height": default[1],
        "buckets": buckets or [[512, 512], [768, 1024], [1024, 768]],
    }


def _declaration() -> dict[str, object]:
    return {
        "version": 1,
        "allowed_modes": ["image", "video"],
        "allowed_preset_ids": ["16:9", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16"],
        "combinations": [
            _combination("image", "exact"),
            _combination("image", "preset"),
            _combination("image", "workflow_native"),
            _combination("video", "exact"),
        ],
    }


def _capability() -> OutputGeometryCapability:
    return declare_output_geometry(_declaration())


def _refuses(value: object) -> None:
    with pytest.raises(OutputGeometryError) as raised:
        declare_output_geometry(value)
    assert str(raised.value) == "Output geometry declaration or request is invalid"


def _request_refuses(value: object) -> None:
    with pytest.raises(OutputGeometryError) as raised:
        resolve_output_geometry(_capability(), value)
    assert str(raised.value) == "Output geometry declaration or request is invalid"


def test_capability_is_owned_frozen_detached_and_non_authoritative() -> None:
    source = _declaration()
    capability = declare_output_geometry(source)
    assert capability.repository_verified is False
    assert capability.workflow_verified is False
    assert capability.graph_binding_verified is False
    assert capability.capability_verified is False
    assert capability.request_authorized is False
    with pytest.raises(TypeError):
        OutputGeometryCapability()  # type: ignore[call-arg]
    with pytest.raises(OutputGeometryError):
        OutputGeometryCapability(object())
    with pytest.raises(FrozenInstanceError):
        capability.allowed_modes = ("image",)  # type: ignore[misc]
    source["allowed_modes"] = []
    combinations = source["combinations"]
    assert isinstance(combinations, list)
    combinations.clear()
    assert capability.allowed_modes == ("image", "video")
    assert len(capability.combinations) == 4


def test_payload_is_fresh_and_round_trips() -> None:
    capability = _capability()
    payload = output_geometry_capability_payload(capability)
    assert declare_output_geometry(payload) == capability
    modes = payload["allowed_modes"]
    assert isinstance(modes, list)
    modes.clear()
    assert capability.allowed_modes == ("image", "video")


def test_exact_dimensions_return_unchanged_with_all_truth_flags_false() -> None:
    resolved = resolve_output_geometry(
        _capability(),
        {"mode": "image", "size_mode": "exact", "width": 640, "height": 832},
    )
    assert (resolved.width, resolved.height) == (640, 832)
    assert resolved.preset_id is None and resolved.source_fit is None
    assert resolved.source_fit_applied is False
    assert resolved.source_dimensions_verified is False
    assert resolved.repository_verified is False
    assert resolved.workflow_verified is False
    assert resolved.graph_binding_verified is False
    assert resolved.capability_verified is False
    assert resolved.request_authorized is False


@pytest.mark.parametrize(
    "geometry_request",
    [
        {"mode": "image", "size_mode": "exact", "width": 63, "height": 512},
        {"mode": "image", "size_mode": "exact", "width": 2112, "height": 512},
        {"mode": "image", "size_mode": "exact", "width": 640, "height": 65},
        {"mode": "image", "size_mode": "exact", "width": 641, "height": 832},
        {"mode": "image", "size_mode": "exact", "width": 64, "height": 2048},
        {"mode": "image", "size_mode": "exact", "width": True, "height": 512},
    ],
)
def test_exact_bounds_multiples_aspect_and_types_refuse(geometry_request: object) -> None:
    _request_refuses(geometry_request)


def test_presets_require_builtin_vocabulary_and_declaration() -> None:
    declaration = _declaration()
    declaration["allowed_preset_ids"] = ["1:1"]
    capability = declare_output_geometry(declaration)
    request = {"mode": "image", "size_mode": "preset", "preset_id": "1:1"}
    assert resolve_output_geometry(capability, request).preset_id == "1:1"
    for preset in ("16:9", "5:4"):
        with pytest.raises(OutputGeometryError):
            resolve_output_geometry(
                capability, {"mode": "image", "size_mode": "preset", "preset_id": preset}
            )


@pytest.mark.parametrize("preset_id", ["1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"])
def test_every_builtin_preset_is_only_vocabulary_until_declared(preset_id: str) -> None:
    resolved = resolve_output_geometry(
        _capability(),
        {"mode": "image", "size_mode": "preset", "preset_id": preset_id},
    )
    assert resolved.preset_id == preset_id
    assert resolved.capability_verified is False
    assert resolved.request_authorized is False


def test_exact_rational_half_tie_snaps_upward() -> None:
    declaration = {
        "version": 1,
        "allowed_modes": ["image"],
        "allowed_preset_ids": ["1:1"],
        "combinations": [
            _combination("image", "preset", default=(64, 128), buckets=[[64, 128], [192, 128]])
        ],
    }
    resolved = resolve_output_geometry(
        declare_output_geometry(declaration),
        {"mode": "image", "size_mode": "preset", "preset_id": "1:1"},
    )
    assert (resolved.width, resolved.height) == (192, 128)


def test_workflow_native_carries_unverified_source_without_applying_fit() -> None:
    resolved = resolve_output_geometry(
        _capability(),
        {
            "mode": "image",
            "size_mode": "workflow_native",
            "source_width": 1920,
            "source_height": 1080,
        },
    )
    assert (resolved.width, resolved.height) == (512, 512)
    assert (resolved.source_width, resolved.source_height) == (1920, 1080)
    assert resolved.source_fit == "workflow_native"
    assert resolved.source_fit_applied is False
    assert resolved.source_dimensions_verified is False


def test_workflow_native_without_source_uses_default() -> None:
    resolved = resolve_output_geometry(
        _capability(), {"mode": "image", "size_mode": "workflow_native"}
    )
    assert (resolved.width, resolved.height) == (512, 512)
    assert resolved.source_width is None and resolved.source_height is None


@pytest.mark.parametrize(
    "geometry_request",
    [
        {"mode": "image", "size_mode": "exact", "width": 512, "height": 512, "preset_id": "1:1"},
        {"mode": "image", "size_mode": "exact", "width": 512, "height": 512, "source_width": 512},
        {"mode": "image", "size_mode": "preset", "preset_id": "1:1", "width": 512},
        {"mode": "image", "size_mode": "preset", "preset_id": "1:1", "source_width": 512},
        {"mode": "image", "size_mode": "workflow_native", "source_width": 512},
        {"mode": "image", "size_mode": "workflow_native", "source_height": 512},
        {"mode": "image", "size_mode": "workflow_native", "source_fit": "workflow_native"},
        {"mode": "image", "size_mode": "exact", "width": 512},
        {"mode": "image", "size_mode": "preset"},
        {"mode": "image", "size_mode": "unknown"},
        {"mode": "video", "size_mode": "preset", "preset_id": "1:1"},
        {"mode": "text", "size_mode": "exact", "width": 512, "height": 512},
    ],
)
def test_required_forbidden_key_matrix_rejects_ignored_fields(geometry_request: object) -> None:
    _request_refuses(geometry_request)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"version": 1},
        {**_declaration(), "unknown": False},
        {**_declaration(), "version": True},
        {**_declaration(), "version": 2},
        {**_declaration(), "allowed_modes": ["video", "image"]},
        {**_declaration(), "allowed_modes": ["image", "image"]},
        {**_declaration(), "allowed_modes": ["audio"]},
        {**_declaration(), "allowed_preset_ids": ["5:4"]},
        {**_declaration(), "combinations": []},
    ],
)
def test_unknown_missing_order_duplicates_and_closed_values_refuse(value: object) -> None:
    _refuses(value)


def test_exact_builtin_types_and_subclasses_are_required() -> None:
    class DictSubclass(dict[str, object]):
        pass

    class ListSubclass(list[object]):
        pass

    class StringSubclass(str):
        pass

    _refuses(DictSubclass(_declaration()))
    _refuses({**_declaration(), "allowed_modes": ListSubclass(["image", "video"])})
    _refuses({**_declaration(), "allowed_modes": [StringSubclass("image"), "video"]})
    declaration = _declaration()
    declaration["combinations"] = [DictSubclass(_combination("image", "exact"))]
    declaration["allowed_modes"] = ["image"]
    declaration["allowed_preset_ids"] = []
    _refuses(declaration)
    _request_refuses(
        DictSubclass({"mode": "image", "size_mode": "exact", "width": 512, "height": 512})
    )


@pytest.mark.parametrize(
    "change",
    [
        {"default_width": 576},
        {"width_multiple": 0},
        {"max_width": MAX_DIMENSION + 1},
        {"max_pixels": MAX_PIXELS + 1},
        {"min_aspect": [2, 4]},
        {"min_aspect": [5, 1], "max_aspect": [4, 1]},
        {"buckets": [[768, 1024], [512, 512], [1024, 768]]},
        {"buckets": [[512, 512], [512, 512]]},
        {"buckets": [[65, 64], [512, 512]]},
    ],
)
def test_defaults_ratios_buckets_bounds_and_order_refuse(change: dict[str, object]) -> None:
    combination = _combination("image", "exact")
    combination.update(change)
    _refuses(
        {
            "version": 1,
            "allowed_modes": ["image"],
            "allowed_preset_ids": [],
            "combinations": [combination],
        }
    )


def test_exact_resolution_enforces_declared_max_pixels() -> None:
    combination = _combination("image", "exact", buckets=[[512, 512]])
    combination["max_pixels"] = 512 * 512
    capability = declare_output_geometry(
        {
            "version": 1,
            "allowed_modes": ["image"],
            "allowed_preset_ids": [],
            "combinations": [combination],
        }
    )
    with pytest.raises(OutputGeometryError):
        resolve_output_geometry(
            capability,
            {"mode": "image", "size_mode": "exact", "width": 1024, "height": 512},
        )


def test_bucket_count_boundary_and_plus_one() -> None:
    buckets = [[64, 64 + 64 * index] for index in range(MAX_GEOMETRY_BUCKETS)]
    combination = _combination("image", "exact", default=(64, 64), buckets=buckets)
    combination["max_height"] = 64 * MAX_GEOMETRY_BUCKETS
    combination["max_pixels"] = 64 * 64 * MAX_GEOMETRY_BUCKETS
    combination["min_aspect"] = [1, MAX_GEOMETRY_BUCKETS]
    declaration = {
        "version": 1,
        "allowed_modes": ["image"],
        "allowed_preset_ids": [],
        "combinations": [combination],
    }
    assert len(declare_output_geometry(declaration).combinations[0].buckets) == MAX_GEOMETRY_BUCKETS
    combination["buckets"] = buckets + [[128, 128]]
    _refuses(declaration)


def test_source_dimensions_enforce_exact_types_bounds_and_pixels() -> None:
    for value in (True, 0, MAX_DIMENSION + 1):
        _request_refuses(
            {
                "mode": "image",
                "size_mode": "workflow_native",
                "source_width": value,
                "source_height": 512,
            }
        )
    _request_refuses(
        {
            "mode": "image",
            "size_mode": "workflow_native",
            "source_width": MAX_DIMENSION,
            "source_height": MAX_DIMENSION,
        }
    )
