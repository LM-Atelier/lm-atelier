from __future__ import annotations

import math
from fractions import Fraction

import pytest

from local_lm.settings_registry import (
    VIDEO_SETTINGS,
    defaults,
    validate_settings,
    workflow_settings,
)
from local_lm.video_length import resolve_video_length_settings, workflow_video_length


def _schema(**contract_changes: object) -> dict[str, object]:
    contract: dict[str, object] = {
        "version": 1,
        "frames_parameter": "frames",
        "fps_parameter": "fps",
        "fps_numerator": 16,
        "fps_denominator": 1,
        "frame_alignment": 16,
        "frame_offset": 1,
    }
    contract.update(contract_changes)
    return {
        "type": "object",
        "properties": {
            "input_image": {"type": "string"},
            "frames": {
                "type": "integer",
                "default": 49,
                "minimum": 17,
                "maximum": 81,
            },
            "fps": {"type": "number", "const": 16},
        },
        "x-lm-atelier-video-length": contract,
    }


def test_workflow_contract_replaces_frame_controls_with_seconds() -> None:
    fields = workflow_settings(VIDEO_SETTINGS, _schema())
    by_key = {field.key: field for field in fields}

    assert "frames" not in by_key
    assert "fps" not in by_key
    length = by_key["duration_seconds"]
    assert length.label == "Length (seconds)"
    assert length.default == 49 / 16
    assert length.minimum == 17 / 16
    assert length.maximum == 81 / 16
    assert defaults(fields)["duration_seconds"] == 49 / 16


def test_request_resolves_to_nearest_supported_frames_and_records_both_lengths() -> None:
    settings, provenance = resolve_video_length_settings(
        {"duration_seconds": 3, "seed": 7},
        _schema(),
    )

    assert settings == {
        "duration_seconds": 3,
        "seed": 7,
        "frames": 49,
        "fps": 16,
    }
    assert provenance == {
        "requested_seconds": 3.0,
        "delivered_seconds": 49 / 16,
        "frames": 49,
        "fps": 16.0,
    }


def test_nearest_tie_uses_the_shorter_supported_clip() -> None:
    settings, provenance = resolve_video_length_settings(
        {"duration_seconds": 41 / 16},
        _schema(),
    )

    assert settings["frames"] == 33
    assert provenance is not None
    assert provenance["delivered_seconds"] == 33 / 16


def test_duration_validation_uses_the_measured_usable_window() -> None:
    fields = workflow_settings(VIDEO_SETTINGS, _schema())

    with pytest.raises(ValueError, match="duration_seconds must be at least"):
        validate_settings({"duration_seconds": 1}, fields)
    with pytest.raises(ValueError, match="duration_seconds must be at most"):
        validate_settings({"duration_seconds": 6}, fields)


def test_published_non_binary_bounds_round_trip_without_widening_the_window() -> None:
    schema = _schema(fps_numerator=30, frame_alignment=1, frame_offset=0)
    schema["properties"]["frames"] = {  # type: ignore[index]
        "type": "integer",
        "default": 151,
        "minimum": 1,
        "maximum": 311,
    }
    schema["properties"]["fps"]["const"] = 30  # type: ignore[index]
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    duration = next(field for field in fields if field.key == "duration_seconds")

    minimum, minimum_provenance = resolve_video_length_settings(
        {"duration_seconds": duration.minimum}, schema
    )
    maximum, maximum_provenance = resolve_video_length_settings(
        {"duration_seconds": duration.maximum}, schema
    )

    assert minimum["frames"] == 1
    assert minimum_provenance is not None
    assert minimum_provenance["requested_seconds"] == duration.minimum
    assert maximum["frames"] == 311
    assert maximum_provenance is not None
    assert maximum_provenance["requested_seconds"] == duration.maximum

    with pytest.raises(ValueError, match="outside this workflow's usable window"):
        resolve_video_length_settings(
            {"duration_seconds": math.nextafter(duration.minimum, -math.inf)}, schema
        )
    with pytest.raises(ValueError, match="outside this workflow's usable window"):
        resolve_video_length_settings(
            {"duration_seconds": math.nextafter(duration.maximum, math.inf)}, schema
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": 2}, "version must be 1"),
        ({"fps_numerator": 32, "fps_denominator": 2}, "FPS must be reduced"),
        ({"frame_alignment": 0}, "frame_alignment"),
        ({"frame_alignment": 16, "frame_offset": 16}, "frame_offset"),
    ],
)
def test_malformed_contracts_fail_closed(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        workflow_video_length(_schema(**change))


def test_window_and_fps_must_match_the_bound_workflow_properties() -> None:
    bad_window = _schema()
    bad_window["properties"]["frames"]["maximum"] = 80  # type: ignore[index]
    with pytest.raises(ValueError, match="frames maximum must follow its alignment"):
        workflow_video_length(bad_window)

    bad_fps = _schema(fps_numerator=24)
    with pytest.raises(ValueError, match="FPS property must match"):
        workflow_video_length(bad_fps)


def test_fractional_fps_uses_the_rational_contract_without_decimal_equality() -> None:
    schema = _schema(fps_numerator=30_000, fps_denominator=1_001)
    schema["properties"]["fps"]["const"] = 30_000 / 1_001  # type: ignore[index]

    contract = workflow_video_length(schema)

    assert contract is not None
    assert contract.fps == Fraction(30_000, 1_001)


def test_non_contract_workflows_keep_their_existing_settings() -> None:
    settings = {"frames": 49, "fps": 16}
    resolved, provenance = resolve_video_length_settings(settings, {"properties": {}})

    assert resolved == settings
    assert provenance is None
