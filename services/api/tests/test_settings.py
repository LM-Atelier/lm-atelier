from __future__ import annotations

import pytest

from local_lm.settings_registry import (
    CHAT_SETTINGS,
    VIDEO_SETTINGS,
    defaults,
    resolve_settings,
    validate_settings,
    workflow_settings,
)


def test_settings_precedence_and_validation() -> None:
    resolved = resolve_settings(
        defaults(CHAT_SETTINGS),
        {"temperature": 0.4, "max_tokens": 500},
        {"temperature": 0.2},
    )
    validated = validate_settings(resolved, CHAT_SETTINGS)
    assert validated["temperature"] == 0.2
    assert validated["max_tokens"] == 500
    assert validated["context_length"] == 8192


def test_unknown_settings_are_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="unsupported settings"):
        validate_settings({"imaginary_knob": True}, CHAT_SETTINGS)


def test_out_of_range_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature must be at most"):
        validate_settings({"temperature": 99}, CHAT_SETTINGS)


def test_workflow_schema_overlays_defaults_constraints_and_custom_controls() -> None:
    fields = workflow_settings(
        VIDEO_SETTINGS,
        {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "const": 832},
                "height": {"type": "integer", "const": 480},
                "codec": {"type": "string", "const": "h264"},
                "frames": {
                    "type": "integer",
                    "default": 81,
                    "minimum": 1,
                    "multipleOf": 4,
                },
                "camera_strength": {
                    "type": "number",
                    "title": "Camera strength",
                    "default": 0.5,
                    "minimum": 0,
                    "maximum": 1,
                },
                "input_image": {"type": "string"},
            },
        },
    )

    resolved = defaults(fields)
    assert resolved["width"] == 832
    assert resolved["height"] == 480
    assert resolved["codec"] == "h264"
    assert resolved["frames"] == 81
    assert resolved["camera_strength"] == 0.5
    assert "input_image" not in resolved
    assert next(field for field in fields if field.key == "width").choices == [832]
    assert next(field for field in fields if field.key == "camera_strength").label == (
        "Camera strength"
    )

    with pytest.raises(ValueError, match="width must be one of"):
        validate_settings({"width": 768}, fields)
    with pytest.raises(ValueError, match="frames must be a multiple of 4"):
        validate_settings({"frames": 82}, fields)


def test_empty_workflow_schema_preserves_legacy_role_settings() -> None:
    assert workflow_settings(VIDEO_SETTINGS, {}) == VIDEO_SETTINGS
