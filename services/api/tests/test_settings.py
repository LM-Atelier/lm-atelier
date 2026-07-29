from __future__ import annotations

import pytest

from local_lm.schemas import EngineCapabilities, SettingField
from local_lm.settings_registry import (
    CHAT_SETTINGS,
    IMAGE_SETTINGS,
    MAX_SETTING_ARRAY_ITEMS,
    MAX_SETTING_NESTING_DEPTH,
    MAX_SETTING_STRING_LENGTH,
    MAX_WORKFLOW_SCHEMA_PROPERTIES,
    VIDEO_SETTINGS,
    capability_settings_for_role,
    defaults,
    normalize_capability_settings,
    resolve_generation_settings,
    resolve_settings,
    validate_settings,
    workflow_settings,
)


def _media_capabilities(
    *,
    image: list[SettingField] | None = None,
    video: list[SettingField] | None = None,
    legacy: list[SettingField] | None = None,
) -> EngineCapabilities:
    settings_by_role = (
        {"image": image or [], "video": video or []}
        if image is not None or video is not None
        else {}
    )
    flat = legacy if legacy is not None else [*(image or []), *(video or [])]
    return EngineCapabilities(
        engine="test-media",
        version="1",
        roles=["image", "video"],
        operations=["text_to_image", "text_to_video"],
        formats=["test"],
        devices=[],
        streaming=False,
        tool_calling=False,
        settings=flat,
        settings_by_role=settings_by_role,
        healthy=True,
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_setting_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="numbers must be finite"):
        validate_settings({"temperature": value}, CHAT_SETTINGS)


def test_nested_non_finite_setting_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="numbers must be finite"):
        validate_settings({"stop": [{"weight": float("nan")}]}, CHAT_SETTINGS)


def test_setting_strings_and_arrays_have_generous_safety_bounds() -> None:
    with pytest.raises(ValueError, match="strings must be at most"):
        validate_settings(
            {"tensor_split": "1" * (MAX_SETTING_STRING_LENGTH + 1)},
            CHAT_SETTINGS,
        )
    with pytest.raises(ValueError, match="arrays must contain at most"):
        validate_settings(
            {"stop": ["stop"] * (MAX_SETTING_ARRAY_ITEMS + 1)},
            CHAT_SETTINGS,
        )


def test_nested_custom_setting_values_have_a_depth_bound() -> None:
    nested: dict[str, object] = {}
    for _ in range(MAX_SETTING_NESTING_DEPTH + 1):
        nested = {"next": nested}
    field = SettingField(
        key="custom",
        label="Custom",
        type="object",
        default={},
        scope="workflow",
    )

    with pytest.raises(ValueError, match="nested too deeply"):
        validate_settings({"custom": nested}, [field])


def test_workflow_custom_controls_have_a_property_bound() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"control_{index}": {"type": "number", "default": 0}
            for index in range(MAX_WORKFLOW_SCHEMA_PROPERTIES + 1)
        },
    }

    with pytest.raises(ValueError, match="must declare at most"):
        workflow_settings(VIDEO_SETTINGS, schema)


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


def test_workflow_read_only_controls_drop_obsolete_overrides() -> None:
    fields = workflow_settings(
        IMAGE_SETTINGS,
        {
            "type": "object",
            "properties": {
                "steps": {"readOnly": True},
            },
        },
    )

    assert "steps" not in {field.key for field in fields}
    resolved = resolve_generation_settings(
        fields,
        profile_defaults=[{"steps": 40}],
    )
    assert "steps" not in resolved
    with pytest.raises(ValueError, match="unsupported settings: steps"):
        validate_settings({"steps": 40}, fields)


def test_persisted_generation_settings_resolve_by_scope() -> None:
    request_fields = [field for field in CHAT_SETTINGS if field.scope != "load"]
    resolved = resolve_generation_settings(
        CHAT_SETTINGS,
        request_fields=request_fields,
        profile_defaults=[
            {"context_length": 16_384, "temperature": 0.6},
            {"temperature": 0.5},
        ],
        project_defaults=[{"temperature": 0.4, "max_tokens": 400}],
        chat_defaults=[{"temperature": 0.3, "max_tokens": 300}],
        turn_overrides={"max_tokens": 200},
    )

    assert resolved["context_length"] == 16_384
    assert resolved["temperature"] == 0.3
    assert resolved["max_tokens"] == 200


def test_obsolete_persisted_values_do_not_break_resolution() -> None:
    request_fields = [field for field in CHAT_SETTINGS if field.scope != "load"]
    resolved = resolve_generation_settings(
        CHAT_SETTINGS,
        request_fields=request_fields,
        chat_defaults=[
            {
                "unknown_old_setting": True,
                "temperature": 99,
                "max_tokens": 321,
                "context_length": 512,
            }
        ],
    )

    assert resolved["temperature"] == 0.7
    assert resolved["max_tokens"] == 321
    assert resolved["context_length"] == 8192


def test_capability_settings_are_role_aware_and_preserve_builtins() -> None:
    image_quality = SettingField(
        key="image_quality",
        label="Image quality",
        type="enum",
        default="balanced",
        choices=["fast", "balanced"],
        scope="workflow",
    )
    motion_curve = SettingField(
        key="motion_curve",
        label="Motion curve",
        type="number",
        default=0.5,
        minimum=0,
        maximum=1,
        scope="workflow",
    )
    narrow_width = next(field for field in IMAGE_SETTINGS if field.key == "width").model_copy(
        update={"maximum": 2048}
    )
    unavailable_sampler = next(
        field for field in IMAGE_SETTINGS if field.key == "sampler"
    ).model_copy(
        update={
            "available": False,
            "unavailable_reason": "This adapter chooses the sampler.",
        }
    )
    capabilities = _media_capabilities(
        image=[narrow_width, unavailable_sampler, image_quality],
        video=[motion_curve],
    )

    normalized = normalize_capability_settings(capabilities)
    image_fields = capability_settings_for_role(normalized, "image")
    video_fields = capability_settings_for_role(normalized, "video")

    assert {field.key for field in IMAGE_SETTINGS} <= {field.key for field in image_fields}
    assert {field.key for field in VIDEO_SETTINGS} <= {field.key for field in video_fields}
    assert next(field for field in image_fields if field.key == "width").maximum == 2048
    assert next(field for field in image_fields if field.key == "sampler").available is False
    assert "image_quality" in {field.key for field in image_fields}
    assert "motion_curve" not in {field.key for field in image_fields}
    assert "motion_curve" in {field.key for field in video_fields}
    assert "image_quality" not in {field.key for field in video_fields}


def test_legacy_capabilities_keep_builtins_and_bounded_custom_fields() -> None:
    custom = SettingField(
        key="legacy_control",
        label="Legacy control",
        type="boolean",
        default=True,
        scope="workflow",
    )
    capabilities = _media_capabilities(legacy=[*IMAGE_SETTINGS, *VIDEO_SETTINGS, custom])

    image_fields = capability_settings_for_role(capabilities, "image")
    video_fields = capability_settings_for_role(capabilities, "video")

    assert next(field for field in image_fields if field.key == "width").minimum == 64
    assert next(field for field in video_fields if field.key == "width").minimum == 128
    assert "frames" not in {field.key for field in image_fields}
    assert "negative_prompt" not in {field.key for field in video_fields}
    assert "legacy_control" in {field.key for field in image_fields}
    assert "legacy_control" in {field.key for field in video_fields}


@pytest.mark.parametrize(
    ("field", "message"),
    [
        (
            next(field for field in IMAGE_SETTINGS if field.key == "width").model_copy(
                update={"maximum": 8192}
            ),
            "cannot raise the built-in maximum",
        ),
        (
            next(field for field in IMAGE_SETTINGS if field.key == "width").model_copy(
                update={"type": "string", "default": "1024"}
            ),
            "cannot change the built-in type",
        ),
        (
            next(field for field in IMAGE_SETTINGS if field.key == "width").model_copy(
                update={"scope": "load"}
            ),
            "cannot change the built-in scope",
        ),
        (
            SettingField(
                key="prompt",
                label="Prompt override",
                type="string",
                default="",
                scope="workflow",
            ),
            "reserved setting key prompt",
        ),
    ],
)
def test_capability_settings_reject_weakened_or_reserved_fields(
    field: SettingField,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        capability_settings_for_role(
            _media_capabilities(image=[field], video=[]),
            "image",
        )


def test_capability_settings_reject_cross_scope_key_collisions() -> None:
    fields = [
        SettingField(
            key="adapter_mode",
            label="Adapter mode",
            type="string",
            default="one",
            scope="request",
        ),
        SettingField(
            key="adapter_mode",
            label="Adapter mode",
            type="string",
            default="two",
            scope="workflow",
        ),
    ]

    with pytest.raises(ValueError, match="conflicting definitions"):
        capability_settings_for_role(
            _media_capabilities(image=fields, video=[]),
            "image",
        )


def test_capability_custom_fields_cannot_exceed_total_role_bound() -> None:
    custom = [
        SettingField(
            key=f"custom_{index}",
            label=f"Custom {index}",
            type="boolean",
            default=False,
            scope="workflow",
        )
        for index in range(250)
    ]

    with pytest.raises(ValueError, match="after built-in controls"):
        capability_settings_for_role(
            _media_capabilities(image=custom, video=[]),
            "image",
        )


def test_workflow_schema_cannot_weaken_engine_fields_or_claim_runtime_keys() -> None:
    with pytest.raises(ValueError, match="cannot raise the engine maximum"):
        workflow_settings(
            IMAGE_SETTINGS,
            {"properties": {"width": {"type": "integer", "maximum": 8192}}},
        )
    with pytest.raises(ValueError, match="reserved setting key prompt"):
        workflow_settings(
            IMAGE_SETTINGS,
            {"properties": {"prompt": {"type": "string", "default": "override"}}},
        )
