from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_lm.domain import Operation
from local_lm.image_edit_strength import (
    ESTIMATOR_VERSION,
    STRENGTH_MODE_PARAMETER,
    EditSettingSource,
    estimate_image_edit_strength,
    resolve_image_edit_strength,
)
from local_lm.schemas import SettingField

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "image_edit_strength_v1.json").read_text(encoding="utf-8")
)
_FIELDS = [
    SettingField(
        key="denoise",
        label="Change strength",
        type="number",
        default=1,
        minimum=0,
        maximum=1,
        step=0.01,
        scope="workflow",
    )
]


@pytest.mark.parametrize("case", _FIXTURES, ids=lambda case: case["name"])
def test_synthetic_prompt_fixture(case: dict[str, object]) -> None:
    prompt = str(case["prompt"])
    first = estimate_image_edit_strength(prompt)
    second = estimate_image_edit_strength(prompt)

    assert first == second
    assert first.scope == case["scope"]
    assert first.confidence == case["confidence"]
    assert float(case["minimum"]) <= first.value <= float(case["maximum"])
    assert [reason.value for reason in first.reason_codes] == case["reasons"]
    provenance = first.provenance()
    assert provenance["estimator_version"] == ESTIMATOR_VERSION
    assert prompt not in json.dumps(provenance)


def test_manual_turn_strength_remains_exact_and_authoritative() -> None:
    settings = {"denoise": 0.623456}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Restyle the scene",
        _FIELDS,
        settings,
        (
            (EditSettingSource.PROFILE_REQUEST, {"denoise": 0.41}),
            (EditSettingSource.TURN, {"denoise": 0.623456}),
        ),
    )

    assert resolution is not None
    assert settings["denoise"] == 0.623456
    assert resolution.provenance() == {
        "mode": "manual",
        "parameter": "denoise",
        "value": 0.623456,
        "applied_bounds": {"minimum": 0.0, "maximum": 1.0},
        "reason_codes": ["explicit_value"],
        "reused": False,
        "source_scope": "turn",
    }


@pytest.mark.parametrize("source", list(EditSettingSource))
def test_each_numeric_settings_scope_is_authoritative(source: EditSettingSource) -> None:
    settings = {"denoise": 0.47}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Restyle the entire image",
        _FIELDS,
        settings,
        ((source, {"denoise": 0.47}),),
    )

    assert resolution is not None
    assert resolution.value == 0.47
    assert resolution.source_scope == source
    assert resolution.provenance()["mode"] == "manual"


def test_scoped_auto_mode_overrides_lower_manual_strength() -> None:
    settings = {"denoise": 0.47}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        _FIELDS,
        settings,
        (
            (EditSettingSource.PROFILE_REQUEST, {"denoise": 0.47}),
            (EditSettingSource.CHAT, {STRENGTH_MODE_PARAMETER: "auto"}),
        ),
    )

    assert resolution is not None
    assert resolution.value == 0.66
    assert resolution.provenance()["mode"] == "auto"
    assert resolution.provenance()["reason_codes"] == [
        "subject_replacement",
        "explicit_auto_mode",
    ]


def test_turn_numeric_strength_overrides_scoped_auto_mode() -> None:
    settings = {"denoise": 0.61}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        _FIELDS,
        settings,
        (
            (EditSettingSource.CHAT, {STRENGTH_MODE_PARAMETER: "auto"}),
            (EditSettingSource.TURN, {"denoise": 0.61}),
        ),
    )

    assert resolution is not None
    assert resolution.value == 0.61
    assert resolution.provenance()["mode"] == "manual"
    assert resolution.source_scope == EditSettingSource.TURN


def test_invalid_stored_strength_does_not_suppress_auto() -> None:
    settings = {"denoise": 1}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        _FIELDS,
        settings,
        ((EditSettingSource.PROFILE_REQUEST, {"denoise": "invalid"}),),
    )

    assert resolution is not None
    assert resolution.value == 0.66
    assert resolution.provenance()["mode"] == "auto"


def test_profile_strength_is_manual_when_no_turn_override() -> None:
    settings = {"denoise": 0.47}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        _FIELDS,
        settings,
        ((EditSettingSource.PROFILE_REQUEST, {"denoise": 0.47}),),
    )

    assert resolution is not None
    assert resolution.value == 0.47
    assert resolution.source_scope == EditSettingSource.PROFILE_REQUEST


def test_auto_strength_respects_workflow_bounds() -> None:
    fields = [_FIELDS[0].model_copy(update={"minimum": 0.6, "maximum": 0.7})]
    settings = {"denoise": 1}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Restyle the entire image",
        fields,
        settings,
        (),
    )

    assert resolution is not None
    assert resolution.value == 0.7
    assert settings["denoise"] == 0.7
    assert resolution.provenance()["applied_bounds"] == {"minimum": 0.6, "maximum": 0.7}


def test_inherited_auto_strength_is_reused() -> None:
    settings = {"denoise": 0.66}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "A materially different edited prompt",
        _FIELDS,
        settings,
        ((EditSettingSource.TURN, {"denoise": 0.66}),),
        inherited_auto={
            "mode": "auto",
            "value": 0.66,
            "scope": "replacement",
            "confidence": "high",
        },
    )

    assert resolution is not None
    assert resolution.value == 0.66
    assert resolution.reused is True
    assert resolution.provenance()["reason_codes"] == ["inherited_auto_value"]


def test_text_to_image_settings_are_unchanged() -> None:
    settings = {"denoise": 0.41}
    resolution = resolve_image_edit_strength(
        Operation.TEXT_TO_IMAGE,
        "Restyle the entire image",
        _FIELDS,
        settings,
        ((EditSettingSource.CHAT, {"denoise": 0.41}),),
    )

    assert resolution is None
    assert settings == {"denoise": 1}


def test_large_adversarial_prompt_is_handled_without_regex_backtracking() -> None:
    prompt = ("do not change " * 20_000) + "slightly brighten the lighting"
    resolution = estimate_image_edit_strength(prompt)

    assert resolution.scope == "minimal"
    assert resolution.value == 0.38


def _calibrated_schema(
    *,
    parameter: str = "strength",
    replacement: float = 0.6,
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            parameter: {
                "type": "number",
                "default": 0.9,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "steps": {"type": "integer", "default": 4},
        },
        "x-lm-atelier-edit-calibration": {
            "version": 1,
            "edit_strength": {
                "parameter": parameter,
                "minimum": 0.0,
                "maximum": 1.0,
                "recommended": {
                    "minimal": 0.3,
                    "localized": 0.45,
                    "replacement": replacement,
                    "global": 0.8,
                    "fallback": 0.5,
                },
            },
            "schedule": {
                "steps_parameter": "steps",
                "minimum_effective_steps": {
                    "localized": 2,
                    "replacement": 3,
                    "global": 3,
                },
            },
        },
    }


def test_workflow_calibration_uses_custom_parameter_and_short_step_budget() -> None:
    fields = [
        SettingField(
            key="strength",
            label="Change strength",
            type="number",
            default=0.9,
            minimum=0,
            maximum=1,
            step=0.01,
            scope="workflow",
        ),
        SettingField(
            key="steps",
            label="Steps",
            type="integer",
            default=4,
            minimum=1,
            maximum=100,
            step=1,
            scope="workflow",
        ),
    ]
    settings = {"strength": 0.9, "steps": 4}

    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        fields,
        settings,
        (),
        workflow_schema=_calibrated_schema(),
    )

    assert resolution is not None
    assert resolution.value == 0.75
    assert settings == {"strength": 0.75, "steps": 4}
    provenance = resolution.provenance()
    assert provenance["parameter"] == "strength"
    assert provenance["reason_codes"] == [
        "subject_replacement",
        "workflow_calibration",
        "schedule_minimum_applied",
    ]
    assert provenance["workflow_calibration"]["version"] == 1
    assert len(provenance["workflow_calibration"]["hash"]) == 64
    assert provenance["schedule_adjustment"] == {
        "steps_parameter": "steps",
        "resolved_steps": 4.0,
        "minimum_effective_steps": 3,
        "value_before": 0.6,
        "value_after": 0.75,
        "effective_steps": 3.0,
    }


def test_runtime_bounds_win_over_calibration_and_bound_schedule_adjustment() -> None:
    fields = [
        SettingField(
            key="strength",
            label="Change strength",
            type="number",
            default=0.7,
            minimum=0.2,
            maximum=0.7,
            step=0.01,
            scope="workflow",
        ),
        SettingField(
            key="steps",
            label="Steps",
            type="integer",
            default=2,
            minimum=1,
            maximum=100,
            step=1,
            scope="workflow",
        ),
    ]
    settings = {"strength": 0.7, "steps": 2}

    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        fields,
        settings,
        (),
        workflow_schema=_calibrated_schema(replacement=0.6),
    )

    assert resolution is not None
    assert resolution.value == 0.7
    assert resolution.maximum == 0.7
    assert settings["strength"] == 0.7
    assert resolution.provenance()["schedule_adjustment"]["effective_steps"] == 1.4


def test_calibration_never_changes_an_explicit_manual_value() -> None:
    fields = [
        SettingField(
            key="strength",
            label="Change strength",
            type="number",
            default=0.9,
            minimum=0,
            maximum=1,
            step=0.01,
            scope="workflow",
        ),
        SettingField(
            key="steps",
            label="Steps",
            type="integer",
            default=1,
            minimum=1,
            maximum=100,
            step=1,
            scope="workflow",
        ),
    ]
    settings = {"strength": 0.43, "steps": 1}

    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "Replace the jacket",
        fields,
        settings,
        ((EditSettingSource.TURN, {"strength": 0.43}),),
        workflow_schema=_calibrated_schema(),
    )

    assert resolution is not None
    assert resolution.value == 0.43
    assert resolution.provenance()["mode"] == "manual"
    assert "schedule_adjustment" not in resolution.provenance()
