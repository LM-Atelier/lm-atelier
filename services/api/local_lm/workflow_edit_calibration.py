from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EDIT_CALIBRATION_SCHEMA_KEY = "x-lm-atelier-edit-calibration"
EDIT_CALIBRATION_VERSION = 1
EDIT_SCOPES = ("minimal", "localized", "replacement", "global", "fallback")
_SCHEDULE_SCOPES = ("localized", "replacement", "global")
_DEFAULT_RECOMMENDED = {
    "minimal": 0.38,
    "localized": 0.50,
    "replacement": 0.66,
    "global": 0.82,
    "fallback": 0.56,
}
_DEFAULT_MINIMUM_EFFECTIVE_STEPS = {
    "localized": 2,
    "replacement": 3,
    "global": 3,
}


@dataclass(frozen=True)
class WorkflowEditCalibration:
    version: int
    parameter: str
    minimum: float
    maximum: float
    recommended: dict[str, float]
    steps_parameter: str | None
    minimum_effective_steps: dict[str, int]
    contract_hash: str


def standard_edit_calibration(
    *,
    parameter: str,
    minimum: float,
    maximum: float,
    steps_parameter: str | None,
) -> dict[str, Any]:
    recommended = {
        scope: _clamp(value, minimum, maximum) for scope, value in _DEFAULT_RECOMMENDED.items()
    }
    result: dict[str, Any] = {
        "version": EDIT_CALIBRATION_VERSION,
        "edit_strength": {
            "parameter": parameter,
            "minimum": minimum,
            "maximum": maximum,
            "recommended": recommended,
        },
    }
    if steps_parameter:
        result["schedule"] = {
            "steps_parameter": steps_parameter,
            "minimum_effective_steps": dict(_DEFAULT_MINIMUM_EFFECTIVE_STEPS),
        }
    return result


def validate_workflow_edit_calibration(
    input_schema: Mapping[str, Any],
) -> WorkflowEditCalibration | None:
    raw = input_schema.get(EDIT_CALIBRATION_SCHEMA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("workflow edit calibration must be an object")
    _require_keys(raw, {"version", "edit_strength", "schedule"}, "workflow edit calibration")
    if raw.get("version") != EDIT_CALIBRATION_VERSION:
        raise ValueError(f"workflow edit calibration version must be {EDIT_CALIBRATION_VERSION}")

    strength = raw.get("edit_strength")
    if not isinstance(strength, Mapping):
        raise ValueError("workflow edit calibration edit_strength must be an object")
    _require_keys(
        strength,
        {"parameter", "minimum", "maximum", "recommended"},
        "workflow edit calibration edit_strength",
    )
    parameter = _parameter(strength.get("parameter"), "edit strength")
    minimum = _finite_number(strength.get("minimum"), "edit strength minimum")
    maximum = _finite_number(strength.get("maximum"), "edit strength maximum")
    if minimum >= maximum:
        raise ValueError("workflow edit calibration minimum must be less than maximum")

    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("workflow edit calibration requires input schema properties")
    parameter_schema = properties.get(parameter)
    if not isinstance(parameter_schema, Mapping) or parameter_schema.get("type") != "number":
        raise ValueError(
            "workflow edit calibration parameter must identify a numeric workflow setting"
        )
    if not any(key in parameter_schema for key in ("default", "const")):
        raise ValueError(
            "workflow edit calibration parameter must declare a default or const value"
        )

    raw_recommended = strength.get("recommended")
    if not isinstance(raw_recommended, Mapping):
        raise ValueError("workflow edit calibration recommended values must be an object")
    _require_keys(
        raw_recommended,
        set(EDIT_SCOPES),
        "workflow edit calibration recommended values",
        required=set(EDIT_SCOPES[:4]),
    )
    recommended: dict[str, float] = {}
    for scope in EDIT_SCOPES:
        if scope not in raw_recommended:
            continue
        value = _finite_number(
            raw_recommended[scope],
            f"workflow edit calibration {scope} recommendation",
        )
        if not minimum <= value <= maximum:
            raise ValueError(
                f"workflow edit calibration {scope} recommendation is outside its bounds"
            )
        recommended[scope] = value

    steps_parameter: str | None = None
    minimum_effective_steps: dict[str, int] = {}
    schedule = raw.get("schedule")
    if schedule is not None:
        if not isinstance(schedule, Mapping):
            raise ValueError("workflow edit calibration schedule must be an object")
        _require_keys(
            schedule,
            {"steps_parameter", "minimum_effective_steps"},
            "workflow edit calibration schedule",
            required={"steps_parameter", "minimum_effective_steps"},
        )
        steps_parameter = _parameter(schedule.get("steps_parameter"), "schedule steps")
        steps_schema = properties.get(steps_parameter)
        if (
            not isinstance(steps_schema, Mapping)
            or steps_schema.get("type") not in {"integer", "number"}
            or not any(key in steps_schema for key in ("default", "const"))
        ):
            raise ValueError(
                "workflow edit calibration steps parameter must identify a numeric "
                "workflow setting with a default or const value"
            )
        raw_minimum_steps = schedule.get("minimum_effective_steps")
        if not isinstance(raw_minimum_steps, Mapping):
            raise ValueError("workflow edit calibration minimum_effective_steps must be an object")
        _require_keys(
            raw_minimum_steps,
            set(_SCHEDULE_SCOPES),
            "workflow edit calibration minimum_effective_steps",
            required=set(_SCHEDULE_SCOPES),
        )
        for scope in _SCHEDULE_SCOPES:
            value = raw_minimum_steps[scope]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
                raise ValueError(
                    "workflow edit calibration minimum effective steps must be "
                    "integers from 1 to 10000"
                )
            minimum_effective_steps[scope] = value

    normalized = {
        "version": EDIT_CALIBRATION_VERSION,
        "edit_strength": {
            "parameter": parameter,
            "minimum": minimum,
            "maximum": maximum,
            "recommended": recommended,
        },
        **(
            {
                "schedule": {
                    "steps_parameter": steps_parameter,
                    "minimum_effective_steps": minimum_effective_steps,
                }
            }
            if steps_parameter
            else {}
        ),
    }
    contract_hash = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return WorkflowEditCalibration(
        version=EDIT_CALIBRATION_VERSION,
        parameter=parameter,
        minimum=minimum,
        maximum=maximum,
        recommended=recommended,
        steps_parameter=steps_parameter,
        minimum_effective_steps=minimum_effective_steps,
        contract_hash=contract_hash,
    )


def safe_workflow_edit_calibration(
    input_schema: Mapping[str, Any] | None,
) -> WorkflowEditCalibration | None:
    if not input_schema:
        return None
    try:
        return validate_workflow_edit_calibration(input_schema)
    except ValueError:
        return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return round(min(max(value, minimum), maximum), 4)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _parameter(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or value.startswith("_")
        or any(character < " " and character != "\t" for character in value)
    ):
        raise ValueError(f"workflow edit calibration {label} parameter is invalid")
    return value


def _require_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
    *,
    required: set[str] | None = None,
) -> None:
    keys = set(value)
    unknown = keys - allowed
    if unknown:
        rendered = ", ".join(sorted(str(item) for item in unknown))
        raise ValueError(f"{label} contains unsupported fields: {rendered}")
    missing = (required or allowed - {"schedule"}) - keys
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
