from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .domain import Operation
from .schemas import SettingField
from .workflow_edit_calibration import (
    WorkflowEditCalibration,
    safe_workflow_edit_calibration,
)

ESTIMATOR_VERSION = "prompt-edit-strength-v1"
STRENGTH_PARAMETER = "denoise"
STRENGTH_MODE_PARAMETER = "_image_edit_strength_mode"


class EditStrengthMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class EditScope(StrEnum):
    MINIMAL = "minimal"
    LOCALIZED = "localized"
    REPLACEMENT = "replacement"
    GLOBAL = "global"
    FALLBACK = "fallback"


class EditConfidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EditReason(StrEnum):
    MINIMAL_ADJUSTMENT = "minimal_adjustment"
    LOCALIZED_CHANGE = "localized_change"
    SUBJECT_REPLACEMENT = "subject_replacement"
    GLOBAL_TRANSFORMATION = "global_transformation"
    PRESERVATION_REQUESTED = "preservation_requested"
    AMBIGUOUS_REQUEST = "ambiguous_request"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    INHERITED_AUTO_VALUE = "inherited_auto_value"
    EXPLICIT_VALUE = "explicit_value"
    EXPLICIT_AUTO_MODE = "explicit_auto_mode"
    WORKFLOW_CALIBRATION = "workflow_calibration"
    SCHEDULE_MINIMUM_APPLIED = "schedule_minimum_applied"


class EditSettingSource(StrEnum):
    PROFILE_LOAD = "profile_load"
    PROFILE_REQUEST = "profile_request"
    DEFAULT_PRESET = "default_preset"
    PROJECT_PRESET = "project_preset"
    PROJECT = "project"
    CHAT_PRESET = "chat_preset"
    CHAT = "chat"
    TURN = "turn"


@dataclass(frozen=True)
class ImageEditStrengthResolution:
    mode: EditStrengthMode
    value: float
    scope: EditScope | None
    confidence: EditConfidence | None
    reason_codes: tuple[EditReason, ...]
    minimum: float
    maximum: float
    source_scope: EditSettingSource | None = None
    reused: bool = False
    parameter: str = STRENGTH_PARAMETER
    calibration_version: int | None = None
    calibration_hash: str | None = None
    schedule_adjustment: dict[str, float | str] | None = None

    @property
    def default_applied(self) -> bool:
        return self.mode == EditStrengthMode.AUTO

    def provenance(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode.value,
            "parameter": self.parameter,
            "value": self.value,
            "applied_bounds": {"minimum": self.minimum, "maximum": self.maximum},
            "reason_codes": [reason.value for reason in self.reason_codes],
            "reused": self.reused,
        }
        if self.mode == EditStrengthMode.AUTO:
            result.update(
                {
                    "scope": (self.scope or EditScope.FALLBACK).value,
                    "confidence": (self.confidence or EditConfidence.LOW).value,
                    "estimator_version": ESTIMATOR_VERSION,
                }
            )
        elif self.source_scope is not None:
            result["source_scope"] = self.source_scope.value
        if self.calibration_version is not None and self.calibration_hash is not None:
            result["workflow_calibration"] = {
                "version": self.calibration_version,
                "hash": self.calibration_hash,
            }
        if self.schedule_adjustment is not None:
            result["schedule_adjustment"] = self.schedule_adjustment
        return result


_CANONICAL_STRENGTH = {
    EditScope.MINIMAL: 0.38,
    EditScope.LOCALIZED: 0.50,
    EditScope.REPLACEMENT: 0.66,
    EditScope.GLOBAL: 0.82,
    EditScope.FALLBACK: 0.56,
}

_GLOBAL_PHRASES = (
    "change the entire image",
    "change the whole image",
    "complete transformation",
    "different composition",
    "new composition",
    "new scene",
    "oil painting",
    "watercolor painting",
    "apply a shallow depth of field",
    "convert it to black and white",
    "desaturate everything",
    "extend the canvas",
    "move the subject",
    "relight the scene",
)
_GLOBAL_WORDS = {
    "recompose",
    "restyle",
    "stylize",
    "transform",
    "watercolor",
    "outpaint",
    "reframe",
    "relight",
}
_REPLACEMENT_PHRASES = (
    "change the background",
    "different background",
    "new background",
    "new clothes",
    "new clothing",
    "new hairstyle",
    "new outfit",
    "replace the background",
    "mirror the image",
    "straighten the horizon",
)
_REPLACEMENT_TARGETS = {
    "background",
    "beard",
    "blazer",
    "boy",
    "clothes",
    "clothing",
    "coat",
    "dress",
    "expression",
    "flower",
    "flowers",
    "girl",
    "glasses",
    "hair",
    "hairstyle",
    "hat",
    "hoodie",
    "jacket",
    "man",
    "necklace",
    "object",
    "outfit",
    "person",
    "scarf",
    "shirt",
    "shoes",
    "skirt",
    "smile",
    "subject",
    "sunglasses",
    "suit",
    "sweatshirt",
    "sweater",
    "top",
    "trousers",
    "wardrobe",
    "woman",
}
_REPLACEMENT_VERBS = {"change", "dress", "give", "make", "put", "replace", "swap"}
_REPLACEMENT_PAIR_WINDOW = 10
_REPLACEMENT_PAIR_BLOCKERS = {
    "keep",
    "preserve",
    "retain",
    "unchanged",
    "without",
}
_COLOR_CHANGE_VERBS = {"change", "make", "recolor", "turn"}
_COLOR_WORDS = {
    "amber",
    "beige",
    "black",
    "blonde",
    "blue",
    "brown",
    "burgundy",
    "charcoal",
    "coral",
    "cream",
    "cyan",
    "gold",
    "gray",
    "green",
    "grey",
    "indigo",
    "ivory",
    "lavender",
    "magenta",
    "maroon",
    "navy",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "teal",
    "turquoise",
    "violet",
    "white",
    "yellow",
}
_COLOR_CHANGE_GLOBAL_TARGETS = {
    "all",
    "background",
    "entire",
    "everything",
    "image",
    "photo",
    "picture",
    "scene",
    "whole",
}
_LOCALIZED_PHRASES = (
    "add a",
    "add an",
    "make it blue",
    "make it green",
    "make it red",
    "remove the",
)
_LOCALIZED_WORDS = {"add", "erase", "insert", "recolor", "remove"}
_MINIMAL_PHRASES = (
    "color correction",
    "colour correction",
    "make it brighter",
    "make it darker",
    "slightly brighter",
    "slightly darker",
    "subtle change",
    "warm lighting",
    "cool down the colors",
    "correct the white balance",
    "less grainy",
    "reduce the blue cast",
    "reduce the highlights",
    "restore the faded colors",
    "soften the harsh shadows",
)
_MINIMAL_WORDS = {
    "balance",
    "brightness",
    "cast",
    "contrast",
    "exposure",
    "grain",
    "grainy",
    "highlights",
    "lighting",
    "noise",
    "shadows",
    "sharpen",
    "slight",
    "slightly",
    "subtle",
    "subtly",
    "warmth",
}
_SELECTIVE_COLOR_WORDS = {"colorize", "colourize", "desaturate"}
_PRESERVATION_PHRASES = (
    "do not alter",
    "do not change",
    "don t alter",
    "don t change",
    "keep everything else",
    "keeping everything else",
    "keep the",
    "preserve identity",
    "preserve the rest",
    "without altering",
    "without changing",
)


def _normalize_prompt(prompt: str) -> tuple[str, set[str], bool]:
    characters: list[str] = []
    has_non_ascii_letter = False
    for character in prompt.casefold():
        if character.isascii() and character.isalnum():
            characters.append(character)
        else:
            if character.isalpha() and not character.isascii():
                has_non_ascii_letter = True
            characters.append(" ")
    normalized = " ".join("".join(characters).split())
    return normalized, set(normalized.split()), has_non_ascii_letter


def _has_phrase(normalized: str, phrases: Sequence[str]) -> bool:
    padded = f" {normalized} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _has_bounded_replacement_pair(normalized: str) -> bool:
    tokens = normalized.split()
    for verb_index, token in enumerate(tokens):
        if token not in _REPLACEMENT_VERBS:
            continue
        prefix = tokens[max(0, verb_index - 2) : verb_index]
        if "not" in prefix or "never" in prefix or prefix[-2:] == ["don", "t"]:
            continue
        end = min(len(tokens), verb_index + _REPLACEMENT_PAIR_WINDOW + 1)
        for target_index in range(verb_index + 1, end):
            candidate = tokens[target_index]
            if candidate in _REPLACEMENT_PAIR_BLOCKERS:
                break
            if candidate in _REPLACEMENT_TARGETS:
                return True
    return False


def _has_bounded_color_change(normalized: str) -> bool:
    tokens = normalized.split()
    for verb_index, token in enumerate(tokens):
        if token not in _COLOR_CHANGE_VERBS:
            continue
        window = tokens[verb_index + 1 : verb_index + _REPLACEMENT_PAIR_WINDOW + 1]
        if "new" in window or "replace" in window or "swap" in window:
            continue
        colors = set(window) & _COLOR_WORDS
        if len(colors) != 1:
            continue
        color_index = next(index for index, candidate in enumerate(window) if candidate in colors)
        if not set(window[:color_index]) & _COLOR_CHANGE_GLOBAL_TARGETS:
            return True
    return False


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return round(min(max(value, minimum), maximum), 4)


def _strength_bounds(
    fields: Sequence[SettingField],
    parameter: str,
    calibration: WorkflowEditCalibration | None = None,
) -> tuple[float, float]:
    field = next((item for item in fields if item.key == parameter), None)
    minimum = (
        float(field.minimum)
        if field and field.minimum is not None
        else calibration.minimum
        if calibration
        else 0.0
    )
    maximum = (
        float(field.maximum)
        if field and field.maximum is not None
        else calibration.maximum
        if calibration
        else 1.0
    )
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return minimum, maximum


def _calibrated_auto_resolution(
    resolution: ImageEditStrengthResolution,
    calibration: WorkflowEditCalibration | None,
    effective_settings: Mapping[str, Any],
) -> ImageEditStrengthResolution:
    if calibration is None:
        return resolution
    scope = resolution.scope or EditScope.FALLBACK
    value = _clamp(
        calibration.recommended.get(scope.value, resolution.value),
        resolution.minimum,
        resolution.maximum,
    )
    reasons = (*resolution.reason_codes, EditReason.WORKFLOW_CALIBRATION)
    schedule_adjustment: dict[str, float | str] | None = None
    steps_parameter = calibration.steps_parameter
    minimum_effective_steps = calibration.minimum_effective_steps.get(scope.value)
    raw_steps = effective_settings.get(steps_parameter) if steps_parameter else None
    if (
        minimum_effective_steps is not None
        and isinstance(raw_steps, int | float)
        and not isinstance(raw_steps, bool)
        and math.isfinite(float(raw_steps))
        and float(raw_steps) > 0
    ):
        resolved_steps = float(raw_steps)
        required_strength = minimum_effective_steps / resolved_steps
        adjusted = _clamp(max(value, required_strength), resolution.minimum, resolution.maximum)
        if adjusted > value:
            assert steps_parameter is not None
            schedule_adjustment = {
                "steps_parameter": steps_parameter,
                "resolved_steps": round(resolved_steps, 4),
                "minimum_effective_steps": minimum_effective_steps,
                "value_before": value,
                "value_after": adjusted,
                "effective_steps": round(adjusted * resolved_steps, 4),
            }
            value = adjusted
            reasons = (*reasons, EditReason.SCHEDULE_MINIMUM_APPLIED)
    return replace(
        resolution,
        value=value,
        reason_codes=reasons,
        parameter=calibration.parameter,
        calibration_version=calibration.version,
        calibration_hash=calibration.contract_hash,
        schedule_adjustment=schedule_adjustment,
    )


def estimate_image_edit_strength(
    prompt: str,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> ImageEditStrengthResolution:
    normalized, words, has_non_ascii_letter = _normalize_prompt(prompt)
    reasons: list[EditReason] = []
    if _has_phrase(normalized, _PRESERVATION_PHRASES):
        reasons.append(EditReason.PRESERVATION_REQUESTED)

    minimal_signal = _has_phrase(normalized, _MINIMAL_PHRASES) or bool(words & _MINIMAL_WORDS)
    preservation_signal = EditReason.PRESERVATION_REQUESTED in reasons
    replacement_pair = _has_bounded_replacement_pair(normalized)
    color_change = _has_bounded_color_change(normalized)
    selective_color = bool(words & _SELECTIVE_COLOR_WORDS) and bool(words & {"except", "only"})
    if selective_color:
        scope = EditScope.LOCALIZED
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.LOCALIZED_CHANGE)
    elif _has_phrase(normalized, _GLOBAL_PHRASES) or words & _GLOBAL_WORDS:
        scope = EditScope.GLOBAL
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.GLOBAL_TRANSFORMATION)
    elif preservation_signal and minimal_signal and not replacement_pair:
        scope = EditScope.MINIMAL
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.MINIMAL_ADJUSTMENT)
    elif _has_phrase(normalized, _LOCALIZED_PHRASES):
        scope = EditScope.LOCALIZED
        confidence = EditConfidence.MEDIUM
        reasons.insert(0, EditReason.LOCALIZED_CHANGE)
    elif color_change:
        scope = EditScope.LOCALIZED
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.LOCALIZED_CHANGE)
    elif _has_phrase(normalized, _REPLACEMENT_PHRASES) or replacement_pair:
        scope = EditScope.REPLACEMENT
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.SUBJECT_REPLACEMENT)
    elif minimal_signal:
        scope = EditScope.MINIMAL
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.MINIMAL_ADJUSTMENT)
    elif _has_phrase(normalized, _LOCALIZED_PHRASES) or words & _LOCALIZED_WORDS:
        scope = EditScope.LOCALIZED
        confidence = EditConfidence.MEDIUM
        reasons.insert(0, EditReason.LOCALIZED_CHANGE)
    else:
        scope = EditScope.FALLBACK
        confidence = EditConfidence.LOW
        reasons.insert(
            0,
            EditReason.UNSUPPORTED_LANGUAGE
            if has_non_ascii_letter
            else EditReason.AMBIGUOUS_REQUEST,
        )

    return ImageEditStrengthResolution(
        mode=EditStrengthMode.AUTO,
        value=_clamp(_CANONICAL_STRENGTH[scope], minimum, maximum),
        scope=scope,
        confidence=confidence,
        reason_codes=tuple(reasons),
        minimum=minimum,
        maximum=maximum,
    )


def resolve_image_edit_strength(
    operation: Operation,
    prompt: str,
    fields: Sequence[SettingField],
    effective_settings: dict[str, Any],
    explicit_layers: Sequence[tuple[EditSettingSource, Mapping[str, Any]]],
    *,
    inherited_auto: Mapping[str, Any] | None = None,
    workflow_schema: Mapping[str, Any] | None = None,
) -> ImageEditStrengthResolution | None:
    calibration = safe_workflow_edit_calibration(workflow_schema)
    parameter = calibration.parameter if calibration else STRENGTH_PARAMETER
    if operation == Operation.TEXT_TO_IMAGE:
        field = next((item for item in fields if item.key == parameter), None)
        if field is not None:
            effective_settings[parameter] = field.default
        return None
    if operation != Operation.IMAGE_TO_IMAGE:
        return None

    minimum, maximum = _strength_bounds(fields, parameter, calibration)
    explicit_source: EditSettingSource | None = None
    explicit_auto = False
    implicit_manual_sources = {
        EditSettingSource.PROJECT_PRESET,
        EditSettingSource.PROJECT,
        EditSettingSource.CHAT_PRESET,
        EditSettingSource.CHAT,
        EditSettingSource.TURN,
    }
    for source, layer in reversed(explicit_layers):
        raw_mode = layer.get(STRENGTH_MODE_PARAMETER)
        raw_strength = layer.get(parameter)
        if raw_mode == EditStrengthMode.AUTO.value:
            explicit_auto = True
            explicit_source = source
            break
        if (
            source in implicit_manual_sources
            and isinstance(raw_strength, int | float)
            and not isinstance(raw_strength, bool)
            and minimum <= float(raw_strength) <= maximum
        ):
            explicit_source = source
            break
        if raw_mode == EditStrengthMode.MANUAL.value:
            explicit_source = source
            break

    inherited_parameter = inherited_auto.get("parameter") if inherited_auto else None
    if (
        inherited_auto is not None
        and explicit_source == EditSettingSource.TURN
        and inherited_parameter in {None, parameter}
    ):
        inherited_value = inherited_auto.get("value")
        if isinstance(inherited_value, int | float) and not isinstance(inherited_value, bool):
            value = _clamp(float(inherited_value), minimum, maximum)
            effective_settings[parameter] = value
            raw_scope = inherited_auto.get("scope")
            raw_confidence = inherited_auto.get("confidence")
            try:
                scope = EditScope(raw_scope) if isinstance(raw_scope, str) else EditScope.FALLBACK
            except ValueError:
                scope = EditScope.FALLBACK
            try:
                confidence = (
                    EditConfidence(raw_confidence)
                    if isinstance(raw_confidence, str)
                    else EditConfidence.LOW
                )
            except ValueError:
                confidence = EditConfidence.LOW
            return ImageEditStrengthResolution(
                mode=EditStrengthMode.AUTO,
                value=value,
                scope=scope,
                confidence=confidence,
                reason_codes=(EditReason.INHERITED_AUTO_VALUE,),
                minimum=minimum,
                maximum=maximum,
                reused=True,
                parameter=parameter,
                calibration_version=calibration.version if calibration else None,
                calibration_hash=calibration.contract_hash if calibration else None,
            )

    if explicit_auto:
        resolution = estimate_image_edit_strength(prompt, minimum=minimum, maximum=maximum)
        resolution = replace(
            resolution,
            reason_codes=(*resolution.reason_codes, EditReason.EXPLICIT_AUTO_MODE),
        )
        resolution = _calibrated_auto_resolution(
            resolution,
            calibration,
            effective_settings,
        )
        effective_settings[parameter] = resolution.value
        return resolution

    if explicit_source is not None:
        value = float(effective_settings[parameter])
        effective_settings[parameter] = value
        return ImageEditStrengthResolution(
            mode=EditStrengthMode.MANUAL,
            value=value,
            scope=None,
            confidence=None,
            reason_codes=(EditReason.EXPLICIT_VALUE,),
            minimum=minimum,
            maximum=maximum,
            source_scope=explicit_source,
            parameter=parameter,
            calibration_version=calibration.version if calibration else None,
            calibration_hash=calibration.contract_hash if calibration else None,
        )

    resolution = estimate_image_edit_strength(prompt, minimum=minimum, maximum=maximum)
    resolution = _calibrated_auto_resolution(resolution, calibration, effective_settings)
    effective_settings[parameter] = resolution.value
    return resolution
