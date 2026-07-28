from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .domain import Operation
from .schemas import SettingField

ESTIMATOR_VERSION = "prompt-edit-strength-v1"
STRENGTH_PARAMETER = "denoise"


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

    @property
    def default_applied(self) -> bool:
        return self.mode == EditStrengthMode.AUTO

    def provenance(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode.value,
            "parameter": STRENGTH_PARAMETER,
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
)
_GLOBAL_WORDS = {
    "recompose",
    "restyle",
    "stylize",
    "transform",
    "watercolor",
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
)
_REPLACEMENT_TARGETS = {
    "background",
    "clothes",
    "clothing",
    "coat",
    "dress",
    "hair",
    "hairstyle",
    "jacket",
    "object",
    "outfit",
    "shirt",
    "suit",
    "wardrobe",
}
_REPLACEMENT_VERBS = {"change", "dress", "give", "make", "replace", "swap"}
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
)
_MINIMAL_WORDS = {
    "brightness",
    "contrast",
    "exposure",
    "lighting",
    "sharpen",
    "slight",
    "slightly",
    "subtle",
}
_PRESERVATION_PHRASES = (
    "do not alter",
    "do not change",
    "don t alter",
    "don t change",
    "keep everything else",
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


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return round(min(max(value, minimum), maximum), 4)


def _strength_bounds(fields: Sequence[SettingField]) -> tuple[float, float]:
    field = next((item for item in fields if item.key == STRENGTH_PARAMETER), None)
    minimum = float(field.minimum) if field and field.minimum is not None else 0.0
    maximum = float(field.maximum) if field and field.maximum is not None else 1.0
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return minimum, maximum


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
    if _has_phrase(normalized, _GLOBAL_PHRASES) or words & _GLOBAL_WORDS:
        scope = EditScope.GLOBAL
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.GLOBAL_TRANSFORMATION)
    elif preservation_signal and minimal_signal:
        scope = EditScope.MINIMAL
        confidence = EditConfidence.HIGH
        reasons.insert(0, EditReason.MINIMAL_ADJUSTMENT)
    elif _has_phrase(normalized, _REPLACEMENT_PHRASES) or (
        bool(words & _REPLACEMENT_TARGETS) and bool(words & _REPLACEMENT_VERBS)
    ):
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
) -> ImageEditStrengthResolution | None:
    if operation != Operation.IMAGE_TO_IMAGE:
        return None

    minimum, maximum = _strength_bounds(fields)
    explicit_source = next(
        (
            source
            for source, layer in reversed(explicit_layers)
            if isinstance(layer.get(STRENGTH_PARAMETER), int | float)
            and not isinstance(layer.get(STRENGTH_PARAMETER), bool)
            and minimum <= float(layer[STRENGTH_PARAMETER]) <= maximum
        ),
        None,
    )
    if inherited_auto is not None and explicit_source == EditSettingSource.TURN:
        inherited_value = inherited_auto.get("value")
        if isinstance(inherited_value, int | float) and not isinstance(inherited_value, bool):
            value = _clamp(float(inherited_value), minimum, maximum)
            effective_settings[STRENGTH_PARAMETER] = value
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
            )

    if explicit_source is not None:
        value = float(effective_settings[STRENGTH_PARAMETER])
        effective_settings[STRENGTH_PARAMETER] = value
        return ImageEditStrengthResolution(
            mode=EditStrengthMode.MANUAL,
            value=value,
            scope=None,
            confidence=None,
            reason_codes=(EditReason.EXPLICIT_VALUE,),
            minimum=minimum,
            maximum=maximum,
            source_scope=explicit_source,
        )

    resolution = estimate_image_edit_strength(prompt, minimum=minimum, maximum=maximum)
    effective_settings[STRENGTH_PARAMETER] = resolution.value
    return resolution
