"""Canonical keys for persisted generation settings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, ValidationInfo

MAX_SETTING_FIELDS = 256


def unusable_as_a_number(candidate: int) -> bool:
    """Whether this whole number cannot be used as a number at all.

    A JSON integer has no size limit and a float does, so a large enough one
    cannot be converted - and several places convert, to compare a declared
    bound against an engine's own. The conversion raises from code that had
    already accepted the value, which is how a frame rate of four hundred digits
    became a server error rather than a refusal naming the field.

    The question is convertibility and nothing else. Whether a number survives a
    browser's own JSON round trip with every digit intact is a different and
    narrower question, and answering it here would refuse workflows that work:
    a control declared with a maximum of ten to the twentieth resolves and
    compiles perfectly well, and the browser spells that bound without an
    exponent, so the same value would be accepted or refused depending only on
    how it was written down.
    """
    try:
        return not math.isfinite(float(candidate))
    except OverflowError:
        return True


def normalize_saved_settings(values: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Copy a saved layer without changing explicit current values or unknown keys."""
    result = dict(values)
    # Leave oversized bags intact so the existing strict size check still refuses them.
    if role == "video" and "guidance" in result and len(result) <= MAX_SETTING_FIELDS:
        previous = result.pop("guidance")
        if "cfg" not in result:
            result["cfg"] = previous
    return result


def _role_settings(values: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
    return normalize_saved_settings(values, (info.data or {}).get("role", ""))


SavedRoleSettings = Annotated[dict[str, Any], AfterValidator(_role_settings)]
RoleName = Literal["chat", "image", "video"]
RoleSettings = dict[RoleName, dict[str, Any]]


def _scoped_settings(values: RoleSettings) -> RoleSettings:
    return {role: normalize_saved_settings(settings, role) for role, settings in values.items()}


GenerationSettingsByRole = Annotated[RoleSettings, AfterValidator(_scoped_settings)]
