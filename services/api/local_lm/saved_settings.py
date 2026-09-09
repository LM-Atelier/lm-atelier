"""Canonical keys for persisted generation settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, ValidationInfo

MAX_SETTING_FIELDS = 256


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
