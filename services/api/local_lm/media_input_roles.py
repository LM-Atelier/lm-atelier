"""Explicit image purposes stay aligned with the selected artifact order."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from .media_image_bindings import validate_image_bindings
from .workflow_image_slots_v1 import verify_image_slots

ImageInputRole = Literal["edit_source", "reference"]
ImageSlotRole = Literal["edit_source", "reference", "unknown"]
_SLOT_NAME = re.compile(r"input_image(?:_(0|[1-9][0-9]?))?\Z")


def workflow_image_role_bindings(
    artifact_ids: list[str],
    roles: list[ImageInputRole] | None,
    input_schema: Mapping[str, Any],
    workflow: dict[str, Any],
) -> dict[str, tuple[int, ...]] | None:
    """Bind explicit purposes only to a complete slot contract on the executing graph."""
    if roles is None:
        return None
    slots = verify_image_slots(input_schema, workflow)
    bindings = image_role_bindings(
        artifact_ids, roles, {name: slot.role for name, slot in slots.items()}
    )
    if bindings is None:
        raise ValueError("Accepted image roles are unavailable.")
    return validate_image_bindings(workflow, bindings, len(artifact_ids))


def parse_input_image_roles(artifact_ids: list[str], value: object) -> list[ImageInputRole] | None:
    """Read a recorded role vector without accepting a partial or changed selection."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Accepted image roles are unavailable.")
    try:
        roles = TypeAdapter(list[ImageInputRole]).validate_python(value)
        validate_input_image_roles(artifact_ids, roles)
    except (ValueError, TypeError, ValidationError):
        raise ValueError("Accepted image roles are unavailable.") from None
    return roles


def image_role_bindings(
    artifact_ids: list[str],
    roles: list[ImageInputRole] | None,
    slot_roles: Mapping[str, ImageSlotRole],
) -> dict[str, tuple[int, ...]] | None:
    """Put each selected picture in a known slot for its purpose, preserving reference order."""
    validate_input_image_roles(artifact_ids, roles)
    if roles is None:
        return None
    ordered: list[tuple[int, str]] = []
    for name, role in slot_roles.items():
        match = _SLOT_NAME.fullmatch(name)
        if (
            match is None
            or int(match.group(1) or 0) >= 64
            or role not in ("edit_source", "reference")
        ):
            raise ValueError(
                "Choose a workflow with separate picture-to-edit and reference inputs."
            )
        ordered.append((int(match.group(1) or 0), name))
    result: dict[str, tuple[int, ...]] = {}
    for role in ("edit_source", "reference"):
        indices = [index for index, selected in enumerate(roles) if selected == role]
        names = [name for _, name in sorted(ordered) if slot_roles[name] == role]
        if len(indices) != len(names):
            raise ValueError("The selected pictures do not match this workflow's image purposes.")
        result.update({name: (index,) for name, index in zip(names, indices, strict=True)})
    return result


def validate_input_image_roles(artifact_ids: list[str], roles: list[ImageInputRole] | None) -> None:
    """Reject an ambiguous source or a role without its exact selected image."""
    if roles is None:
        return
    if len(artifact_ids) != len(roles) or len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("Image roles must match the selected images in order.")
    if any(role not in {"edit_source", "reference"} for role in roles):
        raise ValueError("An image role is unsupported.")
    if roles.count("edit_source") > 1:
        raise ValueError("Choose one picture to edit; the others can be references.")


def image_source_index(artifact_ids: list[str], roles: list[ImageInputRole] | None) -> int | None:
    """Locate an explicit edit source, preserving the older first-picture convention."""
    validate_input_image_roles(artifact_ids, roles)
    if roles is None:
        return 0 if artifact_ids else None
    return roles.index("edit_source") if "edit_source" in roles else None


def resolved_image_roles(
    artifact_ids: list[str],
    roles: list[ImageInputRole] | None,
    dependency_ids: list[str],
    dependency_roles: Mapping[str, ImageInputRole],
) -> tuple[list[str], list[ImageInputRole] | None]:
    """Join accepted pictures and dependency pictures without changing their purposes."""
    validate_input_image_roles(artifact_ids, roles)
    identities = list(dict.fromkeys([*artifact_ids, *dependency_ids]))
    if roles is None:
        return identities, None
    by_id = dict(zip(artifact_ids, roles, strict=True))
    for identity in dependency_ids:
        role = dependency_roles.get(identity)
        if role is None or (identity in by_id and by_id[identity] != role):
            raise ValueError("Accepted dependency image purposes are unavailable.")
        by_id[identity] = role
    result = [by_id[identity] for identity in identities]
    validate_input_image_roles(identities, result)
    return identities, result
