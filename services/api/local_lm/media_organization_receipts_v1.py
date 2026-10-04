"""Validate saved organization previews and content-free committed results."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .models import MediaOrganizationImpact

OrganizationAction = Literal[
    "add-to-album",
    "remove-from-album",
    "reorder-album",
    "add-tag",
    "remove-tag",
    "rename-album",
    "rename-tag",
    "delete-album",
    "delete-tag",
    "merge-tags",
    "set-favorite",
    "trash",
]
OrganizationIdentity = Annotated[StrictStr, Field(pattern=r"^orgimp_[0-9a-f]{32}$")]
OrganizationCount = Annotated[StrictInt, Field(ge=0, le=100_000)]


class OrganizationPreviewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: OrganizationIdentity
    action: OrganizationAction
    selected_count: OrganizationCount
    size_bytes: Annotated[StrictInt, Field(ge=0, le=9_223_372_036_854_775_807)]
    changed_count: OrganizationCount
    expires_at: Annotated[StrictStr, Field(min_length=1, max_length=40)]


class OrganizationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: OrganizationIdentity
    action: OrganizationAction
    selected_count: OrganizationCount
    changed_count: OrganizationCount
    target_version: Annotated[StrictInt, Field(ge=1, le=9_223_372_036_854_775_807)] | None
    deletion_ids: (
        Annotated[
            list[Annotated[StrictStr, Field(pattern=r"^recover_[0-9a-f]{32}$")]],
            Field(min_length=1, max_length=100),
        ]
        | None
    ) = None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def organization_input_digest(impact: MediaOrganizationImpact) -> str:
    return _digest(
        {
            "id": impact.id,
            "request": impact.request_json,
            "preview": impact.preview_json,
            "fingerprint": impact.fingerprint,
            "recovery": impact.recovery_json,
        }
    )


def organization_result_digest(impact: MediaOrganizationImpact) -> str:
    return _digest(
        {
            "id": impact.id,
            "operation_key": impact.operation_key,
            "result": impact.response_json,
        }
    )


def validate_saved_preview(impact: MediaOrganizationImpact, action: str) -> OrganizationPreviewV1:
    if impact.input_sha256 != organization_input_digest(impact):
        raise ValueError("organization-input-invalid")
    preview = OrganizationPreviewV1.model_validate(impact.preview_json)
    expires = datetime.fromisoformat(preview.expires_at)
    if expires.tzinfo is None or expires.utcoffset() != UTC.utcoffset(expires):
        raise ValueError("organization-expiry-invalid")
    if preview.id != impact.id or preview.action != action:
        raise ValueError("organization-preview-invalid")
    ceiling = (
        max(1, preview.selected_count)
        if action in {"rename-album", "rename-tag", "delete-album", "delete-tag"}
        else preview.selected_count
    )
    if preview.changed_count > ceiling:
        raise ValueError("organization-count-invalid")
    return preview


def validate_saved_result(
    impact: MediaOrganizationImpact,
    preview: OrganizationPreviewV1,
    expected_target_version: int | None,
) -> dict[str, object]:
    if impact.response_sha256 != organization_result_digest(impact):
        raise ValueError("organization-result-invalid")
    result = OrganizationResultV1.model_validate(impact.response_json)
    if (
        result.id != preview.id
        or result.action != preview.action
        or result.selected_count != preview.selected_count
        or result.changed_count != preview.changed_count
        or result.target_version != expected_target_version
    ):
        raise ValueError("organization-result-invalid")
    if result.action == "trash":
        ids = result.deletion_ids
        if ids is None or len(set(ids)) != len(ids) or len(ids) != result.selected_count:
            raise ValueError("organization-result-invalid")
    elif "deletion_ids" in result.model_fields_set:
        raise ValueError("organization-result-invalid")
    return result.model_dump(exclude_unset=True)
