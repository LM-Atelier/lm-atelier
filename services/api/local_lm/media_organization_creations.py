"""Confirm a catalog creation retry from its original atomic receipt."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy.orm import Session

from .artifact_library import begin_artifact_write_fence
from .media_organization import MEDIA_ORGANIZATION_INVALID, MediaOrganizationError
from .models import MediaOrganizationCreation

MEDIA_CREATION_CONFLICT = "The saved creation could not be confirmed. Refresh the catalog."


class MediaCreationConflict(ValueError):
    pass


class _AlbumResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Annotated[StrictStr, Field(pattern=r"^collection_[0-9a-f]{32}$")]
    kind: Literal["manual"]
    name: StrictStr
    description: StrictStr
    version: StrictInt


class _TagResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Annotated[StrictStr, Field(pattern=r"^mediatag_[0-9a-f]{32}$")]
    slug: Annotated[StrictStr, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    label: StrictStr
    color: Annotated[StrictStr, Field(pattern=r"^#[0-9a-f]{6}$")] | None
    version: StrictInt


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _response_digest(receipt: MediaOrganizationCreation) -> str:
    return _digest(
        {
            "key": receipt.operation_key,
            "kind": receipt.kind,
            "request": receipt.request_sha256,
            "entity": receipt.entity_id,
            "response": receipt.response_json,
        }
    )


def _validate_response(
    receipt: MediaOrganizationCreation, request: dict[str, object]
) -> dict[str, object]:
    if receipt.response_sha256 != _response_digest(receipt):
        raise ValueError("creation-receipt-invalid")
    if receipt.kind == "albums":
        album = _AlbumResponse.model_validate(receipt.response_json)
        if album.id != receipt.entity_id or album.version != 1:
            raise ValueError("creation-identity-invalid")
        if album.name != request["name"] or album.description != request["description"]:
            raise ValueError("creation-request-invalid")
        return album.model_dump()
    tag = _TagResponse.model_validate(receipt.response_json)
    if tag.id != receipt.entity_id or tag.version != 1:
        raise ValueError("creation-identity-invalid")
    if tag.label != request["label"] or tag.color != request["color"]:
        raise ValueError("creation-request-invalid")
    if tag.slug != "-".join(tag.label.casefold().split()):
        raise ValueError("creation-name-invalid")
    return tag.model_dump()


def create_catalog_choice(
    session: Session,
    *,
    kind: Literal["albums", "tags"],
    payload: dict[str, object],
    create: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Keep the choice and its retry receipt in the caller's write transaction."""
    if "operation_key" not in payload:
        return create()
    key = payload["operation_key"]
    allowed = (
        {"operation_key", "name", "description"}
        if kind == "albums"
        else {"operation_key", "label", "color"}
    )
    if type(key) is not str or re.fullmatch(r"[0-9a-f]{32}", key) is None or set(payload) - allowed:
        raise MediaOrganizationError(MEDIA_ORGANIZATION_INVALID)
    request = (
        {"name": payload.get("name"), "description": payload.get("description", "")}
        if kind == "albums"
        else {"label": payload.get("label"), "color": payload.get("color")}
    )
    try:
        request_sha256 = _digest(request)
    except (TypeError, ValueError, RecursionError) as exc:
        raise MediaOrganizationError(MEDIA_ORGANIZATION_INVALID) from exc
    begin_artifact_write_fence(session)
    existing = session.get(MediaOrganizationCreation, key)
    if existing is not None:
        if existing.kind != kind or existing.request_sha256 != request_sha256:
            raise MediaCreationConflict(MEDIA_CREATION_CONFLICT)
        try:
            return _validate_response(existing, request)
        except (TypeError, ValueError, KeyError, RecursionError) as exc:
            raise MediaCreationConflict(MEDIA_CREATION_CONFLICT) from exc
    response = create()
    receipt = MediaOrganizationCreation(
        operation_key=key,
        kind=kind,
        request_sha256=request_sha256,
        entity_id=response["id"],
        response_json=response,
        response_sha256="",
    )
    receipt.response_sha256 = _response_digest(receipt)
    _validate_response(receipt, request)
    session.add(receipt)
    session.flush()
    return response
