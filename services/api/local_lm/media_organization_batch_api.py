"""Preview and apply exact Media Library organization selections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .artifact_library import ArtifactLibraryConflict, ArtifactLibraryDataError
from .db import SessionLocal
from .media_organization import MediaOrganizationError
from .media_organization_batches import (
    INVALID,
    MERGE_TRASH,
    STALE,
    TAG_CONFLICT,
    OrganizationInvalid,
    OrganizationMergeTrash,
    OrganizationStale,
    OrganizationTagConflict,
    apply_organization,
    preview_organization,
)
from .media_organization_catalog import (
    INVALID as CATALOG_INVALID,
)
from .media_organization_catalog import (
    STALE as CATALOG_STALE,
)
from .media_organization_catalog import (
    CatalogInvalid,
    CatalogStale,
    read_catalog_page,
)
from .recovery_previews import RecoveryPreviewConflict

if TYPE_CHECKING:
    from .main import Services

router = APIRouter(prefix="/media-organization")


def _catalog(kind: str, key: bytes, query: str, limit: int, cursor: str | None) -> dict[str, Any]:
    with SessionLocal() as session:
        return read_catalog_page(
            session, kind=kind, signing_key=key, query=query, limit=limit, cursor=cursor
        )


@router.get("/catalog/{kind}")
async def list_organization_catalog(
    request: Request,
    kind: str,
    query: str = Query(default="", max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, min_length=1, max_length=2048),
) -> JSONResponse:
    services = cast("Services", request.app.state.services)
    key = services.security.local_state_signing_key(b"media-organization-catalog")
    try:
        body = await run_in_threadpool(_catalog, kind, key, query, limit, cursor)
    except CatalogInvalid as exc:
        raise api_error(422, "media-catalog-invalid", CATALOG_INVALID) from exc
    except CatalogStale as exc:
        raise api_error(409, "media-catalog-stale", CATALOG_STALE) from exc
    return JSONResponse(body)


def _execute(payload: object, impact_id: str | None) -> dict[str, Any]:
    with SessionLocal() as session:
        body = (
            preview_organization(session, payload)
            if impact_id is None
            else apply_organization(session, impact_id, payload)
        )
        session.commit()
        return body


async def _body(request: Request, impact_id: str | None = None) -> dict[str, Any]:
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise api_error(422, "media-organization-invalid", INVALID) from exc
    try:
        return await run_in_threadpool(_execute, payload, impact_id)
    except OrganizationTagConflict as exc:
        raise api_error(409, "media-tag-conflict", TAG_CONFLICT) from exc
    except OrganizationMergeTrash as exc:
        raise api_error(409, "media-tag-merge-trash", MERGE_TRASH) from exc
    except (
        OrganizationStale,
        IntegrityError,
        ArtifactLibraryConflict,
        ArtifactLibraryDataError,
        RecoveryPreviewConflict,
    ) as exc:
        raise api_error(409, "media-organization-stale", STALE) from exc
    except (OrganizationInvalid, MediaOrganizationError) as exc:
        raise api_error(422, "media-organization-invalid", INVALID) from exc


@router.post("/impacts")
async def create_organization_impact(request: Request) -> JSONResponse:
    return JSONResponse(await _body(request), status_code=201)


@router.post("/impacts/{impact_id}/apply")
async def apply_organization_impact(request: Request, impact_id: str) -> JSONResponse:
    return JSONResponse(await _body(request, impact_id))
