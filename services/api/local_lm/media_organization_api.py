"""Expose manual Media Library collections through the API."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .db import SessionLocal
from .media_organization import (
    MEDIA_ORGANIZATION_INVALID,
    MediaOrganizationError,
    create_manual_collection,
    list_manual_collections,
)
from .models import MediaCollection

router = APIRouter()


def _collection_body(collection: MediaCollection) -> dict[str, object]:
    return {
        "id": collection.id,
        "kind": collection.kind,
        "name": collection.name,
        "description": collection.description,
        "version": collection.version,
    }


def _create_collection(payload: object) -> dict[str, object]:
    if type(payload) is not dict:
        raise MediaOrganizationError(MEDIA_ORGANIZATION_INVALID)
    with SessionLocal() as session:
        collection = create_manual_collection(
            session,
            name=payload.get("name"),
            description=payload.get("description", ""),
        )
        body = _collection_body(collection)
        session.commit()
        return body


def _list_collections() -> dict[str, Any]:
    with SessionLocal() as session:
        return {
            "collections": [
                _collection_body(collection) for collection in list_manual_collections(session)
            ]
        }


@router.post("/media-collections")
async def create_media_collection(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise api_error(422, "media-collection-invalid", MEDIA_ORGANIZATION_INVALID) from exc
    try:
        body = await run_in_threadpool(_create_collection, payload)
    except MediaOrganizationError as exc:
        raise api_error(422, "media-collection-invalid", str(exc)) from exc
    return JSONResponse(body, status_code=201)


@router.get("/media-collections")
async def list_media_collections() -> JSONResponse:
    return JSONResponse(await run_in_threadpool(_list_collections))
