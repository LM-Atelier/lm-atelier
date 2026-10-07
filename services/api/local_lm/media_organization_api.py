"""Expose manual Media Library collections and tags through the API."""

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
    MediaOrganizationConflict,
    MediaOrganizationError,
    create_manual_collection,
    list_manual_collections,
)
from .media_organization import create_media_tag as _create_stored_tag
from .media_organization import list_media_tags as _list_stored_tags
from .media_organization_creations import (
    MediaCreationConflict,
    create_catalog_choice,
)
from .models import MediaCollection, MediaTag

router = APIRouter()

_UNREADABLE_JSON = (UnicodeDecodeError, json.JSONDecodeError, RecursionError)


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
        body = create_catalog_choice(
            session,
            kind="albums",
            payload=payload,
            create=lambda: _collection_body(
                create_manual_collection(
                    session, name=payload.get("name"), description=payload.get("description", "")
                )
            ),
        )
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
    except _UNREADABLE_JSON as exc:
        raise api_error(422, "media-collection-invalid", MEDIA_ORGANIZATION_INVALID) from exc
    try:
        body = await run_in_threadpool(_create_collection, payload)
    except MediaCreationConflict as exc:
        raise api_error(409, "media-creation-conflict", str(exc)) from exc
    except MediaOrganizationError as exc:
        raise api_error(422, "media-collection-invalid", str(exc)) from exc
    return JSONResponse(body, status_code=201)


@router.get("/media-collections")
async def list_media_collections() -> JSONResponse:
    return JSONResponse(await run_in_threadpool(_list_collections))


def _tag_body(tag: MediaTag) -> dict[str, object]:
    return {
        "id": tag.id,
        "slug": tag.slug,
        "label": tag.label,
        "color": tag.color,
        "version": tag.version,
    }


def _create_tag(payload: object) -> dict[str, object]:
    if type(payload) is not dict:
        raise MediaOrganizationError(MEDIA_ORGANIZATION_INVALID)
    with SessionLocal() as session:
        response = create_catalog_choice(
            session,
            kind="tags",
            payload=payload,
            create=lambda: _tag_body(
                _create_stored_tag(session, label=payload.get("label"), color=payload.get("color"))
            ),
        )
        session.commit()
        return response


def _list_tags() -> dict[str, Any]:
    with SessionLocal() as session:
        return {"tags": [_tag_body(tag) for tag in _list_stored_tags(session)]}


@router.post("/media-tags")
async def create_media_tag(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except _UNREADABLE_JSON as exc:
        raise api_error(422, "media-tag-invalid", MEDIA_ORGANIZATION_INVALID) from exc
    try:
        body = await run_in_threadpool(_create_tag, payload)
    except MediaCreationConflict as exc:
        raise api_error(409, "media-creation-conflict", str(exc)) from exc
    except MediaOrganizationConflict as exc:
        raise api_error(409, "media-tag-conflict", str(exc)) from exc
    except MediaOrganizationError as exc:
        raise api_error(422, "media-tag-invalid", str(exc)) from exc
    return JSONResponse(body, status_code=201)


@router.get("/media-tags")
async def list_media_tags() -> JSONResponse:
    return JSONResponse(await run_in_threadpool(_list_tags))
