"""Read bounded album and tag choices without mixing catalog revisions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from datetime import datetime
from typing import Any, NoReturn

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from .models import MediaCollection, MediaOrganizationCatalogRevision, MediaTag

INVALID = "The album or tag page request is invalid. Start again from the first page."
STALE = "Albums or tags changed. Refresh their choices and try again."
_CONTEXT = b"media-organization-catalog-v1"
_MAX_REVISION = 9_007_199_254_740_991


class CatalogInvalid(ValueError):
    pass


class CatalogStale(ValueError):
    pass


def _invalid() -> NoReturn:
    raise CatalogInvalid(INVALID)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not value or len(value) > 1600 or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        _invalid()
    try:
        result = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except ValueError as exc:
        raise CatalogInvalid(INVALID) from exc
    if _encode(result) != value:
        _invalid()
    return result


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _position(
    token: str, key: bytes, filters: dict[str, object], revision: int
) -> tuple[datetime, str]:
    if not token or len(token) > 2048 or token.count(".") != 1:
        _invalid()
    payload, signature = (_decode(part) for part in token.split("."))
    if len(payload) > 1200:
        _invalid()
    expected = hmac.new(key, _CONTEXT + payload, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        _invalid()
    try:
        value = json.loads(payload, object_pairs_hook=_unique)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CatalogInvalid(INVALID) from exc
    if type(value) is not dict or set(value) != {"filters", "revision", "after", "expires"}:
        _invalid()
    bound = value["filters"]
    if (
        type(bound) is not dict
        or set(bound) != set(filters)
        or type(bound.get("limit")) is not int
        or bound != filters
        or type(value["revision"]) is not int
        or not 1 <= value["revision"] <= _MAX_REVISION
        or type(value["expires"]) is not int
        or not time.time() < value["expires"] <= time.time() + 901
    ):
        _invalid()
    if value["revision"] != revision:
        raise CatalogStale(STALE)
    after = value["after"]
    if type(after) is not list or len(after) != 2:
        _invalid()
    stamp, identity = after
    prefix = "collection" if filters["kind"] == "albums" else "mediatag"
    if (
        not isinstance(stamp, str)
        or len(stamp) > 40
        or not isinstance(identity, str)
        or re.fullmatch(prefix + r"_[0-9a-f]{32}", identity) is None
    ):
        _invalid()
    try:
        date = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise CatalogInvalid(INVALID) from exc
    if date.tzinfo is not None or date.isoformat(timespec="microseconds") != stamp:
        _invalid()
    return date, identity


def _read_snapshot(session: Session) -> None:
    connection = session.connection()
    driver = connection.connection.driver_connection
    if connection.dialect.name == "sqlite" and not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN")


def read_catalog_page(
    session: Session,
    *,
    kind: str,
    signing_key: bytes,
    query: str = "",
    limit: int = 50,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return one indexed page; a changed catalog requires a fresh first page."""
    if (
        kind not in {"albums", "tags"}
        or type(limit) is not int
        or not 1 <= limit <= 100
        or len(query) > 200
        or any(ord(character) < 32 for character in query)
        or type(signing_key) is not bytes
        or len(signing_key) != 32
    ):
        _invalid()
    normalized = query.strip().casefold()
    filters: dict[str, object] = {
        "kind": kind,
        "limit": limit,
        "query": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }
    _read_snapshot(session)
    revision = session.scalar(
        select(MediaOrganizationCatalogRevision.revision).where(
            MediaOrganizationCatalogRevision.kind == kind
        )
    )
    if type(revision) is not int or not 1 <= revision <= _MAX_REVISION:
        raise CatalogStale(STALE)
    after = _position(cursor, signing_key, filters, revision) if cursor is not None else None
    records: list[MediaCollection] | list[MediaTag]
    if kind == "albums":
        albums = select(MediaCollection)
        if normalized:
            albums = albums.where(
                func.lower(MediaCollection.name).contains(normalized, autoescape=True)
            )
        if after:
            albums = albums.where(tuple_(MediaCollection.created_at, MediaCollection.id) > after)
        records = list(
            session.scalars(
                albums.order_by(MediaCollection.created_at, MediaCollection.id).limit(limit + 1)
            )
        )
    else:
        tags = select(MediaTag)
        if normalized:
            tags = tags.where(func.lower(MediaTag.label).contains(normalized, autoescape=True))
        if after:
            tags = tags.where(tuple_(MediaTag.created_at, MediaTag.id) > after)
        records = list(
            session.scalars(tags.order_by(MediaTag.created_at, MediaTag.id).limit(limit + 1))
        )
    items = [
        {
            "id": row.id,
            "kind": row.kind,
            "name": row.name,
            "description": row.description,
            "version": row.version,
        }
        if isinstance(row, MediaCollection)
        else {
            "id": row.id,
            "slug": row.slug,
            "label": row.label,
            "color": row.color,
            "version": row.version,
        }
        for row in records[:limit]
    ]
    next_cursor = None
    if len(records) > limit:
        last = records[limit - 1]
        payload = json.dumps(
            {
                "filters": filters,
                "revision": revision,
                "after": [last.created_at.isoformat(timespec="microseconds"), last.id],
                "expires": int(time.time()) + 900,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        signature = hmac.new(signing_key, _CONTEXT + payload, hashlib.sha256).digest()
        next_cursor = _encode(payload) + "." + _encode(signature)
    return {"items": items, "next_cursor": next_cursor, "revision": revision}
