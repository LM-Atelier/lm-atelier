"""Bounded CRW result discovery for a separately authorized search operation.

This transport does not grant consent or fetch result pages. Its caller owns
the installation/chat permission checks and the exact-query confirmation.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

import httpx

from .network import shared_tls_context

MAX_QUERY_CHARACTERS = 2_000
MAX_RESPONSE_BYTES = 256 * 1024
MAX_RESULTS = 5
REQUEST_TIMEOUT_SECONDS = 20

SearchErrorCode = Literal[
    "search_provider_invalid",
    "search_query_invalid",
    "search_credentials_refused",
    "search_redirect_refused",
    "search_rate_limited",
    "search_unavailable",
    "search_timeout",
    "search_response_invalid",
    "search_response_too_large",
]


class WebSearchError(ValueError):
    def __init__(self, code: SearchErrorCode) -> None:
        super().__init__(code)
        self.code = code


def _url_parts(value: str) -> tuple[str, str]:
    if not value or len(value) > 2_000 or "\\" in value:
        raise ValueError
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise ValueError
    parsed = urlsplit(value)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError
    _ = parsed.port
    return parsed.scheme, parsed.hostname


@dataclass(frozen=True)
class CrwSearchProvider:
    endpoint: str = field(repr=False)
    token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            scheme, host = _url_parts(self.endpoint)
            httpx.URL(self.endpoint)
            parsed = urlsplit(self.endpoint)
            if parsed.query or parsed.fragment or "?" in self.endpoint or "#" in self.endpoint:
                raise ValueError
            if scheme != "https" and (
                scheme != "http" or not ipaddress.ip_address(host).is_loopback
            ):
                raise ValueError
            if self.token is not None and (
                not self.token
                or len(self.token) > 4_096
                or any(ord(character) <= 32 or ord(character) >= 127 for character in self.token)
            ):
                raise ValueError
        except (ValueError, httpx.InvalidURL):
            raise WebSearchError("search_provider_invalid") from None


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str


@dataclass(frozen=True)
class SearchResults:
    results: tuple[SearchResult, ...]
    truncated: bool
    discarded_results: int


def _check_response(response: httpx.Response) -> None:
    errors: dict[int, SearchErrorCode] = {
        401: "search_credentials_refused",
        403: "search_credentials_refused",
        429: "search_rate_limited",
    }
    if 300 <= response.status_code < 400:
        raise WebSearchError("search_redirect_refused")
    if response.status_code != 200:
        raise WebSearchError(errors.get(response.status_code, "search_unavailable"))
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise WebSearchError("search_response_invalid")
    declared = response.headers.get("content-length")
    if declared is not None:
        if not declared.isascii() or not declared.isdecimal():
            raise WebSearchError("search_response_invalid")
        if len(declared) > 10 or int(declared) > MAX_RESPONSE_BYTES:
            raise WebSearchError("search_response_too_large")


async def _read_body(response: httpx.Response) -> bytes:
    _check_response(response)
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
            raise WebSearchError("search_response_too_large")
        body.extend(chunk)
    return bytes(body)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _result_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        scheme, host = _url_parts(value)
        httpx.URL(value)
        if scheme != "https":
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host.lower() == "localhost" or host.lower().endswith(".localhost"):
                return None
        else:
            if not address.is_global:
                return None
    except (ValueError, httpx.InvalidURL):
        return None
    # No DNS lookup here: choosing to read this URL is a separate operation
    # whose existing page-retrieval guard validates resolved addresses.
    return value


def _text(value: object, maximum: int) -> tuple[str, bool]:
    if not isinstance(value, str):
        return "", False
    normalized = " ".join(
        "".join(
            character for character in value if character.isprintable() or character.isspace()
        ).split()
    )
    return normalized[:maximum], len(normalized) > maximum


def _decode_results(body: bytes) -> SearchResults:
    try:
        envelope = json.loads(body, object_pairs_hook=_json_object)
    except (ValueError, RecursionError):
        raise WebSearchError("search_response_invalid") from None
    if not isinstance(envelope, dict) or envelope.get("success") is not True:
        raise WebSearchError("search_response_invalid")
    rows = envelope.get("data")
    if isinstance(rows, dict):
        rows = rows.get("results")
    return _normalize_results(rows)


def _normalize_results(rows: object) -> SearchResults:
    if not isinstance(rows, list):
        raise WebSearchError("search_response_invalid")
    results: list[SearchResult] = []
    seen: set[str] = set()
    truncated = False
    discarded = 0
    for row in rows:
        if not isinstance(row, dict):
            discarded += 1
            continue
        url = _result_url(row.get("url"))
        title, title_cut = _text(row.get("title"), 200)
        if url is None or not title or url in seen:
            discarded += 1
            continue
        seen.add(url)
        snippet, snippet_cut = _text(row.get("snippet", row.get("description")), 2_000)
        truncated = truncated or title_cut or snippet_cut
        if len(results) >= MAX_RESULTS:
            truncated = True
            continue
        results.append(SearchResult(url=url, title=title, snippet=snippet))
    return SearchResults(tuple(results), truncated, discarded)


def validate_search_query(query: str) -> None:
    if (
        len(query) > MAX_QUERY_CHARACTERS
        or not query.strip()
        or any(unicodedata.category(character) in {"Cc", "Cs", "Zl", "Zp"} for character in query)
    ):
        raise WebSearchError("search_query_invalid")
    try:
        query.encode("utf-8")
    except UnicodeError:
        raise WebSearchError("search_query_invalid") from None


async def search_crw(
    provider: CrwSearchProvider,
    query: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchResults:
    """Discover results after the caller has authorized this exact operation."""
    validate_search_query(query)
    headers = {"accept": "application/json", "accept-encoding": "identity"}
    if provider.token is not None:
        headers["authorization"] = "Bearer " + provider.token
    try:
        async with (
            asyncio.timeout(REQUEST_TIMEOUT_SECONDS),
            httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                verify=shared_tls_context(trust_environment=False),
                timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
                transport=transport,
            ) as client,
            client.stream(
                "POST",
                provider.endpoint.rstrip("/") + "/v1/search",
                headers=headers,
                json={
                    "query": query,
                    "limit": MAX_RESULTS,
                    "answer": False,
                    "summarizeResults": False,
                    "queryExpand": False,
                },
            ) as response,
        ):
            body = await _read_body(response)
    except (TimeoutError, httpx.TimeoutException):
        raise WebSearchError("search_timeout") from None
    except httpx.HTTPError:
        raise WebSearchError("search_unavailable") from None
    return _decode_results(body)


def search_results_from_record(record: object) -> SearchResults:
    """Restore bounded evidence without authorizing another provider request."""
    if not isinstance(record, dict):
        raise WebSearchError("search_response_invalid")
    normalized = _normalize_results(record.get("results"))
    discarded = record.get("discarded_results", 0)
    return SearchResults(
        normalized.results,
        normalized.truncated or record.get("truncated") is True,
        normalized.discarded_results
        + (discarded if type(discarded) is int and discarded >= 0 else 0),
    )
