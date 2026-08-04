"""Retrieving one page, and refusing every hop that should not be reached."""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.web_retrieval import MAX_REDIRECTS, WebRetrievalError, fetch_source

pytestmark = pytest.mark.asyncio


class _Response:
    def __init__(
        self,
        status_code: int = 200,
        *,
        content: bytes = b"<html><title>A page</title><p>Body text.</p></html>",
        content_type: str = "text/html",
        location: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}
        if location is not None:
            self.headers["location"] = location


def _public(host: str, _port: object) -> list[tuple[object, ...]]:
    """Every hostname resolves somewhere public unless a test says otherwise."""
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def _private(host: str, _port: object) -> list[tuple[object, ...]]:
    return [(2, 1, 6, "", ("127.0.0.1", 443))]


def _responder(*responses: _Response) -> Any:
    queue = list(responses)
    seen: list[str] = []

    async def request(url: str) -> _Response:
        seen.append(url)
        return queue.pop(0)

    request.seen = seen  # type: ignore[attr-defined]
    return request


async def test_reads_a_page_and_reports_where_it_ended_up() -> None:
    source = await fetch_source(
        "https://example.com/a", request=_responder(_Response()), resolve=_public
    )

    assert source.url == "https://example.com/a"
    assert source.final_url == "https://example.com/a"
    assert source.title == "A page"
    assert "Body text." in source.text


async def test_follows_a_redirect_and_reports_the_address_it_landed_on() -> None:
    source = await fetch_source(
        "https://example.com/a",
        request=_responder(
            _Response(302, location="https://example.com/b"),
            _Response(content=b"<html><p>Moved here.</p></html>"),
        ),
        resolve=_public,
    )

    assert source.url == "https://example.com/a"
    assert source.final_url == "https://example.com/b"
    assert "Moved here." in source.text


async def test_a_redirect_into_this_machine_is_refused_at_the_redirect() -> None:
    """A permitted host redirecting to a forbidden one is the obvious attack."""
    hosts = {"example.com": _public, "localhost": _private}

    def resolve(host: str, port: object) -> list[tuple[object, ...]]:
        return hosts.get(host, _public)(host, port)

    with pytest.raises(WebRetrievalError) as refused:
        await fetch_source(
            "https://example.com/a",
            request=_responder(_Response(302, location="https://localhost/admin")),
            resolve=resolve,
        )

    assert refused.value.code == "web-private-address-refused"


async def test_a_redirect_back_to_itself_is_refused_rather_than_looped() -> None:
    with pytest.raises(WebRetrievalError) as refused:
        await fetch_source(
            "https://example.com/a",
            request=_responder(
                _Response(302, location="https://example.com/b"),
                _Response(302, location="https://example.com/a"),
            ),
            resolve=_public,
        )

    assert refused.value.code == "web-redirect-loop"


async def test_too_many_redirects_stops_rather_than_chasing() -> None:
    chain = [
        _Response(302, location=f"https://example.com/{index}")
        for index in range(MAX_REDIRECTS + 2)
    ]
    with pytest.raises(WebRetrievalError) as refused:
        await fetch_source("https://example.com/a", request=_responder(*chain), resolve=_public)

    assert refused.value.code == "web-too-many-redirects"


async def test_a_redirect_naming_nowhere_is_refused() -> None:
    with pytest.raises(WebRetrievalError) as refused:
        await fetch_source(
            "https://example.com/a", request=_responder(_Response(302)), resolve=_public
        )

    assert refused.value.code == "web-redirect-invalid"


async def test_a_non_page_response_is_refused_before_it_is_read_as_text() -> None:
    with pytest.raises(WebRetrievalError) as refused:
        await fetch_source(
            "https://example.com/a.zip",
            request=_responder(_Response(content=b"PK\x03\x04", content_type="application/zip")),
            resolve=_public,
        )

    assert refused.value.code == "web-content-unreadable"


async def test_the_first_address_is_validated_before_any_request_is_made() -> None:
    request = _responder(_Response())

    with pytest.raises(WebRetrievalError):
        await fetch_source("http://example.com/insecure", request=request, resolve=_public)

    assert request.seen == []  # type: ignore[attr-defined]
