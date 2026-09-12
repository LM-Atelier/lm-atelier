"""Search discovers bounded evidence without reading result pages."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest


def _body(*, nested: bool = False) -> dict[str, object]:
    rows = [{"url": "https://example.test/article", "title": "Article", "snippet": "Evidence"}]
    return {"success": True, "data": {"results": rows} if nested else rows}


@pytest.mark.parametrize("nested", [False, True])
async def test_search_sends_the_exact_query_without_scraping(nested: bool) -> None:
    from local_lm.web_search import CrwSearchProvider, search_crw

    seen: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body(nested=nested))

    result = await search_crw(
        CrwSearchProvider("https://search.example.test/prefix", token="constructed-token"),
        "  neutral query  ",
        transport=httpx.MockTransport(serve),
    )
    assert len(seen) == 1
    assert str(seen[0].url) == "https://search.example.test/prefix/v1/search"
    assert seen[0].method == "POST"
    assert seen[0].headers["authorization"] == "Bearer constructed-token"
    assert json.loads(seen[0].content) == {
        "query": "  neutral query  ",
        "limit": 5,
        "answer": False,
        "summarizeResults": False,
        "queryExpand": False,
    }
    assert result.results[0].url == "https://example.test/article"
    assert result.results[0].title == "Article"
    assert result.results[0].snippet == "Evidence"
    assert not result.truncated
    assert result.discarded_results == 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://search.example.test",
        "http://localhost:3000",
        "https://user:secret@example.test",
        "https://example.test/?token=secret",
        "https://example.test/#fragment",
        "file:///tmp/search",
        "https://example.test\\other",
        "https://example.test/\n",
        "https://example.test:99999",
    ],
)
def test_invalid_provider_addresses_are_refused_without_echo(endpoint: str) -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError

    with pytest.raises(WebSearchError) as caught:
        CrwSearchProvider(endpoint)
    assert str(caught.value) == "search_provider_invalid"


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:3000", "http://[::1]:3000"])
async def test_explicit_loopback_provider_needs_no_hosted_credential(endpoint: str) -> None:
    from local_lm.web_search import CrwSearchProvider, search_crw

    def serve(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=_body())

    assert (
        len(
            (
                await search_crw(
                    CrwSearchProvider(endpoint), "neutral", transport=httpx.MockTransport(serve)
                )
            ).results
        )
        == 1
    )


@pytest.mark.parametrize("query", ["", " \t ", "x" * 2001, "neutral\x00query"])
async def test_invalid_queries_never_dispatch(query: str) -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_body())

    with pytest.raises(WebSearchError) as caught:
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            query,
            transport=httpx.MockTransport(serve),
        )
    assert caught.value.code == "search_query_invalid"
    assert calls == []


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (302, "search_redirect_refused"),
        (401, "search_credentials_refused"),
        (403, "search_credentials_refused"),
        (429, "search_rate_limited"),
        (503, "search_unavailable"),
    ],
)
async def test_http_refusals_do_not_echo_or_follow(status: int, code: str) -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            status,
            headers={"location": "https://other.test/leak"},
            text="constructed secret response",
        )

    with pytest.raises(WebSearchError) as caught:
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(serve),
        )
    assert str(caught.value) == code
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"success": False, "error": "constructed secret"},
        {"success": 1, "data": []},
        {"success": True, "data": {"web": []}},
        {"success": True, "data": None},
        {"success": True, "data": {"results": "not a list"}},
    ],
)
async def test_malformed_envelopes_are_not_empty_success(body: object) -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    with pytest.raises(WebSearchError) as caught:
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
        )
    assert str(caught.value) == "search_response_invalid"


async def test_results_remain_bounded_and_cannot_advertise_local_or_active_urls() -> None:
    from local_lm.web_search import CrwSearchProvider, search_crw

    rows = [
        {"url": "javascript:alert(1)", "title": "Bad"},
        {"url": "https://127.0.0.1/private", "title": "Local"},
        {"url": "https://user:secret@example.test", "title": "Credentials"},
        {"url": "https://[::1]/", "title": "Local IPv6"},
        {"url": "https://bad\ud800.test/", "title": "Malformed"},
        {"url": "https://example.test/valid", "title": "T" * 500, "description": "S" * 4000},
        *[
            {"url": f"https://example.test/{i}", "title": str(i), "snippet": "text"}
            for i in range(12)
        ],
    ]
    result = await search_crw(
        CrwSearchProvider("https://search.example.test"),
        "neutral",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=json.dumps({"success": True, "data": rows}).encode("ascii")
            )
        ),
    )
    assert len(result.results) == 5
    assert result.discarded_results == 5
    assert result.truncated is True
    assert result.results[0].url == "https://example.test/valid"
    assert result.results[0].title == "T" * 200
    assert result.results[0].snippet == "S" * 2000


class _BodyStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, stall: bool = False) -> None:
        self.chunks = chunks
        self.stall = stall
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.stall:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


async def test_streaming_response_limit_closes_the_connection() -> None:
    from local_lm.web_search import (
        MAX_RESPONSE_BYTES,
        CrwSearchProvider,
        WebSearchError,
        search_crw,
    )

    stream = _BodyStream([b"x" * (MAX_RESPONSE_BYTES // 2), b"x" * (MAX_RESPONSE_BYTES // 2 + 1)])
    with pytest.raises(WebSearchError) as caught:
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
        )
    assert caught.value.code == "search_response_too_large"
    assert stream.closed


@pytest.mark.parametrize("headers", [{"content-length": "999999999"}, {"content-encoding": "gzip"}])
async def test_oversized_or_encoded_response_is_refused_before_body_read(
    headers: dict[str, str],
) -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    stream = _BodyStream([], stall=True)
    with pytest.raises(WebSearchError):
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, headers=headers, stream=stream)
            ),
        )
    assert stream.closed


async def test_whole_request_deadline_cancels_a_stalled_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.web_search as search

    monkeypatch.setattr(search, "REQUEST_TIMEOUT_SECONDS", 0.02)
    stream = _BodyStream([], stall=True)
    with pytest.raises(search.WebSearchError) as caught:
        await search.search_crw(
            search.CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
        )
    assert caught.value.code == "search_timeout"
    assert stream.closed


async def test_caller_cancellation_is_preserved() -> None:
    from local_lm.web_search import CrwSearchProvider, search_crw

    started = asyncio.Event()

    async def serve(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(
        search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(serve),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_transport_errors_do_not_disclose_request_or_response_details() -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    def serve(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("constructed secret", request=request)

    with pytest.raises(WebSearchError) as caught:
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral",
            transport=httpx.MockTransport(serve),
        )
    assert str(caught.value) == "search_unavailable"
    assert caught.value.__cause__ is None


def test_malformed_unicode_provider_is_refused_at_configuration() -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError

    with pytest.raises(WebSearchError) as caught:
        CrwSearchProvider("https://bad\ud800.test")
    assert str(caught.value) == "search_provider_invalid"


async def test_malformed_unicode_query_is_refused_before_dispatch() -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_body())

    with pytest.raises(WebSearchError) as caught:
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "neutral\ud800",
            transport=httpx.MockTransport(serve),
        )
    assert str(caught.value) == "search_query_invalid"
    assert calls == []


@pytest.mark.parametrize("character", ["\x85", "\u2028", "\u2029"])
async def test_query_controls_never_dispatch(character: str) -> None:
    from local_lm.web_search import CrwSearchProvider, WebSearchError, search_crw

    calls = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_body())

    with pytest.raises(WebSearchError, match="search_query_invalid"):
        await search_crw(
            CrwSearchProvider("https://search.example.test"),
            "copper" + character + "steel",
            transport=httpx.MockTransport(serve),
        )
    assert calls == []


@pytest.mark.parametrize(
    "character",
    [
        "\u00ad",
        "\u034f",
        "\u061c",
        "\u115f",
        "\u17b4",
        "\u180b",
        "\u180e",
        "\u200b",
        "\u200c",
        "\u200d",
        "\u202e",
        "\u2060",
        "\u3164",
        "\ufe0f",
        "\ufeff",
        "\uffa0",
        "\ufff0",
        "\ufff9",
        "\ufffa",
        "\ufffb",
        "\u0600",
        "\u06dd",
        "\u070f",
        "\u0890",
        "\u08e2",
        "\U000110bd",
        "\U00013430",
        "\ue000",
        "\U000f0000",
        "\ufffe",
        "\ufdd0",
        "\u2800",
        "\U0001bca0",
        "\U0001d173",
        "\U000e007f",
        "\U000e0100",
    ],
)
async def test_disclosed_query_formatting_is_sent_unchanged(character: str) -> None:
    from local_lm.web_search import CrwSearchProvider, search_crw

    query = "copper" + character + "steel"
    calls = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["query"])
        return httpx.Response(200, json=_body())

    await search_crw(
        CrwSearchProvider("https://search.example.test"),
        query,
        transport=httpx.MockTransport(serve),
    )
    assert calls == [query]
