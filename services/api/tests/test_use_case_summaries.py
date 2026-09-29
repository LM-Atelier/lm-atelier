"""Description summaries stay local and remain suggestions until saved."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def test_returns_a_bounded_summary_from_the_local_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries = importlib.import_module("local_lm.use_case_summaries")
    requests: list[dict[str, Any]] = []
    clients: list[httpx.AsyncClient] = []
    original_client = httpx.AsyncClient

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://127.0.0.1:12341/v1/chat/completions"
        requests.append(json.loads(request.content))
        events = [
            {"choices": [{"delta": {"content": "Watercolor "}}]},
            {"choices": [{"delta": {"content": "landscapes."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    def client(**kwargs: Any) -> httpx.AsyncClient:
        created = original_client(**kwargs, transport=httpx.MockTransport(reply))
        clients.append(created)
        return created

    monkeypatch.setattr(httpx, "AsyncClient", client)
    result = await summaries.suggest_use_case_summary(
        "http://127.0.0.1:12341", "Paints watercolor landscapes."
    )
    assert result == "Watercolor landscapes."
    assert len(requests) == 1
    assert len(requests[0]["messages"]) == 2
    assert json.loads(requests[0]["messages"][1]["content"]) == {
        "description": "Paints watercolor landscapes."
    }
    assert "tools" not in requests[0]
    assert requests[0]["max_tokens"] <= 256
    assert clients and all(created.is_closed for created in clients)


@pytest.mark.parametrize(
    "origin", ["https://models.example", "http://192.0.2.1", "http://localhost/path"]
)
async def test_refuses_nonlocal_or_nonorigin_targets_before_opening_a_client(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    summaries = importlib.import_module("local_lm.use_case_summaries")

    def forbidden(**kwargs: Any) -> None:
        pytest.fail("Invalid worker origin reached a client constructor")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    with pytest.raises(summaries.UseCaseSummaryError, match="Use-case suggestion is unavailable"):
        await summaries.suggest_use_case_summary(origin, "Paints watercolor landscapes.")


@pytest.mark.parametrize("description", ["", " \n ", "x" * 8_001, "\ud800"])
async def test_invalid_descriptions_never_reach_transport(
    monkeypatch: pytest.MonkeyPatch, description: str
) -> None:
    summaries = importlib.import_module("local_lm.use_case_summaries")

    def forbidden(**kwargs: Any) -> None:
        pytest.fail("Invalid description reached a client constructor")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    with pytest.raises(summaries.UseCaseSummaryError):
        await summaries.suggest_use_case_summary("http://127.0.0.1:12341", description)


def response_stream(events: list[dict[str, Any]]) -> httpx.Response:
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return httpx.Response(200, text=body + "data: [DONE]\n\n")


@pytest.mark.parametrize(
    "failure", ["empty", "long", "length", "tool", "event_limit", "status", "malformed"]
)
async def test_refuses_incomplete_or_unbounded_model_results_without_echoing_them(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    summaries = importlib.import_module("local_lm.use_case_summaries")
    original_client = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []
    requests: list[httpx.Request] = []
    events: list[dict[str, Any]] = [
        {"choices": [{"delta": {"content": "Constructed result."}}]},
    ]
    if failure == "empty":
        events.clear()
    elif failure == "long":
        events[0]["choices"][0]["delta"]["content"] = "x" * 501
    elif failure == "tool":
        events.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"name": "unexpected", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        )
    elif failure == "event_limit":
        events.extend({"usage": {"completion_tokens": 1}, "choices": []} for _ in range(1_024))
    events.append(
        {"choices": [{"delta": {}, "finish_reason": "length" if failure == "length" else "stop"}]}
    )

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "status":
            return httpx.Response(500, text="Constructed response body")
        if failure == "malformed":
            return httpx.Response(200, text="data: {malformed}\n\n")
        return response_stream(events)

    def client(**kwargs: Any) -> httpx.AsyncClient:
        created = original_client(**kwargs, transport=httpx.MockTransport(reply))
        clients.append(created)
        return created

    monkeypatch.setattr(httpx, "AsyncClient", client)
    with pytest.raises(summaries.UseCaseSummaryError) as caught:
        await summaries.suggest_use_case_summary(
            "http://127.0.0.1:12341", "Constructed description"
        )
    assert str(caught.value) == "Use-case suggestion is unavailable."
    assert len(requests) == 1
    assert clients and all(created.is_closed for created in clients)


@pytest.mark.parametrize("ending", ["timeout", "cancel"])
async def test_abandoned_suggestions_close_the_local_client(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    summaries = importlib.import_module("local_lm.use_case_summaries")
    entered = asyncio.Event()
    blocked = asyncio.Event()
    original_client = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []

    async def reply(request: httpx.Request) -> httpx.Response:
        entered.set()
        await blocked.wait()
        raise AssertionError("Blocked transport unexpectedly resumed")

    def client(**kwargs: Any) -> httpx.AsyncClient:
        created = original_client(**kwargs, transport=httpx.MockTransport(reply))
        clients.append(created)
        return created

    monkeypatch.setattr(httpx, "AsyncClient", client)
    if ending == "timeout":
        monkeypatch.setattr(summaries, "SUMMARY_TIMEOUT_SECONDS", 0.01)
        with pytest.raises(summaries.UseCaseSummaryError):
            await summaries.suggest_use_case_summary(
                "http://127.0.0.1:12341", "Watercolor landscapes"
            )
    else:
        task = asyncio.create_task(
            summaries.suggest_use_case_summary("http://127.0.0.1:12341", "Watercolor landscapes")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert clients and all(created.is_closed for created in clients)
