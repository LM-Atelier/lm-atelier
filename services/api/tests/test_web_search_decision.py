"""Only a bounded isolated query proposal may proceed to visible consent."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from local_lm.adapters.base import ChatEvent, ChatRequest


class DecisionAdapter:
    def __init__(self, events: list[ChatEvent]) -> None:
        self.events = events
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event


def _event(arguments: object, *, name: str = "search_web", index: object = 0) -> ChatEvent:
    return ChatEvent(
        "tool_delta",
        data={
            "tool_calls": [{"index": index, "function": {"name": name, "arguments": arguments}}],
        },
    )


async def _choose(adapter: Any, *, enabled: Any = True) -> Any:
    decision = importlib.import_module("local_lm.web_search_decision")
    return await decision.propose_search(
        adapter,
        enabled=enabled,
        text="Compare copper and aluminum",
        run_id="constructed-search-run",
        persistence_scope="ephemeral",
        scope_id="constructed-scope",
    )


async def test_isolated_pass_preserves_exact_query_and_conversation_scope() -> None:
    adapter = DecisionAdapter(
        [
            _event('{"query":"  material'),
            _event(' comparison  ","reason":"Find recent comparisons"}', name=""),
        ]
    )
    chosen = await _choose(adapter)
    assert chosen is not None and chosen.query == "  material comparison  "
    assert chosen.reason == "Find recent comparisons"
    (request,) = adapter.requests
    assert [tool["function"]["name"] for tool in request.tools] == ["search_web"]
    assert request.parallel_tool_calls is False
    assert request.persistence_scope == "ephemeral" and request.scope_id == "constructed-scope"
    assert request.messages[-1] == {"role": "user", "content": "Compare copper and aluminum"}


@pytest.mark.parametrize("enabled", [False, None, "true", 1])
async def test_disabled_search_is_not_advertised_or_sent_to_an_adapter(enabled: Any) -> None:
    adapter = DecisionAdapter([])
    assert await _choose(adapter, enabled=enabled) is None
    assert not adapter.requests


@pytest.mark.parametrize(
    "arguments",
    [
        "not json",
        '{"query":"first","query":"second","reason":"A reason"}',
        {"query": "", "reason": "A reason"},
        {"query": "valid", "reason": ""},
        {"query": "valid", "reason": "x" * 201},
        {"query": "valid", "reason": "two\nlines"},
        {"query": "x" * 2001, "reason": "A reason"},
        {"query": "valid", "reason": "A reason", "url": "https://example.test"},
        {"query": "\ud800", "reason": "A reason"},
    ],
)
async def test_malformed_decision_never_becomes_an_outbound_query(arguments: object) -> None:
    assert await _choose(DecisionAdapter([_event(arguments)])) is None


@pytest.mark.parametrize("index", [1, -1, True, "0"])
async def test_a_second_or_malformed_tool_call_is_refused(index: object) -> None:
    arguments = json.dumps({"query": "materials", "reason": "A reason"})
    assert await _choose(DecisionAdapter([_event(arguments, index=index)])) is None


async def test_an_unrelated_tool_cannot_leave_the_isolated_pass() -> None:
    assert (
        await _choose(
            DecisionAdapter(
                [
                    _event({"query": "materials", "reason": "A reason"}, name="read_web_page"),
                ]
            )
        )
        is None
    )


async def test_an_oversized_stream_is_stopped_before_the_next_chunk() -> None:
    consumed_after_limit: list[bool] = []

    class OversizedAdapter:
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            yield _event("x" * 8193)
            consumed_after_limit.append(True)
            yield ChatEvent("complete")

    assert await _choose(OversizedAdapter()) is None
    assert not consumed_after_limit


async def test_cancelling_the_decision_propagates_to_the_calling_execution() -> None:
    importlib.import_module("local_lm.web_search_decision")
    entered = asyncio.Event()

    class WaitingAdapter:
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            entered.set()
            await asyncio.Event().wait()
            yield ChatEvent("complete")

    task = asyncio.create_task(_choose(WaitingAdapter()))
    async with asyncio.timeout(30):
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_answer_prose_is_not_mistaken_for_a_search() -> None:
    assert await _choose(DecisionAdapter([ChatEvent("delta", text="A comparison")])) is None


async def test_adapter_failure_is_a_no_search_outcome() -> None:
    assert (
        await _choose(DecisionAdapter([ChatEvent("error", data={"error": "neutral failure"})]))
        is None
    )


async def test_nullable_stream_fields_are_treated_as_absent_fragments() -> None:
    adapter = DecisionAdapter(
        [
            _event('{"query":"material",'),
            ChatEvent(
                "tool_delta",
                data={
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"name": None, "arguments": '"reason":"Compare facts"}'},
                        }
                    ]
                },
            ),
        ]
    )
    proposed = await _choose(adapter)
    assert proposed is not None and proposed.query == "material"


@pytest.mark.parametrize(
    "query",
    [
        "\u0645\u06cc\u200c\u0631\u0648\u0645",
        "\u0915\u094d\u200d\u0937",
        "\u2764\ufe0f",
        "\U0001f469\u200d\U0001f4bb",
        "\u1820\u180e\u1820",
        "caf\xe9",
        "\u6750\u6599",
        "\U0001f600",
    ],
)
async def test_unicode_spelling_is_not_silently_dropped(query: str) -> None:
    adapter = DecisionAdapter([_event(json.dumps({"query": query, "reason": "Compare materials"}))])
    proposal = await _choose(adapter)
    assert proposal is not None
    assert proposal.query == query
