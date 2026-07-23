from __future__ import annotations

from collections.abc import AsyncIterator

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.capability_probe import probe_structured_tools


class ThinkingMockChatAdapter(MockChatAdapter):
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        user_prompt = str(request.messages[-1]["content"])
        if not user_prompt.startswith("/no_think\n"):
            yield ChatEvent(type="delta", text="I should reason about the route.")
            yield ChatEvent(type="complete", data={"finish_reason": "length"})
            return
        async for event in super().stream(request):
            yield event


async def test_mock_engine_passes_declared_tool_schema_probe() -> None:
    result = await probe_structured_tools(MockChatAdapter())
    assert result.advertised is True
    assert result.passed is True
    assert result.tool_name == "choose_route"
    assert result.arguments == {"mode": "image", "confidence": 1}


async def test_probe_disables_thinking_before_requesting_tool_call() -> None:
    result = await probe_structured_tools(ThinkingMockChatAdapter())
    assert result.passed is True
    assert result.tool_name == "choose_route"
