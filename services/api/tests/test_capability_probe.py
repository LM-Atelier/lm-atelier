from __future__ import annotations

from local_lm.adapters.mock import MockChatAdapter
from local_lm.capability_probe import probe_structured_tools


async def test_mock_engine_passes_declared_tool_schema_probe() -> None:
    result = await probe_structured_tools(MockChatAdapter())
    assert result.advertised is True
    assert result.passed is True
    assert result.tool_name == "choose_route"
    assert result.arguments == {"mode": "image", "confidence": 1}
