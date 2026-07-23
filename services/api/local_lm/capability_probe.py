from __future__ import annotations

import json

from .adapters.base import ChatAdapter, ChatRequest
from .domain import new_id
from .schemas import ToolCapabilityProbe

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_route",
        "description": "Choose the local model role required for a user request.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["text", "image", "video"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["mode", "confidence"],
            "additionalProperties": False,
        },
    },
}

# Qwen3 chat templates honor this directive and otherwise may spend the probe's
# small output budget on hidden reasoning before attempting the structured call.
PROBE_USER_PROMPT = "/no_think\nCreate an image of a violet observatory."


async def probe_structured_tools(adapter: ChatAdapter) -> ToolCapabilityProbe:
    capabilities = await adapter.capabilities()
    if not capabilities.tool_calling:
        return ToolCapabilityProbe(
            engine=capabilities.engine,
            version=capabilities.version,
            advertised=False,
            passed=False,
            error="engine does not advertise tool calling",
        )

    calls: dict[int, dict[str, str]] = {}
    error: str | None = None
    request = ChatRequest(
        run_id=new_id("probe"),
        messages=[
            {
                "role": "system",
                "content": "Always call choose_route. Do not answer with ordinary text.",
            },
            {
                "role": "user",
                "content": PROBE_USER_PROMPT,
            },
        ],
        tools=[PROBE_TOOL],
        settings={"temperature": 0, "max_tokens": 96},
    )
    async for event in adapter.stream(request):
        if event.type == "error":
            error = str(event.data.get("error", "engine probe failed"))
        if event.type != "tool_delta":
            continue
        for raw in event.data.get("tool_calls", []):
            if not isinstance(raw, dict):
                continue
            index = int(raw.get("index", 0))
            call = calls.setdefault(index, {"name": "", "arguments": ""})
            function = raw.get("function") or {}
            if isinstance(function, dict):
                if function.get("name"):
                    call["name"] += str(function["name"])
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    call["arguments"] += arguments
                elif isinstance(arguments, dict):
                    call["arguments"] = json.dumps(arguments)

    if not calls:
        return ToolCapabilityProbe(
            engine=capabilities.engine,
            version=capabilities.version,
            advertised=True,
            passed=False,
            error=error or "engine returned no structured tool call",
        )
    call = calls[min(calls)]
    try:
        arguments = json.loads(call["arguments"])
    except (json.JSONDecodeError, TypeError) as exc:
        return ToolCapabilityProbe(
            engine=capabilities.engine,
            version=capabilities.version,
            advertised=True,
            passed=False,
            tool_name=call["name"] or None,
            error=f"tool arguments were not valid JSON: {exc}",
        )
    confidence = arguments.get("confidence") if isinstance(arguments, dict) else None
    valid = (
        call["name"] == "choose_route"
        and isinstance(arguments, dict)
        and arguments.get("mode") in {"text", "image", "video"}
        and isinstance(confidence, int | float)
        and not isinstance(confidence, bool)
        and 0 <= confidence <= 1
    )
    return ToolCapabilityProbe(
        engine=capabilities.engine,
        version=capabilities.version,
        advertised=True,
        passed=valid,
        tool_name=call["name"] or None,
        arguments=arguments if isinstance(arguments, dict) else None,
        error=None if valid else "tool call did not satisfy the declared schema",
    )
