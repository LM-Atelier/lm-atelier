from __future__ import annotations

import json
import re
from typing import Any

from .adapters.base import ChatAdapter, ChatRequest
from .domain import Operation, RoutingMode, new_id
from .schemas import RoutingPlan

_IMAGE_CREATE = re.compile(
    r"\b(?:create|draw|generate|make|paint|render|design|illustrate|visuali[sz]e)\b.*"
    r"\b(?:image|picture|photo|portrait|illustration|artwork|logo|icon|wallpaper|poster)\b",
    re.IGNORECASE,
)
_VIDEO_CREATE = re.compile(
    r"\b(?:create|generate|make|render|produce|animate)\b.*"
    r"\b(?:video|animation|clip|movie|footage)\b",
    re.IGNORECASE,
)
_DIRECT_IMAGE = re.compile(r"^\s*(?:draw|paint|illustrate|render)\b", re.IGNORECASE)
_DIRECT_VIDEO = re.compile(r"^\s*(?:animate|make (?:this|that) move)\b", re.IGNORECASE)
_DISCUSSION = re.compile(
    r"^\s*(?:explain|describe|compare|what|why|how|when|where|who|tell me about|write about)\b",
    re.IGNORECASE,
)

ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_route",
        "description": (
            "Choose whether the user wants a normal text answer, a generated image, or a "
            "generated video. Resolve conversational media follow-ups into a standalone prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["text", "image", "video"]},
                "standalone_prompt": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
                "use_prior_image": {"type": "boolean"},
            },
            "required": [
                "mode",
                "standalone_prompt",
                "confidence",
                "reason",
                "use_prior_image",
            ],
            "additionalProperties": False,
        },
    },
}


class RouteConfirmationRequired(ValueError):
    def __init__(self, plan: RoutingPlan) -> None:
        super().__init__("Confirm the suggested media generation route.")
        self.plan = plan


class ModalityRouter:
    async def plan_with_model(
        self,
        *,
        adapter: ChatAdapter,
        text: str,
        mode: RoutingMode,
        input_artifact_ids: list[str],
        has_prior_image: bool = False,
        conversation: list[dict[str, Any]] | None = None,
    ) -> RoutingPlan:
        fallback = self.plan(
            text=text,
            mode=mode,
            input_artifact_ids=input_artifact_ids,
            has_prior_image=has_prior_image,
        )
        if mode != RoutingMode.AUTO:
            return fallback

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Route the latest user message. A request to discuss, explain, write, or "
                    "analyze media is text; only route image/video when the user wants media "
                    "created or modified. Use prior-image context only for a clear visual "
                    "follow-up. Always call choose_route and do not answer normally.\n"
                    f"Prior generated image available: {'yes' if has_prior_image else 'no'}."
                ),
            }
        ]
        messages.extend((conversation or [])[-8:])
        messages.append({"role": "user", "content": text})
        calls: dict[int, dict[str, str]] = {}
        request = ChatRequest(
            run_id=new_id("route"),
            messages=messages,
            tools=[ROUTING_TOOL],
            settings={"temperature": 0, "max_tokens": 256},
        )
        try:
            async for event in adapter.stream(request):
                if event.type == "error":
                    return fallback
                if event.type != "tool_delta":
                    continue
                for raw in event.data.get("tool_calls", []):
                    if not isinstance(raw, dict):
                        continue
                    index = int(raw.get("index", 0))
                    call = calls.setdefault(index, {"name": "", "arguments": ""})
                    function = raw.get("function") or {}
                    if not isinstance(function, dict):
                        continue
                    if function.get("name"):
                        call["name"] += str(function["name"])
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        call["arguments"] += arguments
                    elif isinstance(arguments, dict):
                        call["arguments"] = json.dumps(arguments)
        except Exception:
            return fallback
        if not calls:
            return fallback
        call = calls[min(calls)]
        if call["name"] != "choose_route":
            return fallback
        try:
            arguments = json.loads(call["arguments"])
            return self._from_tool(
                arguments,
                fallback=fallback,
                text=text,
                input_artifact_ids=input_artifact_ids,
                has_prior_image=has_prior_image,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return fallback

    def plan(
        self,
        *,
        text: str,
        mode: RoutingMode,
        input_artifact_ids: list[str],
        has_prior_image: bool = False,
    ) -> RoutingPlan:
        normalized = text.strip()
        if mode == RoutingMode.TEXT:
            return self._text(normalized, "explicit text mode", 1)
        if mode == RoutingMode.IMAGE:
            operation = Operation.IMAGE_TO_IMAGE if input_artifact_ids else Operation.TEXT_TO_IMAGE
            return self._media(operation, normalized, input_artifact_ids, "explicit image mode", 1)
        if mode == RoutingMode.VIDEO:
            operation = Operation.IMAGE_TO_VIDEO if input_artifact_ids else Operation.TEXT_TO_VIDEO
            return self._media(operation, normalized, input_artifact_ids, "explicit video mode", 1)

        if _DISCUSSION.search(normalized) and not re.search(
            r"\b(?:for me|now|instead)\b", normalized, re.IGNORECASE
        ):
            return self._text(normalized, "question or discussion phrasing", 0.94)

        if _VIDEO_CREATE.search(normalized) or _DIRECT_VIDEO.search(normalized):
            operation = (
                Operation.IMAGE_TO_VIDEO
                if input_artifact_ids or has_prior_image
                else Operation.TEXT_TO_VIDEO
            )
            return self._media(
                operation,
                normalized,
                input_artifact_ids,
                "clear video creation request",
                0.96,
            )

        if _IMAGE_CREATE.search(normalized) or _DIRECT_IMAGE.search(normalized):
            operation = Operation.IMAGE_TO_IMAGE if input_artifact_ids else Operation.TEXT_TO_IMAGE
            return self._media(
                operation,
                normalized,
                input_artifact_ids,
                "clear image creation request",
                0.96,
            )

        return self._text(normalized, "no clear media creation intent", 0.9)

    def _from_tool(
        self,
        arguments: Any,
        *,
        fallback: RoutingPlan,
        text: str,
        input_artifact_ids: list[str],
        has_prior_image: bool,
    ) -> RoutingPlan:
        if not isinstance(arguments, dict):
            raise ValueError("route arguments must be an object")
        mode = arguments.get("mode")
        confidence = arguments.get("confidence")
        standalone_prompt = arguments.get("standalone_prompt")
        reason = arguments.get("reason")
        use_prior_image = arguments.get("use_prior_image")
        if (
            mode not in {"text", "image", "video"}
            or not isinstance(confidence, int | float)
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
            or not isinstance(standalone_prompt, str)
            or not standalone_prompt.strip()
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(use_prior_image, bool)
        ):
            raise ValueError("route arguments do not satisfy the schema")
        if mode == "text":
            return self._text(text.strip(), f"model planner: {reason.strip()}", float(confidence))
        has_image_input = bool(input_artifact_ids) or (has_prior_image and use_prior_image)
        if mode == "image":
            operation = Operation.IMAGE_TO_IMAGE if has_image_input else Operation.TEXT_TO_IMAGE
        else:
            operation = Operation.IMAGE_TO_VIDEO if has_image_input else Operation.TEXT_TO_VIDEO
        plan = self._media(
            operation,
            standalone_prompt.strip(),
            input_artifact_ids,
            f"model planner: {reason.strip()}",
            float(confidence),
        )
        # A low-confidence model answer cannot override a strong deterministic text safeguard.
        if (
            fallback.operation == Operation.TEXT
            and fallback.confidence >= 0.94
            and confidence < 0.9
        ):
            return fallback
        return plan

    @staticmethod
    def _text(prompt: str, reason: str, confidence: float) -> RoutingPlan:
        return RoutingPlan(
            operation=Operation.TEXT,
            standalone_prompt=prompt,
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _media(
        operation: Operation,
        prompt: str,
        input_artifact_ids: list[str],
        reason: str,
        confidence: float,
    ) -> RoutingPlan:
        return RoutingPlan(
            operation=operation,
            standalone_prompt=prompt,
            input_artifact_ids=input_artifact_ids,
            confidence=confidence,
            reason=reason,
        )
