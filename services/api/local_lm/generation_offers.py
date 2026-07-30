from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import ValidationError

from .adapters.base import ChatAdapter, ChatRequest
from .domain import Operation, new_id
from .schemas import (
    GenerationOffer,
    OrderedStepIntent,
    OrderedWorkIntent,
    RoutingPlan,
    RoutingReasonCode,
)

GENERATION_OFFER_VERSION = "generation-offer-v1"
MAX_GENERATION_OFFER_ARGUMENT_CHARS = 200_000
MAX_GENERATION_OFFER_SOURCE_CHARS = 100_000
GENERATION_OFFER_TIMEOUT_SECONDS = 8.0
_GENERATION_PROMPT_REFERENCE = re.compile(
    r"\b(?:image|picture|photo|illustration|video|animation)\s+prompts?\b",
    re.IGNORECASE,
)
GENERATION_OFFER_TOOL = {
    "type": "function",
    "function": {
        "name": "offer_generation",
        "description": (
            "Offer to generate one or more images or videos from prompts you drafted. "
            "Use this only after fulfilling a text request that naturally produced media "
            "prompts. Ask for confirmation; calling this tool never starts generation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "maxLength": 1_000,
                    "description": (
                        "A concise question asking whether to generate the offered prompts."
                    ),
                },
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["image", "video"]},
                            "prompt": {"type": "string", "maxLength": 20_000},
                        },
                        "required": ["mode", "prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["message", "items"],
            "additionalProperties": False,
        },
    },
}

_EXPLICIT_ASSENTS = {
    "yes",
    "yes please",
    "sure",
    "sure please",
    "ok",
    "okay",
    "go ahead",
    "please go ahead",
    "do it",
    "please do it",
    "generate it",
    "generate them",
    "generate these",
    "generate those",
    "generate all",
    "generate all of them",
    "please generate it",
    "please generate them",
    "please generate these",
    "please generate those",
    "create it",
    "create them",
    "make it",
    "make them",
    "start it",
    "start them",
}


def should_extract_generation_offer(text: str) -> bool:
    return (
        0 < len(text) <= MAX_GENERATION_OFFER_SOURCE_CHARS
        and _GENERATION_PROMPT_REFERENCE.search(text) is not None
    )


async def extract_generation_offer(
    adapter: ChatAdapter,
    assistant_text: str,
) -> GenerationOffer | None:
    if not 0 < len(assistant_text) <= MAX_GENERATION_OFFER_SOURCE_CHARS:
        return None
    request = ChatRequest(
        run_id=new_id("offer"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Review the completed assistant response below. If it contains one or "
                    "more concrete image or video generation prompts, call offer_generation "
                    "with those prompts in their original order and a concise confirmation "
                    "question. Do not invent prompts, execute work, include paths or runtime "
                    "identifiers, or call the tool for general discussion about prompting."
                ),
            },
            {"role": "user", "content": assistant_text},
        ],
        tools=[GENERATION_OFFER_TOOL],
        settings={"temperature": 0, "max_tokens": 1_024},
    )
    collector = GenerationOfferCollector()
    try:
        async with asyncio.timeout(GENERATION_OFFER_TIMEOUT_SECONDS):
            async for event in adapter.stream(request):
                if event.type == "error":
                    return None
                if event.type == "tool_delta":
                    collector.add(event.data)
    except Exception:
        return None
    return collector.offer()


def is_explicit_generation_assent(text: str) -> bool:
    normalized = " ".join(text.casefold().split()).rstrip(".!?")
    normalized = " ".join(normalized.replace(",", " ").split())
    if normalized in _EXPLICIT_ASSENTS:
        return True
    return bool(
        re.fullmatch(
            r"(?:yes|sure|ok|okay) (?:please )?(?:go ahead|do it|"
            r"generate|generate (?:it|them|these|those|all|all of them)|"
            r"create (?:it|them)|make (?:it|them)|start (?:it|them))",
            normalized,
        )
    )


def validate_generation_offer(value: object) -> GenerationOffer | None:
    try:
        offer = GenerationOffer.model_validate(value)
    except (TypeError, ValidationError):
        return None
    message = offer.message.strip()
    prompts = [item.prompt.strip() for item in offer.items]
    if not message or any(not prompt for prompt in prompts):
        return None
    return offer.model_copy(
        update={
            "message": message,
            "items": [
                item.model_copy(update={"prompt": prompt})
                for item, prompt in zip(offer.items, prompts, strict=True)
            ],
        }
    )


def generation_offer_metadata(offer: GenerationOffer) -> dict[str, Any]:
    return {
        "version": GENERATION_OFFER_VERSION,
        **offer.model_dump(mode="json"),
    }


def generation_offer_from_metadata(value: object) -> GenerationOffer | None:
    if not isinstance(value, dict) or value.get("version") != GENERATION_OFFER_VERSION:
        return None
    return validate_generation_offer({key: item for key, item in value.items() if key != "version"})


def routing_plan_for_offer(offer: GenerationOffer) -> RoutingPlan:
    if len(offer.items) != 1:
        raise ValueError("A single-offer routing plan requires exactly one item.")
    item = offer.items[0]
    return RoutingPlan(
        operation=(Operation.TEXT_TO_IMAGE if item.mode == "image" else Operation.TEXT_TO_VIDEO),
        standalone_prompt=item.prompt,
        confidence=1,
        reason_code=RoutingReasonCode.GENERATION_OFFER_ACCEPTED,
        reason="explicit assent to the latest model generation offer",
    )


def ordered_intent_for_offer(offer: GenerationOffer) -> OrderedWorkIntent:
    if len(offer.items) < 2:
        raise ValueError("An ordered generation offer requires at least two items.")
    steps: list[OrderedStepIntent] = []
    for index, item in enumerate(offer.items, start=1):
        step_id = f"offer_{index}"
        steps.append(
            OrderedStepIntent(
                id=step_id,
                mode=item.mode,
                prompt=item.prompt,
                depends_on=[f"offer_{index - 1}"] if index > 1 else [],
                inputs=[],
            )
        )
    return OrderedWorkIntent(
        steps=steps,
        confidence=1,
        reason="explicit assent to the latest model generation offer",
        requires_confirmation=True,
    )


class GenerationOfferCollector:
    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}
        self.malformed = False

    @property
    def saw_tool_calls(self) -> bool:
        return bool(self._calls) or self.malformed

    def add(self, data: object) -> None:
        if self.malformed:
            return
        if not isinstance(data, dict):
            self.malformed = True
            return
        raw_calls = data.get("tool_calls")
        if not isinstance(raw_calls, list):
            self.malformed = True
            return
        for raw in raw_calls:
            if not isinstance(raw, dict):
                self.malformed = True
                continue
            raw_index = raw.get("index", 0)
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                self.malformed = True
                continue
            index = raw_index
            if index < 0 or index >= 8:
                self.malformed = True
                continue
            function = raw.get("function")
            if not isinstance(function, dict):
                self.malformed = True
                continue
            call = self._calls.setdefault(index, {"name": "", "arguments": ""})
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str):
                    self.malformed = True
                    continue
                if len(call["name"]) + len(name) > 100:
                    self.malformed = True
                    continue
                call["name"] += name
            arguments = function.get("arguments")
            if arguments is not None:
                if isinstance(arguments, dict):
                    if call["arguments"]:
                        self.malformed = True
                        continue
                    argument_delta = json.dumps(arguments)
                elif isinstance(arguments, str):
                    argument_delta = arguments
                else:
                    self.malformed = True
                    continue
                if (
                    len(call["arguments"]) + len(argument_delta)
                    > MAX_GENERATION_OFFER_ARGUMENT_CHARS
                ):
                    self.malformed = True
                    continue
                call["arguments"] += argument_delta

    def offer(self) -> GenerationOffer | None:
        if self.malformed or len(self._calls) != 1:
            return None
        call = self._calls[min(self._calls)]
        if call["name"] != "offer_generation":
            return None
        try:
            payload = json.loads(call["arguments"])
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return validate_generation_offer(payload)
