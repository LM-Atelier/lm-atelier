from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from .adapters.base import ChatAdapter, ChatRequest
from .domain import RoutingMode, new_id
from .routing import DISCUSSION_OPENING
from .schemas import OrderedStepInput, OrderedStepIntent, OrderedWorkIntent

MAX_ORDERED_PLAN_STEPS = 8
MAX_ORDERED_PLAN_PROMPT_CHARS = 50_000

_SEQUENCE_SEPARATOR = re.compile(
    r"\s*(?:->|\u2192|;|\bthen\b|\bnext\b|\bafter\s+that\b|\bfinally\b)\s*",
    re.IGNORECASE,
)
_IMAGE_ACTION = re.compile(
    r"\b(?:create|generate|make|draw|paint|render|design|illustrate|visuali[sz]e)\b"
    r".*\b(?:image|picture|photo|illustration|artwork|poster|logo)\b",
    re.IGNORECASE,
)
_VIDEO_ACTION = re.compile(
    r"\b(?:create|generate|make|render|produce|animate|turn)\b"
    r".*\b(?:video|animation|clip|movie|footage|move)\b",
    re.IGNORECASE,
)
_TEXT_ACTION = re.compile(
    r"\b(?:write|draft|compose|tell|describe|plan|outline|summari[sz]e|"
    r"critique|review|explain|analy[sz]e)\b",
    re.IGNORECASE,
)
_IMAGE_PROMPT_TEXT = re.compile(
    r"\b(?:write|draft|compose|create|make)\b.*\bimage\s+prompt\b",
    re.IGNORECASE,
)
_MODEL_SEQUENCE_MEDIA = re.compile(
    r"\b(?:image|picture|photo|illustration|artwork|video|animation|clip|"
    r"sketch|draw|paint|render|visual|animate|motion)\b",
    re.IGNORECASE,
)

_ALLOWED_DATA_TRANSITIONS = {
    ("text", "text"),
    ("text", "image"),
    ("text", "video"),
    ("image", "text"),
    ("image", "image"),
    ("image", "video"),
    ("video", "text"),
}
StepMode = Literal["text", "image", "video"]
ORDERED_PLANNER_VERSION = "ordered-work-v1"
ORDERED_PLANNER_TIMEOUT_SECONDS = 8.0
ORDERED_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "compile_ordered_work",
        "description": (
            "Identify an explicitly requested ordered chain of two or more text, image, "
            "or video actions. Return typed intent only, never runtime identifiers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "is_ordered": {"type": "boolean"},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "requires_confirmation": {"type": "boolean"},
                "steps": {
                    "type": "array",
                    "maxItems": MAX_ORDERED_PLAN_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                            "mode": {
                                "type": "string",
                                "enum": ["text", "image", "video"],
                            },
                            "prompt": {"type": "string", "maxLength": 20_000},
                            "depends_on": {
                                "type": "array",
                                "maxItems": MAX_ORDERED_PLAN_STEPS,
                                "items": {"type": "string"},
                            },
                            "inputs": {
                                "type": "array",
                                "maxItems": MAX_ORDERED_PLAN_STEPS,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_step_id": {"type": "string"},
                                        "kind": {
                                            "type": "string",
                                            "enum": ["text_context", "artifact"],
                                        },
                                    },
                                    "required": ["source_step_id", "kind"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "id",
                            "mode",
                            "prompt",
                            "depends_on",
                            "inputs",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "is_ordered",
                "reason",
                "confidence",
                "requires_confirmation",
                "steps",
            ],
            "additionalProperties": False,
        },
    },
}


class OrderedPlanConfirmationRequired(ValueError):
    def __init__(
        self,
        intent: OrderedWorkIntent,
        *,
        estimate: dict[str, int | float] | None = None,
    ) -> None:
        super().__init__("Confirm the ordered multi-model plan.")
        self.intent = intent
        self.estimate = estimate or {}


class OrderedPlanCompiler:
    """Compile bounded model-agnostic intent, never runtime identifiers."""

    @classmethod
    def deterministic(
        cls,
        text: str,
        mode: RoutingMode,
        *,
        has_media_input: bool = False,
    ) -> OrderedWorkIntent | None:
        if mode != RoutingMode.AUTO:
            return None
        clauses = [
            clause.strip(" ,.\t\r\n")
            for clause in _SEQUENCE_SEPARATOR.split(text)
            if clause.strip(" ,.\t\r\n")
        ]
        if len(clauses) < 2:
            return None
        modes = [cls._clause_mode(clause) for clause in clauses]
        if any(candidate is None for candidate in modes):
            return None
        normalized_modes = [candidate for candidate in modes if candidate is not None]
        if not has_media_input and not any(
            candidate in {"image", "video"} for candidate in normalized_modes
        ):
            return None
        # Only reject an over-long plan once this is known to be one. Counting
        # first rejected any message that merely happened to contain enough
        # separators, such as a semicolon-separated list.
        if len(clauses) > MAX_ORDERED_PLAN_STEPS:
            raise ValueError(f"Ordered plans can contain at most {MAX_ORDERED_PLAN_STEPS} steps.")

        steps: list[OrderedStepIntent] = []
        for index, (clause, step_mode) in enumerate(
            zip(clauses, normalized_modes, strict=True),
            start=1,
        ):
            step_id = f"step_{index}"
            depends_on: list[str] = []
            inputs: list[OrderedStepInput] = []
            if index > 1:
                source_id = f"step_{index - 1}"
                source_mode = normalized_modes[index - 2]
                if (source_mode, step_mode) not in _ALLOWED_DATA_TRANSITIONS:
                    return None
                depends_on.append(source_id)
                inputs.append(
                    OrderedStepInput(
                        source_step_id=source_id,
                        kind="text_context" if source_mode == "text" else "artifact",
                    )
                )
            steps.append(
                OrderedStepIntent(
                    id=step_id,
                    mode=step_mode,
                    prompt=clause,
                    depends_on=depends_on,
                    inputs=inputs,
                )
            )
        return cls.validate(
            OrderedWorkIntent(
                steps=steps,
                confidence=0.96,
                reason="explicit ordered multi-model request",
                requires_confirmation=False,
            )
        )

    @classmethod
    async def plan_with_model(
        cls,
        *,
        adapter: ChatAdapter,
        text: str,
        mode: RoutingMode,
        conversation: list[dict[str, Any]] | None = None,
        has_media_input: bool = False,
    ) -> OrderedWorkIntent | None:
        fallback = cls.deterministic(
            text,
            mode,
            has_media_input=has_media_input,
        )
        if fallback or mode != RoutingMode.AUTO:
            return fallback
        if len(_SEQUENCE_SEPARATOR.findall(text)) < 1 or not _MODEL_SEQUENCE_MEDIA.search(text):
            return None
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    f"Planner schema {ORDERED_PLANNER_VERSION}. Compile only when the user "
                    "explicitly requests at least two actions in a specific order and at least "
                    "one action creates or consumes image/video media. Explicit discussion of "
                    "pipelines is not a work request. Use only text/image/video intent, prompts, "
                    "earlier step IDs, and typed inputs. Never emit file paths, model/profile/"
                    "workflow IDs, settings, node names, executable actions, downloads, or more "
                    f"than {MAX_ORDERED_PLAN_STEPS} steps. A source text uses text_context; a "
                    "source image/video uses artifact. Always call compile_ordered_work."
                    f"\nExplicit media input available: {'yes' if has_media_input else 'no'}."
                ),
            }
        ]
        messages.extend((conversation or [])[-8:])
        messages.append({"role": "user", "content": text})
        request = ChatRequest(
            run_id=new_id("ordered_plan"),
            messages=messages,
            tools=[ORDERED_PLAN_TOOL],
            settings={"temperature": 0, "max_tokens": 768},
        )
        calls: dict[int, dict[str, str]] = {}
        try:
            async with asyncio.timeout(ORDERED_PLANNER_TIMEOUT_SECONDS):
                async for event in adapter.stream(request):
                    if event.type == "error":
                        return None
                    if event.type != "tool_delta":
                        continue
                    for raw in event.data.get("tool_calls", []):
                        if not isinstance(raw, dict):
                            continue
                        index = int(raw.get("index", 0))
                        call = calls.setdefault(index, {"name": "", "arguments": ""})
                        function = raw.get("function")
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
            return None
        if not calls:
            return None
        call = calls[min(calls)]
        if call["name"] != "compile_ordered_work":
            return None
        try:
            payload = json.loads(call["arguments"])
            if not isinstance(payload, dict) or payload.get("is_ordered") is not True:
                return None
            intent = cls.validate(
                OrderedWorkIntent.model_validate(
                    {
                        "steps": payload.get("steps"),
                        "confidence": payload.get("confidence"),
                        "reason": payload.get("reason"),
                        "requires_confirmation": payload.get("requires_confirmation"),
                    }
                )
            )
            if not has_media_input and not any(
                step.mode in {"image", "video"} for step in intent.steps
            ):
                return None
            return intent
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @classmethod
    def validate(cls, intent: OrderedWorkIntent) -> OrderedWorkIntent:
        if not 2 <= len(intent.steps) <= MAX_ORDERED_PLAN_STEPS:
            raise ValueError(f"Ordered plans must contain 2 to {MAX_ORDERED_PLAN_STEPS} steps.")
        if sum(len(step.prompt) for step in intent.steps) > MAX_ORDERED_PLAN_PROMPT_CHARS:
            raise ValueError("The ordered plan prompt budget is too large.")
        by_id = {step.id: step for step in intent.steps}
        if len(by_id) != len(intent.steps):
            raise ValueError("Ordered plan step identifiers must be unique.")
        index_by_id = {step.id: index for index, step in enumerate(intent.steps)}
        for index, step in enumerate(intent.steps):
            if len(set(step.depends_on)) != len(step.depends_on):
                raise ValueError(f"Step {step.id} repeats a dependency.")
            for dependency_id in step.depends_on:
                dependency_index = index_by_id.get(dependency_id)
                if dependency_index is None:
                    raise ValueError(f"Step {step.id} has an unknown dependency.")
                if dependency_index >= index:
                    raise ValueError(f"Step {step.id} must depend only on an earlier step.")
            for binding in step.inputs:
                source = by_id.get(binding.source_step_id)
                if not source or binding.source_step_id not in step.depends_on:
                    raise ValueError(f"Step {step.id} has an input without a declared dependency.")
                expected_kind = "text_context" if source.mode == "text" else "artifact"
                if binding.kind != expected_kind:
                    raise ValueError(f"Step {step.id} has an incompatible input binding.")
                if (source.mode, step.mode) not in _ALLOWED_DATA_TRANSITIONS:
                    raise ValueError(
                        f"Step {step.id} cannot consume {source.mode} output as {step.mode} input."
                    )
        return intent

    @staticmethod
    def _clause_mode(clause: str) -> StepMode | None:
        # A question about how to do something names the medium it asks about,
        # so it matches the media patterns below. The router guards against that
        # already; this compiler runs before the router, so it has to as well.
        if DISCUSSION_OPENING.search(clause):
            return "text"
        if _IMAGE_PROMPT_TEXT.search(clause):
            return "text"
        if _VIDEO_ACTION.search(clause):
            return "video"
        if _IMAGE_ACTION.search(clause):
            return "image"
        if _TEXT_ACTION.search(clause):
            return "text"
        return None
