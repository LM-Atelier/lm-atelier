from __future__ import annotations

import asyncio
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
_DIRECT_VIDEO = re.compile(r"^\s*(?:animate|make (?:it|this|that) move)\b", re.IGNORECASE)
_PRIOR_IMAGE_EDIT = re.compile(
    r"^\s*(?:please\s+|now\s+)*(?:"
    r"(?:make|change|turn|edit|modify|adjust)\s+(?:it|this|that|the\s+"
    r"(?:image|picture|photo|illustration|artwork|logo|icon))\b|"
    r"(?:make|change|edit|modify|adjust|recolor)\s+"
    r"(?:(?:her|his|their|my|our|your)\s+"
    r"(?:top|shirt|blouse|sweater|sweatshirt|jacket|coat|dress|outfit|"
    r"pants|trousers|skirt|shoes|hair)|"
    r"the\s+(?:background|foreground|sky|lighting|color|colour|"
    r"brightness|contrast|exposure|saturation|person|woman|man|"
    r"girl|boy))\b|"
    r"(?:increase|decrease|raise|lower|reduce)\s+(?:the\s+)?"
    r"(?:brightness|contrast|exposure|saturation|warmth|sharpness)\b|"
    r"(?:give|dress|put)\s+(?:her|him|them|the\s+(?:person|subject|woman|"
    r"man|girl|boy))\b.{0,80}\b(?:top|shirt|blouse|sweater|sweatshirt|"
    r"jacket|coat|dress|outfit|pants|trousers|skirt|shoes|hair)\b|"
    r"(?:add|remove|replace|recolor|crop|resize|brighten|darken|blur|sharpen|"
    r"rotate|flip)\b"
    r")",
    re.IGNORECASE,
)
_PRIOR_IMAGE_SOURCE = re.compile(
    r"\b(?:previous|prior|earlier|last|above)\s+"
    r"(?:image|picture|photo|illustration|artwork|logo|icon)\b|"
    r"^\s*(?:please\s+|now\s+)*(?:use|reuse|remix|restyle|transform|redo|"
    r"recreate|continue)\b.*\b(?:it|this|that|the\s+"
    r"(?:image|picture|photo|illustration|artwork|logo|icon))\b",
    re.IGNORECASE,
)
_DISCUSSION = re.compile(
    r"^\s*(?:explain|describe|compare|what|why|how|when|where|who|tell me about|write about)\b",
    re.IGNORECASE,
)
_TEXT_TASK = re.compile(
    r"^\s*(?:answer|reply|respond|say|write|draft|compose|summari[sz]e|translate|"
    r"rewrite|proofread|review|analy[sz]e|list|count|calculate|compute|solve|"
    r"brainstorm|code|program|create|generate)\b",
    re.IGNORECASE,
)
_TEXT_EDIT = re.compile(
    r"^\s*(?:please\s+|now\s+)*(?:"
    r"(?:add|remove|replace|change|edit|modify)\s+(?:the\s+|an?\s+)?"
    r"(?:wording|grammar|punctuation|ambiguity|typo|sentence|paragraph|"
    r"text response|answer)|"
    r"(?:add|remove|replace|change)\b.{0,80}\b(?:in|from|to)\s+"
    r"(?:this|the)\s+(?:sentence|paragraph|text|answer|response|document|code)|"
    r"change\s+the\s+subject\s+of\s+(?:this|the)\s+"
    r"(?:sentence|paragraph|text|answer|response|document)"
    r")\b",
    re.IGNORECASE,
)
_UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|possibly|might want to)\b", re.IGNORECASE)
_TEXT_CONTEXT_REFERENCE = re.compile(
    r"\b(?:(?:previous|prior|earlier|above|last|preceding)\s+"
    r"(?:story|response|answer|message|text|description|scene|passage|poem|idea|"
    r"concept|discussion|conversation)|"
    r"(?:that|this|the)\s+(?:story|response|answer|message|text|description|scene|"
    r"passage|poem|idea|concept|discussion|conversation)|"
    r"what\s+(?:you|i|we)\s+(?:wrote|described|said)|"
    r"(?:our|the)\s+(?:conversation|discussion)|"
    r"(?:based on|inspired by|drawn from)\s+(?:that|this|it|what|the\s+"
    r"(?:previous|prior|earlier|above|last)))\b",
    re.IGNORECASE,
)
_CONTEXT_IMAGE_CREATE = re.compile(
    r"\b(?:image|picture|illustration|artwork)\s+(?:of|from)\s+(?:it|that|this)\b|"
    r"^\s*(?:illustrate|visuali[sz]e|depict)\s+(?:it|that|this)\b|"
    r"\b(?:turn|make|convert)\s+(?:it|that|this)\s+(?:into|as)\s+(?:an?\s+)?"
    r"(?:image|picture|illustration|artwork)\b",
    re.IGNORECASE,
)
_CONTEXT_VIDEO_CREATE = re.compile(
    r"\b(?:video|animation)\s+(?:of|from)\s+(?:it|that|this)\b|"
    r"\b(?:turn|make|convert)\s+(?:it|that|this)\s+(?:into|as)\s+(?:an?\s+)?"
    r"(?:video|animation)\b",
    re.IGNORECASE,
)
_MEDIA_CONTEXT_SUMMARY = re.compile(r"^\s*Generated (?:image|video|\d+ images?|\d+ videos?)\b")
_ANOTHER_GENERATION = re.compile(
    r"^\s*(?:(?:please|now|ok(?:ay)?),?\s+)*"
    r"(?:(?:make|generate|create|do|give)\s+(?:me\s+|us\s+)?(?:another|one\s+more|a\s+new)"
    r"(?:\s+(?:one|image|picture|photo|video|clip|variation|version))?"
    r"|another(?:\s+one)?|one\s+more)"
    r"[\s.!]*$",
    re.IGNORECASE,
)
_GENERATED_PROMPT = re.compile(
    r"^\s*Generated (?P<media>image|video|\d+ images?|\d+ videos?)\b"
    r'.*?prompt \(visual contents not inspected\): "(?P<prompt>.*)"\.\s*$',
    re.DOTALL,
)
_ORDINAL_SELECTION = re.compile(
    r"^\s*(?:(?:please|now|ok(?:ay)?),?\s+)*"
    r"(?:make|generate|create|draw|render|do|give)\s+(?:me\s+|us\s+)?"
    r"(?:the\s+)?"
    r"(?:(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"|(?:number\s+|#\s*)(?P<number>\d{1,2}))"
    r"(?:\s+(?:one|suggestion|option|idea|prompt))?"
    r"(?:\s+as\s+(?:an?\s+)?(?P<medium>image|picture|video|animation))?"
    r"[\s.!]*$",
    re.IGNORECASE,
)
_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_LIST_ITEM = re.compile(r"^\s*(?:\d{1,2}[.)]\s+|[-*•]\s+)(?P<item>\S.*?)\s*$")
_OUTPUT_COUNT = re.compile(
    r"\b(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen)\s+"
    r"(?:different\s+|distinct\s+)?"
    r"(?:images?|pictures?|videos?|clips?|variations?|versions?|options?|renders?)\b",
    re.IGNORECASE,
)
_PER_OUTPUT_DISTRIBUTOR = re.compile(
    r"\b(?:with\s+)?each(?:\s+(?:one|image|picture|photo|video|clip))?\b\s*",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
}

ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_route",
        "description": (
            "Choose whether the user wants a normal text answer, a generated image, or a "
            "generated video. Resolve conversational media follow-ups into a standalone "
            "prompt grounded in the referenced chat text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["text", "image", "video"]},
                "standalone_prompt": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
                "use_prior_image": {"type": "boolean"},
                "output_count": {"type": "integer", "minimum": 1, "maximum": 16},
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
PLANNER_TIMEOUT_SECONDS = 8.0


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
            conversation=conversation,
        )
        referenced_text = self._referenced_text_context(text, conversation or [])
        if mode != RoutingMode.AUTO:
            return fallback
        if fallback.confidence >= 0.94 and not referenced_text:
            return fallback

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Route the latest user message. A request to discuss, explain, write, or "
                    "analyze media is text; only route image/video when the user wants media "
                    "created or modified. Use prior-image context only for a clear visual "
                    "follow-up. An assistant message beginning with 'Generated image' includes "
                    "the source prompt as a semantic description of that image. Use it to resolve "
                    "references such as 'it' and rewrite edits as complete standalone prompts. "
                    "When media is requested from a previous story, response, description, or "
                    "other chat text, extract its concrete visual subjects, setting, mood, and "
                    "style into the standalone prompt; never leave an unresolved reference such "
                    "as 'the previous story'. Keep the standalone prompt concise. "
                    "When the user asks for another of the last generation, reuse that "
                    "generation's quoted prompt as the standalone prompt. When the user "
                    "selects an item from a list in an earlier assistant answer, such as "
                    "'make me the first one', build the standalone prompt from that "
                    "list item's text. "
                    "Always call choose_route and do not answer normally.\n"
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
            settings={"temperature": 0, "max_tokens": 192 if referenced_text else 96},
        )
        try:
            async with asyncio.timeout(PLANNER_TIMEOUT_SECONDS):
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
            plan = self._from_tool(
                arguments,
                fallback=fallback,
                text=text,
                input_artifact_ids=input_artifact_ids,
                has_prior_image=has_prior_image,
            )
            if (
                referenced_text
                and not input_artifact_ids
                and fallback.operation in {Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO}
                and plan.operation in {Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO}
            ):
                plan = plan.model_copy(
                    update={
                        "operation": fallback.operation,
                        "input_artifact_ids": [],
                    }
                )
            if (
                referenced_text
                and fallback.operation != Operation.TEXT
                and plan.operation == Operation.TEXT
            ):
                return fallback
            return self._with_text_context(plan, referenced_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return fallback

    def plan(
        self,
        *,
        text: str,
        mode: RoutingMode,
        input_artifact_ids: list[str],
        has_prior_image: bool = False,
        conversation: list[dict[str, Any]] | None = None,
    ) -> RoutingPlan:
        normalized = text.strip()
        referenced_text = self._referenced_text_context(text, conversation or [])

        def grounded(plan: RoutingPlan) -> RoutingPlan:
            return self._with_text_context(plan, referenced_text)

        if mode == RoutingMode.TEXT:
            return self._text(normalized, "explicit text mode", 1)
        if mode == RoutingMode.IMAGE:
            use_prior_image = bool(
                has_prior_image
                and not referenced_text
                and (_PRIOR_IMAGE_EDIT.search(normalized) or _PRIOR_IMAGE_SOURCE.search(normalized))
            )
            operation = (
                Operation.IMAGE_TO_IMAGE
                if input_artifact_ids or use_prior_image
                else Operation.TEXT_TO_IMAGE
            )
            return grounded(
                self._media(operation, normalized, input_artifact_ids, "explicit image mode", 1)
            )
        if mode == RoutingMode.VIDEO:
            use_prior_image = bool(
                has_prior_image
                and not referenced_text
                and (
                    _DIRECT_VIDEO.search(normalized)
                    or _PRIOR_IMAGE_EDIT.search(normalized)
                    or _PRIOR_IMAGE_SOURCE.search(normalized)
                )
            )
            operation = (
                Operation.IMAGE_TO_VIDEO
                if input_artifact_ids or use_prior_image
                else Operation.TEXT_TO_VIDEO
            )
            return grounded(
                self._media(operation, normalized, input_artifact_ids, "explicit video mode", 1)
            )

        if not input_artifact_ids and _ANOTHER_GENERATION.match(normalized):
            repeat = self._last_generation(conversation or [])
            if repeat:
                modality, prompt = repeat
                operation = (
                    Operation.TEXT_TO_VIDEO if modality == "video" else Operation.TEXT_TO_IMAGE
                )
                return self._media(
                    operation,
                    prompt,
                    [],
                    "repeat of the last generation's prompt",
                    0.96,
                )

        selection = None if input_artifact_ids else _ORDINAL_SELECTION.match(normalized)
        if selection:
            items = self._listed_suggestions(conversation or [])
            ordinal = selection.group("ordinal")
            index = (
                _ORDINAL_WORDS[ordinal.casefold()] if ordinal else int(selection.group("number"))
            ) - 1
            if 0 <= index < len(items):
                medium = (selection.group("medium") or "image").casefold()
                operation = (
                    Operation.TEXT_TO_VIDEO
                    if medium in {"video", "animation"}
                    else Operation.TEXT_TO_IMAGE
                )
                return self._media(
                    operation,
                    items[index],
                    [],
                    f"selected suggestion {index + 1} from the last assistant list",
                    0.95,
                )

        if _DISCUSSION.search(normalized) and not re.search(
            r"\b(?:for me|now|instead)\b", normalized, re.IGNORECASE
        ):
            return self._text(normalized, "question or discussion phrasing", 0.94)

        if _TEXT_EDIT.search(normalized):
            return self._text(normalized, "clear text editing request", 0.96)

        if (
            _VIDEO_CREATE.search(normalized)
            or _DIRECT_VIDEO.search(normalized)
            or _CONTEXT_VIDEO_CREATE.search(normalized)
        ):
            operation = (
                Operation.IMAGE_TO_VIDEO
                if input_artifact_ids or (has_prior_image and not referenced_text)
                else Operation.TEXT_TO_VIDEO
            )
            return grounded(
                self._media(
                    operation,
                    normalized,
                    input_artifact_ids,
                    "clear video creation request",
                    0.9 if _UNCERTAIN.search(normalized) else 0.96,
                )
            )

        if (
            (has_prior_image or input_artifact_ids)
            and not referenced_text
            and _PRIOR_IMAGE_EDIT.search(normalized)
        ):
            return grounded(
                self._media(
                    Operation.IMAGE_TO_IMAGE,
                    normalized,
                    input_artifact_ids,
                    "clear prior-image edit request",
                    0.97,
                )
            )

        if (
            _IMAGE_CREATE.search(normalized)
            or _DIRECT_IMAGE.search(normalized)
            or _CONTEXT_IMAGE_CREATE.search(normalized)
        ):
            operation = Operation.IMAGE_TO_IMAGE if input_artifact_ids else Operation.TEXT_TO_IMAGE
            return grounded(
                self._media(
                    operation,
                    normalized,
                    input_artifact_ids,
                    "clear image creation request",
                    0.9 if _UNCERTAIN.search(normalized) else 0.96,
                )
            )

        if _TEXT_TASK.search(normalized):
            return self._text(normalized, "clear text task", 0.95)

        return self._text(normalized, "no clear media creation intent", 0.9)

    @staticmethod
    def _referenced_text_context(text: str, conversation: list[dict[str, Any]]) -> str | None:
        if not (
            _TEXT_CONTEXT_REFERENCE.search(text)
            or _CONTEXT_IMAGE_CREATE.search(text)
            or _CONTEXT_VIDEO_CREATE.search(text)
        ):
            return None

        blocks: list[str] = []
        remaining = 6_000
        for message in reversed(conversation):
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if not content or _MEDIA_CONTEXT_SUMMARY.match(content):
                continue
            label = "Assistant" if role == "assistant" else "User"
            block = f"{label}: {content}"
            if len(block) > remaining:
                block = block[: max(0, remaining - 3)].rstrip() + "..."
            if not block:
                break
            blocks.append(block)
            remaining -= len(block) + 2
            if remaining <= 0 or len(blocks) >= 4:
                break
        if not blocks:
            return None
        return "\n\n".join(reversed(blocks))

    @staticmethod
    def _with_text_context(plan: RoutingPlan, referenced_text: str | None) -> RoutingPlan:
        if not referenced_text or plan.operation == Operation.TEXT:
            return plan
        if referenced_text in plan.standalone_prompt:
            return plan
        return plan.model_copy(
            update={
                "standalone_prompt": (
                    f"{plan.standalone_prompt.strip()}\n\nSource chat text:\n{referenced_text}"
                )
            }
        )

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
        output_count = arguments.get("output_count", self.requested_output_count(text))
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
            or isinstance(output_count, bool)
            or not isinstance(output_count, int)
            or not 1 <= output_count <= 16
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
        plan.output_count = max(output_count, self.requested_output_count(text))
        # A low-confidence model answer cannot override a strong deterministic text safeguard.
        if (
            fallback.operation == Operation.TEXT
            and fallback.confidence >= 0.94
            and confidence < 0.9
        ):
            return fallback
        return plan

    @staticmethod
    def _last_generation(conversation: list[dict[str, Any]]) -> tuple[str, str] | None:
        """Return the modality and prompt of the most recent generated-media summary."""
        for message in reversed(conversation):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            if not _MEDIA_CONTEXT_SUMMARY.match(content):
                continue
            match = _GENERATED_PROMPT.match(content)
            if not match:
                return None
            modality = "video" if "video" in match.group("media") else "image"
            return modality, match.group("prompt")
        return None

    @staticmethod
    def _listed_suggestions(conversation: list[dict[str, Any]]) -> list[str]:
        """Return list items from the most recent assistant text answer."""
        for message in reversed(conversation):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            if _MEDIA_CONTEXT_SUMMARY.match(content):
                continue
            items = []
            for line in content.splitlines():
                match = _LIST_ITEM.match(line)
                if match:
                    items.append(match.group("item"))
            return items
        return []

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
            output_count=ModalityRouter.requested_output_count(prompt),
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def requested_output_count(text: str) -> int:
        match = _OUTPUT_COUNT.search(text)
        if not match:
            return 1
        raw = match.group("count").casefold()
        return int(raw) if raw.isdigit() else _COUNT_WORDS[raw]

    @staticmethod
    def per_output_media_prompt(
        prompt: str,
        operation: Operation,
        output_count: int,
    ) -> str:
        """Compile one engine prompt from a request for several media outputs."""

        cleaned = prompt.strip()
        if output_count <= 1 or operation == Operation.TEXT:
            return cleaned
        generation_prompt, separator, context = cleaned.partition("\n\nSource chat text:")
        medium = (
            "video" if operation in {Operation.TEXT_TO_VIDEO, Operation.IMAGE_TO_VIDEO} else "image"
        )
        generation_prompt, substitutions = _OUTPUT_COUNT.subn(
            f"one {medium}",
            generation_prompt,
            count=1,
        )
        if not substitutions:
            return cleaned
        generation_prompt = _PER_OUTPUT_DISTRIBUTOR.sub("", generation_prompt)
        generation_prompt = re.sub(r"[ \t]+", " ", generation_prompt)
        generation_prompt = re.sub(r"\s+([,.;:!?])", r"\1", generation_prompt).strip()
        return f"{generation_prompt}{separator}{context}" if separator else generation_prompt
