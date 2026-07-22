from __future__ import annotations

import re

from .domain import Operation, RoutingMode
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


class ModalityRouter:
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
