from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.domain import Operation, RoutingMode
from local_lm.routing import ModalityRouter


class CapturingMockChatAdapter(MockChatAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.last_request: ChatRequest | None = None

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.last_request = request
        async for event in super().stream(request):
            yield event


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Explain how image generation works", Operation.TEXT),
        ("Compare the best local video models", Operation.TEXT),
        ("Draw a lighthouse in a storm", Operation.TEXT_TO_IMAGE),
        ("Create an image of a friendly robot", Operation.TEXT_TO_IMAGE),
        ("Generate a short video of a sunrise", Operation.TEXT_TO_VIDEO),
        ("Animate this scene", Operation.TEXT_TO_VIDEO),
    ],
)
def test_auto_routing_distinguishes_creation_from_discussion(
    text: str, expected: Operation
) -> None:
    plan = ModalityRouter().plan(
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )
    assert plan.operation == expected


def test_explicit_mode_always_wins() -> None:
    plan = ModalityRouter().plan(
        text="Explain image generation",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
    )
    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert plan.confidence == 1


def test_image_input_selects_image_to_video() -> None:
    plan = ModalityRouter().plan(
        text="Animate that",
        mode=RoutingMode.AUTO,
        input_artifact_ids=["sha256:example"],
    )
    assert plan.operation == Operation.IMAGE_TO_VIDEO


@pytest.mark.asyncio
async def test_structured_planner_preserves_text_discussion() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=MockChatAdapter(),
        text="Explain why diffusion images can look oversaturated",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )
    assert plan.operation == Operation.TEXT
    assert plan.reason.startswith("model planner:")


@pytest.mark.asyncio
async def test_structured_planner_resolves_prior_image_follow_up() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=MockChatAdapter(),
        text="Make it dusk and add warm window lights",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
        conversation=[{"role": "user", "content": "Draw a mountain cabin"}],
    )
    assert plan.operation == Operation.IMAGE_TO_IMAGE
    assert plan.standalone_prompt == "Make it dusk and add warm window lights"


@pytest.mark.asyncio
async def test_structured_planner_uses_one_system_message_for_template_compatibility() -> None:
    adapter = CapturingMockChatAdapter()
    await ModalityRouter().plan_with_model(
        adapter=adapter,
        text="Explain image generation",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
    )
    assert adapter.last_request is not None
    system_messages = [
        message for message in adapter.last_request.messages if message["role"] == "system"
    ]
    assert len(system_messages) == 1
    assert "Prior generated image available: yes." in system_messages[0]["content"]
