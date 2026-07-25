from __future__ import annotations

import asyncio
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


class UnavailableChatAdapter(MockChatAdapter):
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="error", data={"error": "chat worker unavailable"})


class UnexpectedChatAdapter(MockChatAdapter):
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        raise AssertionError(f"model planner should not run for {request.messages[-1]}")
        yield


class HangingChatAdapter(MockChatAdapter):
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        del request
        await asyncio.Event().wait()
        yield ChatEvent(type="complete")


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
async def test_prior_image_edit_routes_without_a_model_planner() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnavailableChatAdapter(),
        text="Make it green",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
    )

    assert plan.operation == Operation.IMAGE_TO_IMAGE
    assert plan.confidence == 0.97
    assert plan.reason == "clear prior-image edit request"


def test_prior_image_motion_request_still_routes_to_video() -> None:
    plan = ModalityRouter().plan(
        text="Make it move",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
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
    assert plan.reason == "question or discussion phrasing"


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
        text="Surprise me with something visual",
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


@pytest.mark.asyncio
async def test_referential_media_prompt_is_grounded_in_prior_chat_text() -> None:
    story = "At dusk, a silver fox crossed a glass city while paper lanterns reflected in the rain."
    plan = await ModalityRouter().plan_with_model(
        adapter=MockChatAdapter(),
        text="Make an image based on the previous story",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
        conversation=[
            {"role": "user", "content": "Write a short story about a silver fox."},
            {"role": "assistant", "content": story},
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert plan.standalone_prompt.startswith("Make an image based on the previous story")
    assert "Source chat text:" in plan.standalone_prompt
    assert story in plan.standalone_prompt


@pytest.mark.asyncio
async def test_explicit_media_mode_uses_prior_chat_text_without_planner() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnexpectedChatAdapter(),
        text="Illustrate that story",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
        conversation=[
            {
                "role": "assistant",
                "content": "A tiny observatory drifted between violet clouds.",
            }
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert "A tiny observatory drifted between violet clouds." in plan.standalone_prompt


@pytest.mark.asyncio
async def test_referential_media_prompt_keeps_context_when_planner_is_unavailable() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnavailableChatAdapter(),
        text="Make an image from that description",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[
            {
                "role": "assistant",
                "content": "An ancient library carved into an iceberg under green aurora.",
            }
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert "An ancient library carved into an iceberg" in plan.standalone_prompt


@pytest.mark.asyncio
async def test_pronoun_media_conversion_uses_prior_chat_text() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnavailableChatAdapter(),
        text="Turn that into an image",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[
            {
                "role": "assistant",
                "content": "A clockwork whale surfaced beside a floating garden.",
            }
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert "A clockwork whale surfaced beside a floating garden." in plan.standalone_prompt


def test_local_media_route_keeps_referenced_chat_text() -> None:
    plan = ModalityRouter().plan(
        text="Make an image based on the previous story",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
        conversation=[
            {
                "role": "assistant",
                "content": "A lighthouse walked across the ocean on brass legs.",
            }
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert "A lighthouse walked across the ocean on brass legs." in plan.standalone_prompt


@pytest.mark.asyncio
async def test_clear_auto_route_does_not_invoke_model_planner() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnexpectedChatAdapter(),
        text="Make me an image of an apple",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert plan.reason == "clear image creation request"


@pytest.mark.asyncio
async def test_clear_text_task_does_not_invoke_model_planner() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnexpectedChatAdapter(),
        text="Reply with exactly: Auto ready",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )

    assert plan.operation == Operation.TEXT
    assert plan.reason == "clear text task"


@pytest.mark.asyncio
async def test_hedged_media_request_still_uses_model_planner() -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=MockChatAdapter(),
        text="Maybe create an image of a quiet harbor",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert plan.confidence == 0.6
    assert plan.reason == "model planner: deterministic mock planner"


@pytest.mark.asyncio
async def test_ambiguous_auto_route_has_a_bounded_model_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_lm.routing.PLANNER_TIMEOUT_SECONDS", 0.01)

    plan = await ModalityRouter().plan_with_model(
        adapter=HangingChatAdapter(),
        text="Surprise me",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )

    assert plan.operation == Operation.TEXT
    assert plan.reason == "no clear media creation intent"
