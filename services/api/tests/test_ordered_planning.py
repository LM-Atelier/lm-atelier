from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.domain import RoutingMode
from local_lm.ordered_planning import OrderedPlanCompiler
from local_lm.schemas import OrderedStepInput, OrderedStepIntent, OrderedWorkIntent


class OrderedToolAdapter(MockChatAdapter):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__()
        self.payload = payload
        self.called = False

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.called = True
        assert request.tools[0]["function"]["name"] == "compile_ordered_work"
        yield ChatEvent(
            type="tool_delta",
            data={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "name": "compile_ordered_work",
                            "arguments": self.payload,
                        },
                    }
                ]
            },
        )


def test_compiles_text_image_video_text_chain() -> None:
    intent = OrderedPlanCompiler.deterministic(
        (
            "Write a short story about a paper boat, then create an image based on it, "
            "then animate the image into a video, then summarize the result"
        ),
        RoutingMode.AUTO,
    )
    assert intent
    assert [step.mode for step in intent.steps] == ["text", "image", "video", "text"]
    assert [step.depends_on for step in intent.steps] == [
        [],
        ["step_1"],
        ["step_2"],
        ["step_3"],
    ]
    assert [binding.kind for step in intent.steps for binding in step.inputs] == [
        "text_context",
        "artifact",
        "artifact",
    ]


def test_explicit_media_can_start_an_ordered_vision_text_chain() -> None:
    intent = OrderedPlanCompiler.deterministic(
        "Analyze the attached image, then write a concise critique",
        RoutingMode.AUTO,
        has_media_input=True,
    )
    assert intent
    assert [step.mode for step in intent.steps] == ["text", "text"]
    assert intent.steps[1].inputs[0].kind == "text_context"


@pytest.mark.parametrize("mode", [RoutingMode.TEXT, RoutingMode.IMAGE, RoutingMode.VIDEO])
def test_explicit_modes_never_compile_an_ordered_plan(mode: RoutingMode) -> None:
    assert (
        OrderedPlanCompiler.deterministic(
            "Write a story, then create an image from it",
            mode,
        )
        is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "Explain how text to image to video systems work",
        "Write a story, then summarize it",
        "Create an image with a road that turns into a river",
        "Open C:\\private\\workflow.json then execute it",
    ],
)
def test_false_or_unsafe_sequences_do_not_compile(text: str) -> None:
    assert OrderedPlanCompiler.deterministic(text, RoutingMode.AUTO) is None


def test_rejects_forward_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="earlier step"):
        OrderedPlanCompiler.validate(
            OrderedWorkIntent(
                confidence=1,
                reason="invalid cycle",
                steps=[
                    OrderedStepIntent(
                        id="step_1",
                        mode="text",
                        prompt="Write",
                        depends_on=["step_2"],
                    ),
                    OrderedStepIntent(
                        id="step_2",
                        mode="image",
                        prompt="Draw",
                    ),
                ],
            )
        )


def test_rejects_incompatible_binding_type() -> None:
    with pytest.raises(ValueError, match="incompatible input"):
        OrderedPlanCompiler.validate(
            OrderedWorkIntent(
                confidence=1,
                reason="invalid binding",
                steps=[
                    OrderedStepIntent(id="step_1", mode="text", prompt="Write"),
                    OrderedStepIntent(
                        id="step_2",
                        mode="image",
                        prompt="Draw",
                        depends_on=["step_1"],
                        inputs=[
                            OrderedStepInput(
                                source_step_id="step_1",
                                kind="artifact",
                            )
                        ],
                    ),
                ],
            )
        )


def test_schema_bounds_step_count_and_prompt_size() -> None:
    with pytest.raises(ValidationError):
        OrderedWorkIntent(confidence=1, reason="too short", steps=[])
    with pytest.raises(ValidationError):
        OrderedStepIntent(id="step_1", mode="text", prompt="x" * 20_001)
    with pytest.raises(ValueError, match="at most 8"):
        OrderedPlanCompiler.deterministic(
            " then ".join(["create an image"] * 9),
            RoutingMode.AUTO,
        )


async def test_model_planner_emits_only_validated_typed_intent() -> None:
    intent = await OrderedPlanCompiler.plan_with_model(
        adapter=OrderedToolAdapter(
            {
                "is_ordered": True,
                "reason": "two requested creative actions",
                "confidence": 0.82,
                "requires_confirmation": True,
                "steps": [
                    {
                        "id": "visual",
                        "mode": "image",
                        "prompt": "Sketch the concept",
                        "depends_on": [],
                        "inputs": [],
                    },
                    {
                        "id": "motion",
                        "mode": "video",
                        "prompt": "Bring the visual to life",
                        "depends_on": ["visual"],
                        "inputs": [
                            {
                                "source_step_id": "visual",
                                "kind": "artifact",
                            }
                        ],
                    },
                ],
            }
        ),
        text="Sketch the concept, then bring it to life",
        mode=RoutingMode.AUTO,
    )
    assert intent
    assert intent.requires_confirmation is True
    assert [step.mode for step in intent.steps] == ["image", "video"]


async def test_model_planner_rejects_forward_dependency() -> None:
    intent = await OrderedPlanCompiler.plan_with_model(
        adapter=OrderedToolAdapter(
            {
                "is_ordered": True,
                "reason": "invalid",
                "confidence": 1,
                "requires_confirmation": False,
                "steps": [
                    {
                        "id": "later",
                        "mode": "image",
                        "prompt": "Draw",
                        "depends_on": ["future"],
                        "inputs": [],
                    },
                    {
                        "id": "future",
                        "mode": "text",
                        "prompt": "Write",
                        "depends_on": [],
                        "inputs": [],
                    },
                ],
            }
        ),
        text="Draw something, then write about it",
        mode=RoutingMode.AUTO,
    )
    assert intent is None


async def test_model_planner_does_not_delay_ordinary_then_reasoning() -> None:
    adapter = OrderedToolAdapter({})
    intent = await OrderedPlanCompiler.plan_with_model(
        adapter=adapter,
        text="If the first condition is true, then explain the second condition",
        mode=RoutingMode.AUTO,
    )
    assert intent is None
    assert adapter.called is False
