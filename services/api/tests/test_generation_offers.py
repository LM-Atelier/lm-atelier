from __future__ import annotations

import json

import pytest

from local_lm.adapters.base import ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.generation_offers import (
    GENERATION_OFFER_TOOL,
    MAX_GENERATION_OFFER_ARGUMENT_CHARS,
    GenerationOfferCollector,
    generation_offer_from_metadata,
    generation_offer_metadata,
    is_explicit_generation_assent,
    ordered_intent_for_offer,
    routing_plan_for_offer,
    validate_generation_offer,
)
from local_lm.schemas import RoutingReasonCode


def tool_delta(*, name: str = "offer_generation", arguments: object) -> dict[str, object]:
    return {
        "tool_calls": [
            {
                "index": 0,
                "function": {"name": name, "arguments": arguments},
            }
        ]
    }


def test_generation_offer_tool_is_bounded_and_non_executing() -> None:
    function = GENERATION_OFFER_TOOL["function"]
    assert function["name"] == "offer_generation"
    assert "never starts generation" in function["description"]
    parameters = function["parameters"]
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["items"]["minItems"] == 1
    assert parameters["properties"]["items"]["maxItems"] == 8


@pytest.mark.asyncio
async def test_mock_adapter_emits_an_offer_only_when_the_prompt_requests_one() -> None:
    adapter = MockChatAdapter()
    offered = [
        event
        async for event in adapter.stream(
            ChatRequest(
                run_id="offer",
                messages=[{"role": "user", "content": "Offer two image prompts."}],
                tools=[GENERATION_OFFER_TOOL],
            )
        )
    ]
    assert [event.type for event in offered] == ["tool_delta", "complete"]
    collector = GenerationOfferCollector()
    collector.add(offered[0].data)
    offer = collector.offer()
    assert offer is not None
    assert len(offer.items) == 2

    ordinary = [
        event
        async for event in adapter.stream(
            ChatRequest(
                run_id="ordinary",
                messages=[{"role": "user", "content": "Explain local inference."}],
                tools=[GENERATION_OFFER_TOOL],
            )
        )
    ]
    assert any(event.type == "delta" for event in ordinary)
    assert all(event.type != "tool_delta" for event in ordinary)


def test_collector_reassembles_fragmented_offer_and_builds_single_plan() -> None:
    collector = GenerationOfferCollector()
    collector.add(
        tool_delta(
            arguments='{"message":"Generate this?","items":[{"mode":"ima',
        )
    )
    collector.add(
        tool_delta(
            name="",
            arguments='ge","prompt":"A blue cup."}]}',
        )
    )

    offer = collector.offer()
    assert offer is not None
    assert offer.message == "Generate this?"
    assert offer.items[0].prompt == "A blue cup."
    plan = routing_plan_for_offer(offer)
    assert plan.operation.value == "text_to_image"
    assert plan.standalone_prompt == "A blue cup."
    assert plan.confidence == 1
    assert plan.reason_code == RoutingReasonCode.GENERATION_OFFER_ACCEPTED
    assert plan.model_dump(mode="json")["reason_code"] == "generation_offer_accepted"


def test_multiple_offer_items_become_ordered_independent_media_steps() -> None:
    offer = validate_generation_offer(
        {
            "message": "Generate these?",
            "items": [
                {"mode": "image", "prompt": "First image."},
                {"mode": "video", "prompt": "Second video."},
                {"mode": "image", "prompt": "Third image."},
            ],
        }
    )
    assert offer is not None

    intent = ordered_intent_for_offer(offer)
    assert [step.mode for step in intent.steps] == ["image", "video", "image"]
    assert [step.depends_on for step in intent.steps] == [[], ["offer_1"], ["offer_2"]]
    assert all(not step.inputs for step in intent.steps)
    assert intent.requires_confirmation is True


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "", "items": [{"mode": "image", "prompt": "Valid"}]},
        {"message": "Generate?", "items": []},
        {"message": "Generate?", "items": [{"mode": "audio", "prompt": "Valid"}]},
        {"message": "Generate?", "items": [{"mode": "image", "prompt": "   "}]},
        {
            "message": "Generate?",
            "items": [{"mode": "image", "prompt": "Valid", "path": "C:/private"}],
        },
        {
            "message": "Generate?",
            "items": [{"mode": "image", "prompt": str(index)} for index in range(9)],
        },
    ],
)
def test_malformed_offer_payloads_are_rejected(payload: object) -> None:
    collector = GenerationOfferCollector()
    collector.add(tool_delta(arguments=json.dumps(payload)))
    assert collector.saw_tool_calls is True
    assert collector.offer() is None


def test_wrong_tool_multiple_calls_and_invalid_json_are_rejected() -> None:
    wrong_name = GenerationOfferCollector()
    wrong_name.add(tool_delta(name="run_generation", arguments="{}"))
    assert wrong_name.offer() is None

    invalid_json = GenerationOfferCollector()
    invalid_json.add(tool_delta(arguments="{"))
    assert invalid_json.offer() is None

    multiple = GenerationOfferCollector()
    multiple.add(tool_delta(arguments='{"message":"Generate?","items":[]}'))
    multiple.add(
        {
            "tool_calls": [
                {
                    "index": 1,
                    "function": {"name": "offer_generation", "arguments": "{}"},
                }
            ]
        }
    )
    assert multiple.offer() is None


def test_collector_rejects_non_integer_indexes_and_oversized_fragments() -> None:
    invalid_index = GenerationOfferCollector()
    invalid_index.add(
        {
            "tool_calls": [
                {
                    "index": True,
                    "function": {"name": "offer_generation", "arguments": "{}"},
                }
            ]
        }
    )
    assert invalid_index.offer() is None

    oversized = GenerationOfferCollector()
    oversized.add(tool_delta(arguments="x" * (MAX_GENERATION_OFFER_ARGUMENT_CHARS + 1)))
    assert oversized.malformed is True
    oversized.add(tool_delta(arguments="ignored"))
    assert oversized.offer() is None


def test_offer_metadata_round_trip_requires_the_current_version() -> None:
    offer = validate_generation_offer(
        {
            "message": "Generate this?",
            "items": [{"mode": "video", "prompt": "A slow pan."}],
        }
    )
    assert offer is not None
    metadata = generation_offer_metadata(offer)
    assert generation_offer_from_metadata(metadata) == offer
    assert generation_offer_from_metadata({**metadata, "version": "future"}) is None


@pytest.mark.parametrize(
    "text",
    [
        "Yes",
        "Yes, please!",
        "Go ahead.",
        "Please generate them",
        "Okay, generate all of them.",
        "Do it",
    ],
)
def test_explicit_generation_assents_are_narrowly_recognized(text: str) -> None:
    assert is_explicit_generation_assent(text) is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes" + "!" * 100_000, True),
        ("Yes," + " " * 100_000 + "please", True),
        ("No" + "!" * 100_000, False),
        ("Yes," + " " * 100_000 + "but change it", False),
    ],
    ids=[
        "punctuation-assent",
        "spacing-assent",
        "punctuation-refusal",
        "spacing-qualified-reply",
    ],
)
def test_assent_normalization_is_linear_on_long_repeated_input(
    text: str,
    expected: bool,
) -> None:
    assert is_explicit_generation_assent(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "No thanks",
        "Maybe later",
        "What time is it?",
        "Yes, but change the second prompt",
        "Generate a different image instead",
        "I said yes yesterday",
    ],
)
def test_refusals_and_unrelated_text_are_not_assent(text: str) -> None:
    assert is_explicit_generation_assent(text) is False
