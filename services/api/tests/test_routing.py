from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.domain import Operation, RoutingMode
from local_lm.routing import ModalityRouter

CORPUS = Path(__file__).parent / "fixtures" / "routing-corpus-v1.json"


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


def test_versioned_routing_corpus_meets_precision_and_recall_gate() -> None:
    document = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    cases = document["cases"]
    operations = [operation.value for operation in Operation]
    confusion = {expected: {actual: 0 for actual in operations} for expected in operations}
    failures: list[str] = []

    for case in cases:
        plan = ModalityRouter().plan(
            text=case["text"],
            mode=RoutingMode(case["mode"]),
            input_artifact_ids=case.get("input_artifact_ids", []),
            has_prior_image=case.get("has_prior_image", False),
            conversation=case.get("conversation", []),
        )
        expected = case["expected"]
        actual = plan.operation.value
        confusion[expected][actual] += 1
        if actual != expected:
            failures.append(f"{case['id']}: expected {expected}, got {actual}")

    for operation in {
        Operation.TEXT.value,
        Operation.TEXT_TO_IMAGE.value,
        Operation.IMAGE_TO_IMAGE.value,
        Operation.TEXT_TO_VIDEO.value,
        Operation.IMAGE_TO_VIDEO.value,
    }:
        true_positive = confusion[operation][operation]
        predicted = sum(confusion[expected][operation] for expected in operations)
        actual = sum(confusion[operation].values())
        precision = true_positive / predicted if predicted else 0
        recall = true_positive / actual if actual else 0
        assert precision >= 0.9, f"{operation} precision was {precision:.3f}"
        assert recall >= 0.9, f"{operation} recall was {recall:.3f}"
    assert not failures, "\n".join(failures)


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


@pytest.mark.parametrize(
    ("text", "mode", "expected"),
    [
        ("Create four variations of a lighthouse", RoutingMode.IMAGE, 4),
        ("Generate 8 distinct clips of a sunrise", RoutingMode.VIDEO, 8),
        ("Make one image of an apple", RoutingMode.IMAGE, 1),
    ],
)
def test_media_output_count_is_parsed_deterministically(
    text: str,
    mode: RoutingMode,
    expected: int,
) -> None:
    plan = ModalityRouter().plan(
        text=text,
        mode=mode,
        input_artifact_ids=[],
    )
    assert plan.output_count == expected


@pytest.mark.parametrize(
    ("prompt", "operation", "expected"),
    [
        (
            "Make me 5 images, each one showing a single blue cup",
            Operation.TEXT_TO_IMAGE,
            "Make me one image, showing a single blue cup",
        ),
        (
            "Generate four variations of a lighthouse",
            Operation.TEXT_TO_IMAGE,
            "Generate one image of a lighthouse",
        ),
        (
            "Create 3 clips, with each one showing a quiet lake",
            Operation.TEXT_TO_VIDEO,
            "Create one video, showing a quiet lake",
        ),
    ],
)
def test_multi_output_media_prompts_are_compiled_for_one_engine_output(
    prompt: str,
    operation: Operation,
    expected: str,
) -> None:
    assert ModalityRouter.per_output_media_prompt(prompt, operation, 5) == expected


def test_per_output_prompt_preserves_source_chat_text_verbatim() -> None:
    prompt = (
        "Make five images based on the previous story"
        "\n\nSource chat text:\nThe wall displayed five images of a blue cup."
    )
    assert ModalityRouter.per_output_media_prompt(
        prompt,
        Operation.TEXT_TO_IMAGE,
        5,
    ) == (
        "Make one image based on the previous story"
        "\n\nSource chat text:\nThe wall displayed five images of a blue cup."
    )


def test_numeric_text_request_does_not_create_multiple_outputs() -> None:
    plan = ModalityRouter().plan(
        text="List four options for a database index",
        mode=RoutingMode.TEXT,
        input_artifact_ids=[],
    )
    assert plan.operation == Operation.TEXT
    assert plan.output_count == 1


@pytest.mark.parametrize(
    ("mode", "text", "expected"),
    [
        (RoutingMode.IMAGE, "Make it green", Operation.IMAGE_TO_IMAGE),
        (
            RoutingMode.IMAGE,
            "Use the previous image in a watercolor style",
            Operation.IMAGE_TO_IMAGE,
        ),
        (RoutingMode.VIDEO, "Make it move", Operation.IMAGE_TO_VIDEO),
        (
            RoutingMode.VIDEO,
            "Animate the previous picture with a slow camera orbit",
            Operation.IMAGE_TO_VIDEO,
        ),
        (
            RoutingMode.IMAGE,
            "Create a fresh image of a green apple",
            Operation.TEXT_TO_IMAGE,
        ),
        (
            RoutingMode.VIDEO,
            "Create a fresh video of a green apple",
            Operation.TEXT_TO_VIDEO,
        ),
    ],
)
def test_explicit_media_mode_uses_prior_image_only_for_clear_follow_ups(
    mode: RoutingMode,
    text: str,
    expected: Operation,
) -> None:
    plan = ModalityRouter().plan(
        text=text,
        mode=mode,
        input_artifact_ids=[],
        has_prior_image=True,
    )

    assert plan.operation == expected


def test_explicit_media_mode_prefers_referenced_chat_text_over_prior_image() -> None:
    plan = ModalityRouter().plan(
        text="Illustrate the previous story",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
        has_prior_image=True,
        conversation=[
            {
                "role": "assistant",
                "content": "A glass orchard floated above a quiet sea.",
            }
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert "A glass orchard floated above a quiet sea." in plan.standalone_prompt


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


@pytest.mark.parametrize(
    "text",
    [
        "Make her top red",
        "Change his jacket to black",
        "Recolor their shoes white",
        "Increase the brightness",
        "Reduce saturation slightly",
        "Give the person a blue coat",
        "Remove the coffee cup",
        "Replace the background with a beach",
        "Recolor the second person's jacket orange",
        "Change only the rightmost person into a marble statue",
        "Correct the harsh green color cast and brighten the foreground subjects",
        "Make the car blue",
        "Give him a short beard",
        "Straighten the horizon",
        "Correct the white balance",
        "Apply a shallow depth of field",
        (
            "Transform the entire photograph into a richly textured gouache "
            "illustration while preserving the subjects and composition"
        ),
    ],
)
async def test_natural_language_prior_image_edits_route_without_a_model_planner(
    text: str,
) -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnexpectedChatAdapter(),
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
    )

    assert plan.operation == Operation.IMAGE_TO_IMAGE
    assert plan.reason == "clear prior-image edit request"


@pytest.mark.parametrize(
    "text",
    [
        "Write a poem about her red top",
        "Explain how to increase brightness in a photograph",
        "Describe the background",
        "Make her a list of outfit ideas",
        "Change the subject of this paragraph",
        "Remove ambiguity from this paragraph",
        "Replace the word cat with dog in this sentence",
        "Add a paragraph about the color blue",
        "Change the first person to third person in this paragraph",
        "Generate code that blurs an image",
        "Change the shirt metaphor in the poem",
        "Give him advice about growing a beard",
        "Write instructions for changing a background",
        "Transform this paragraph into a richly textured description",
    ],
)
def test_visual_edit_language_does_not_override_clear_text_requests(text: str) -> None:
    plan = ModalityRouter().plan(
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
    )

    assert plan.operation == Operation.TEXT


def test_large_visual_edit_intent_is_handled_without_backtracking() -> None:
    plan = ModalityRouter().plan(
        text=("please " * 20_000) + "make the car blue",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
    )

    assert plan.operation == Operation.IMAGE_TO_IMAGE


def test_visual_text_on_an_image_remains_an_image_edit() -> None:
    plan = ModalityRouter().plan(
        text="Replace the word SALE with OPEN on the sign",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        has_prior_image=True,
    )

    assert plan.operation == Operation.IMAGE_TO_IMAGE


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


@pytest.mark.parametrize(
    "text",
    [
        "Make me a summary of the video call",
        "Make notes about the animation industry",
        "Create bullet points describing a movie plot",
        "Summarize the article and design a logo brief",
        "Proofread my caption and make it fit a poster",
        "Make a list of image prompts",
        "Write an outline for a documentary film",
    ],
)
def test_writing_about_a_medium_is_not_a_generation_request(text: str) -> None:
    """Each of these matched a media pattern and generated instead of writing."""
    plan = ModalityRouter().plan(text=text, mode=RoutingMode.AUTO, input_artifact_ids=[])

    assert plan.operation == Operation.TEXT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Make an image based on the previous story", Operation.TEXT_TO_IMAGE),
        ("Create a poster from the outline above", Operation.TEXT_TO_IMAGE),
        ("Generate a clip art logo of a cat", Operation.TEXT_TO_IMAGE),
        ("Make a short clip of rain on a window", Operation.TEXT_TO_VIDEO),
    ],
)
def test_the_first_thing_named_is_what_is_being_asked_for(text: str, expected: Operation) -> None:
    """A textual noun after the medium is source material, not the deliverable."""
    plan = ModalityRouter().plan(text=text, mode=RoutingMode.AUTO, input_artifact_ids=[])

    assert plan.operation == expected


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Reply with exactly: Auto ready", "clear text task"),
        ("Generate code that blurs an image", "clear text task about media"),
    ],
)
async def test_clear_text_task_does_not_invoke_model_planner(text: str, reason: str) -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnexpectedChatAdapter(),
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )

    assert plan.operation == Operation.TEXT
    assert plan.reason == reason


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


_IMAGE_SUMMARY = (
    "Generated image requested with this prompt (visual contents not inspected): "
    '"A lighthouse on a basalt cliff at dusk, oil painting".'
)
_VIDEO_SUMMARY = (
    "Generated video requested with this prompt (visual contents not inspected): "
    '"A paper boat drifting down a rain-soaked street".'
)
_SUGGESTION_LIST = (
    "Here are some image ideas:\n"
    "1. A fox curled beneath a snow-covered pine\n"
    "2. A tram crossing a viaduct in golden fog\n"
    "3. A tidepool reflecting a violet aurora"
)


@pytest.mark.parametrize(
    ("text", "summary", "expected_operation", "expected_prompt"),
    [
        (
            "Make me another",
            _IMAGE_SUMMARY,
            Operation.TEXT_TO_IMAGE,
            "A lighthouse on a basalt cliff at dusk, oil painting",
        ),
        (
            "one more",
            _VIDEO_SUMMARY,
            Operation.TEXT_TO_VIDEO,
            "A paper boat drifting down a rain-soaked street",
        ),
        (
            "please generate another one",
            _IMAGE_SUMMARY,
            Operation.TEXT_TO_IMAGE,
            "A lighthouse on a basalt cliff at dusk, oil painting",
        ),
    ],
)
def test_repeat_command_reuses_the_last_generation_prompt(
    text: str,
    summary: str,
    expected_operation: Operation,
    expected_prompt: str,
) -> None:
    plan = ModalityRouter().plan(
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[{"role": "assistant", "content": summary}],
    )

    assert plan.operation == expected_operation
    assert plan.standalone_prompt == expected_prompt


def test_repeat_command_without_a_generation_stays_text() -> None:
    plan = ModalityRouter().plan(
        text="Make me another one",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[{"role": "assistant", "content": "Only prose here."}],
    )

    assert plan.operation == Operation.TEXT


@pytest.mark.parametrize(
    ("text", "expected_operation", "expected_prompt"),
    [
        (
            "Make me the first one",
            Operation.TEXT_TO_IMAGE,
            "A fox curled beneath a snow-covered pine",
        ),
        (
            "generate #3",
            Operation.TEXT_TO_IMAGE,
            "A tidepool reflecting a violet aurora",
        ),
        (
            "Make the second one as a video",
            Operation.TEXT_TO_VIDEO,
            "A tram crossing a viaduct in golden fog",
        ),
    ],
)
def test_ordinal_selection_resolves_the_listed_suggestion(
    text: str,
    expected_operation: Operation,
    expected_prompt: str,
) -> None:
    plan = ModalityRouter().plan(
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[{"role": "assistant", "content": _SUGGESTION_LIST}],
    )

    assert plan.operation == expected_operation
    assert plan.standalone_prompt == expected_prompt


def test_ordinal_selection_beyond_the_list_stays_text() -> None:
    plan = ModalityRouter().plan(
        text="Make me the tenth one",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[{"role": "assistant", "content": _SUGGESTION_LIST}],
    )

    assert plan.operation == Operation.TEXT


@pytest.mark.parametrize(
    "bullet",
    ["-", "*", "•", "‣", "◦"],
    ids=["hyphen", "asterisk", "bullet", "triangular", "white-bullet"],
)
def test_ordinal_selection_reads_bulleted_lists(bullet: str) -> None:
    """A mangled bullet in the list pattern silently dropped these for a while."""
    listing = (
        "Here are some image ideas:\n"
        f"{bullet} A fox curled beneath a snow-covered pine\n"
        f"{bullet} A tram crossing a viaduct in golden fog"
    )

    plan = ModalityRouter().plan(
        text="Make me the second one",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[{"role": "assistant", "content": listing}],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert plan.standalone_prompt == "A tram crossing a viaduct in golden fog"


def test_ordinal_selection_skips_media_summaries_to_find_the_list() -> None:
    plan = ModalityRouter().plan(
        text="Make me the first one",
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
        conversation=[
            {"role": "assistant", "content": _SUGGESTION_LIST},
            {"role": "user", "content": "Nice ideas"},
            {"role": "assistant", "content": _IMAGE_SUMMARY},
        ],
    )

    assert plan.operation == Operation.TEXT_TO_IMAGE
    assert plan.standalone_prompt == "A fox curled beneath a snow-covered pine"
