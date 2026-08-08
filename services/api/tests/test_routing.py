from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.domain import Operation, RoutingMode
from local_lm.ordered_planning import OrderedPlanCompiler
from local_lm.routing import ModalityRouter
from local_lm.schemas import RoutingReasonCode

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
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.called = True
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


async def test_versioned_routing_corpus_matches_production_pipeline() -> None:
    document = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    cases = document["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    operations = [operation.value for operation in Operation]
    confusion = {expected: {actual: 0 for actual in operations} for expected in operations}
    failures: list[str] = []
    adapter = UnavailableChatAdapter()
    router = ModalityRouter()

    for case in cases:
        ordered = OrderedPlanCompiler.deterministic(
            case["text"],
            RoutingMode(case["mode"]),
            has_media_input=bool(case.get("input_artifact_ids", [])),
        )
        expected_ordered = case.get("expected_ordered_modes")
        if expected_ordered is not None:
            actual_ordered = [step.mode for step in ordered.steps] if ordered else None
            if actual_ordered != expected_ordered:
                failures.append(
                    f"{case['id']}: expected ordered {expected_ordered}, got {actual_ordered}"
                )
            continue
        if ordered is not None:
            failures.append(
                f"{case['id']}: expected a single route, got ordered "
                f"{[step.mode for step in ordered.steps]}"
            )
            continue

        plan = await router.plan_with_model(
            adapter=adapter,
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

    assert not failures, "\n".join(failures)
    assert adapter.called, "corpus must exercise a route that reaches the model planner"

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
    assert plan.reason_code == RoutingReasonCode.EXPLICIT_IMAGE_MODE


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
    assert plan.reason_code == RoutingReasonCode.EXPLICIT_TEXT_MODE


@pytest.mark.parametrize(
    ("mode", "text", "expected"),
    [
        (RoutingMode.IMAGE, "Make it green", Operation.IMAGE_TO_IMAGE),
        (
            RoutingMode.IMAGE,
            "Use the previous image in a watercolor style",
            Operation.IMAGE_TO_IMAGE,
        ),
        (
            RoutingMode.IMAGE,
            "Make the jacket matte black instead, as in the most recent image",
            Operation.IMAGE_TO_IMAGE,
        ),
        (
            RoutingMode.IMAGE,
            "Make the jacket matte black instead",
            Operation.IMAGE_TO_IMAGE,
        ),
        (
            RoutingMode.IMAGE,
            "Restyle the latest image with softer lighting",
            Operation.IMAGE_TO_IMAGE,
        ),
        (
            RoutingMode.IMAGE,
            "Increase the contrast in the current picture",
            Operation.IMAGE_TO_IMAGE,
        ),
        (
            RoutingMode.IMAGE,
            "Change only the cube to bright red",
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
            RoutingMode.IMAGE,
            "Make an image of a green apple",
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
    expected_code = (
        RoutingReasonCode.EXPLICIT_IMAGE_MODE
        if mode == RoutingMode.IMAGE
        else RoutingReasonCode.EXPLICIT_VIDEO_MODE
    )
    assert plan.reason_code == expected_code


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
    assert plan.reason_code == RoutingReasonCode.PRIOR_IMAGE_EDIT
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
    assert plan.reason_code == RoutingReasonCode.PRIOR_IMAGE_EDIT


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
    assert plan.reason_code == RoutingReasonCode.DISCUSSION


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
    assert plan.reason_code == RoutingReasonCode.IMAGE_CREATION


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
    ("text", "reason_code"),
    [
        ("Reply with exactly: Auto ready", RoutingReasonCode.TEXT_TASK),
        ("Generate code that blurs an image", RoutingReasonCode.TEXT_MEDIA_TASK),
    ],
)
async def test_clear_text_task_does_not_invoke_model_planner(
    text: str,
    reason_code: RoutingReasonCode,
) -> None:
    plan = await ModalityRouter().plan_with_model(
        adapter=UnexpectedChatAdapter(),
        text=text,
        mode=RoutingMode.AUTO,
        input_artifact_ids=[],
    )

    assert plan.operation == Operation.TEXT
    assert plan.reason_code == reason_code


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
    assert plan.reason_code == RoutingReasonCode.MODEL_PLANNER


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
    assert plan.reason_code == RoutingReasonCode.DEFAULT_TEXT


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
    assert plan.reason_code == RoutingReasonCode.REPEAT_LAST_GENERATION


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
    assert plan.reason_code == RoutingReasonCode.ASSISTANT_SUGGESTION_SELECTED


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


_PRIOR_VISUAL_MODES = (RoutingMode.IMAGE, RoutingMode.VIDEO)


@pytest.mark.parametrize("mode", _PRIOR_VISUAL_MODES)
def test_references_prior_visual_agrees_with_the_plan_it_predicts(mode: RoutingMode) -> None:
    """The composer's answer must be the one the submitted turn acts on.

    The browser used to decide this with its own copy of the router's patterns,
    which drifted until it misread most of the corpus. `references_prior_visual`
    exists so there is one implementation; this asserts it stays the same one.
    """
    router = ModalityRouter()
    cases = json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]
    reuse = {Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO}
    disagreements = []
    for case in cases:
        text = str(case["text"])
        plan = router.plan(
            text=text,
            mode=mode,
            input_artifact_ids=[],
            has_prior_image=True,
        )
        predicted = router.references_prior_visual(text=text, mode=mode)
        if predicted != (plan.operation in reuse):
            disagreements.append((case["id"], text, predicted, plan.operation))
    assert not disagreements, disagreements


@pytest.mark.parametrize("mode", _PRIOR_VISUAL_MODES)
def test_references_prior_visual_defers_to_a_referenced_text_answer(mode: RoutingMode) -> None:
    """Wording that points at prior prose is not a request to edit the image."""
    router = ModalityRouter()
    conversation = [{"role": "assistant", "content": "A glass orchard above a quiet sea."}]

    assert router.references_prior_visual(text="Recolor the previous image", mode=mode)
    assert not router.references_prior_visual(
        text="Illustrate the previous story",
        mode=mode,
        conversation=conversation,
    )


def test_references_prior_visual_is_false_in_text_mode() -> None:
    assert not ModalityRouter().references_prior_visual(
        text="Recolor the previous image",
        mode=RoutingMode.TEXT,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Explain why diffusion models are popular now",
        "Describe the scene instead",
        "What is the fastest sampler for me",
        "How do I make a video loop smoothly?",
    ],
)
def test_an_incidental_word_does_not_cost_a_planner_round_trip(text: str) -> None:
    """These are questions, and answering them needs no model planner.

    The discussion branch used to be skipped whenever `for me`, `now` or
    `instead` appeared *anywhere* in the message, which dropped confidence from
    0.94 to 0.90 - and 0.94 is the threshold that skips the planner. So an
    unambiguous question spent a full planner round trip because of a trailing
    word. Anchoring the phrase to the tail would not have helped: in the first
    case the "now" is at the tail.
    """
    plan = ModalityRouter().plan(text=text, mode=RoutingMode.AUTO, input_artifact_ids=[])

    assert plan.operation == Operation.TEXT
    assert plan.confidence >= 0.94, "this question should not reach the model planner"


@pytest.mark.parametrize(
    "text",
    [
        "Explain how this works, now draw me a cat",
        "What is a diffusion model, then draw me one",
        "Why is this blurry; then sharpen it",
    ],
)
def test_a_question_followed_by_a_request_still_reaches_the_planner(text: str) -> None:
    """The escape has to survive: these ask for something to be made.

    Narrowing it must not become removing it - a later imperative clause is a
    real request, and the planner is what decides between the two halves.
    """
    plan = ModalityRouter().plan(text=text, mode=RoutingMode.AUTO, input_artifact_ids=[])

    assert plan.confidence < 0.94, "this should still be offered to the model planner"


def _scene_conversation(assistant_text: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Write me a scene"},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": "no, make it darker and remove the second figure"},
        {"role": "assistant", "content": "A single figure stands on the pier at dusk."},
    ]


def test_a_media_prompt_receives_one_passage_not_a_transcript() -> None:
    """A diffusion model weights every token it is given.

    The referenced-text builder assembles up to four labelled turns and 6,000
    characters, which is right for deciding whether a message refers to earlier
    text and wrong for an image prompt: it hands the model the user's own
    instructions and several contradictory actions at once. A monitored session
    measured that degrading person count, pose and anatomy.
    """
    plan = ModalityRouter().plan(
        text="Make an image of the last scene",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
        conversation=_scene_conversation("Two figures argue beneath a broken lighthouse."),
    )

    prompt = plan.standalone_prompt
    assert "A single figure stands on the pier at dusk." in prompt
    # The user's contradictory instruction must not reach the image prompt.
    assert "remove the second figure" not in prompt
    assert "User:" not in prompt
    assert "Assistant:" not in prompt


def test_the_visual_source_is_cut_at_a_sentence_not_mid_word() -> None:
    """The old builder truncated with a bare slice and an ellipsis."""
    from local_lm.routing import MAX_VISUAL_SOURCE_CHARS

    sentence = "A lantern sways above the water. "
    long_scene = sentence * 80
    assert len(long_scene) > MAX_VISUAL_SOURCE_CHARS

    plan = ModalityRouter().plan(
        text="Make an image of the last scene",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
        conversation=[{"role": "assistant", "content": long_scene}],
    )

    source = plan.standalone_prompt.split("Source chat text:\n", 1)[1]
    assert len(source) <= MAX_VISUAL_SOURCE_CHARS
    assert source.endswith(".")
    assert "..." not in source


def test_the_source_passage_is_carried_apart_from_the_prompt() -> None:
    """Concatenation loses which half is the request and which is the source.

    The prompt compiler needs them separately: the request states what the user
    wants and wins any disagreement, the passage supplies what is visible.
    """
    plan = ModalityRouter().plan(
        text="Make an image of the last scene",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
        conversation=_scene_conversation("Two figures argue beneath a broken lighthouse."),
    )

    assert plan.text_context == "A single figure stands on the pier at dusk."
    assert plan.text_context in plan.standalone_prompt


def test_a_prompt_with_no_referenced_text_carries_no_source_passage() -> None:
    plan = ModalityRouter().plan(
        text="Draw a blue cup on a windowsill",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=[],
        conversation=_scene_conversation("Two figures argue beneath a broken lighthouse."),
    )

    assert plan.text_context is None


def test_text_operations_still_receive_the_full_referenced_context() -> None:
    """Only media prompts are narrowed; a text turn keeps its conversation."""
    router = ModalityRouter()
    conversation = _scene_conversation("Two figures argue beneath a broken lighthouse.")

    plan = router.plan(
        text="Summarize the previous story",
        mode=RoutingMode.TEXT,
        input_artifact_ids=[],
        conversation=conversation,
    )

    assert plan.operation == Operation.TEXT
    # `_with_text_context` returns early for text, so nothing is appended at all.
    assert "Source chat text:" not in plan.standalone_prompt


def test_whitespace_before_punctuation_is_dropped_in_one_pass() -> None:
    """The regex this replaced was quadratic: it retried at every offset inside
    a whitespace run, so a pasted column of blank lines cost time in proportion
    to its square - 621ms measured for 16,000 characters."""

    from local_lm.routing import _tighten_punctuation

    assert _tighten_punctuation("hello   , world  ! ok") == "hello, world! ok"
    assert _tighten_punctuation("a\n\n\n. b") == "a. b"
    assert _tighten_punctuation("tabs\t\t; and\r\n\r\n: mixed") == "tabs; and: mixed"
    assert _tighten_punctuation("no punctuation here") == "no punctuation here"
    assert _tighten_punctuation("") == ""
    assert _tighten_punctuation("   ,") == ","

    # Linear, so a pathological run finishes rather than stalling the request.
    started = time.perf_counter()
    assert _tighten_punctuation(" " * 200_000 + "x").endswith("x")
    assert time.perf_counter() - started < 1.0
