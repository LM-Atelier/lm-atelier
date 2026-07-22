from __future__ import annotations

import pytest

from local_lm.domain import Operation, RoutingMode
from local_lm.routing import ModalityRouter


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
