"""Image-purpose selections remain explicit in requests and accepted snapshots."""

from typing import Any

import pytest
from pydantic import ValidationError

from local_lm.accepted_turn_context import AcceptedContext
from local_lm.media_input_roles import (
    image_role_bindings,
    image_source_index,
    parse_input_image_roles,
)
from local_lm.schemas import TurnRequest


@pytest.mark.parametrize(
    ("roles", "source"),
    [(["reference", "edit_source"], 1), (["reference", "reference"], None), (None, 0)],
)
def test_turn_keeps_image_purposes_in_selection_order(roles: Any, source: int | None) -> None:
    request = TurnRequest.model_validate(
        {
            "text": "Arrange the garden",
            "input_artifact_ids": ["garden-layout", "garden-canvas"],
            "input_image_roles": roles,
        }
    )
    assert request.input_image_roles == roles
    assert image_source_index(request.input_artifact_ids, request.input_image_roles) == source
    assert request.for_role("image").input_image_roles == roles


@pytest.mark.parametrize(
    ("artifacts", "roles", "reason"),
    [
        (["one", "two"], ["reference"], "match the selected images"),
        (["one"], ["reference", "edit_source"], "match the selected images"),
        (["one", "one"], ["reference", "edit_source"], "match the selected images"),
        (["one", "two"], ["edit_source", "edit_source"], "Choose one picture"),
        (["one"], ["style"], "literal_error"),
    ],
)
def test_turn_refuses_image_purposes_that_do_not_bind_one_source(
    artifacts: list[str], roles: list[str], reason: str
) -> None:
    with pytest.raises(ValidationError, match=reason):
        TurnRequest.model_validate(
            {
                "text": "Arrange the garden",
                "input_artifact_ids": artifacts,
                "input_image_roles": roles,
            }
        )


def snapshot_payload() -> dict[str, Any]:
    return {
        "run_id": "garden-run",
        "chat_id": "garden-chat",
        "messages": [],
        "input_artifact_ids": ["layout", "canvas"],
        "visual_artifact_ids": [],
        "strict_artifact_ids": [],
        "dependencies": [],
        "compiled_prompt": None,
        "standalone_prompt": "Arrange the garden",
        "artifact_ids": ["layout", "canvas"],
        "vision_settings": {},
        "vision_sampling": {"max_images": 4, "max_video_frames": 3, "max_frame_dimension": 512},
        "vision_bridge_max_tokens": 128,
        "context_limit": 1024,
        "operation": "image_to_image",
        "profile_id": None,
        "vision_profile_id": None,
        "workflow_revision_id": None,
        "settings": {},
        "chat_engine": "mock",
        "media_engine": "mock",
        "media_prompt": "Arrange the garden",
        "workflow": None,
        "workflow_activation": None,
        "auxiliary_assets": {},
        "profile": None,
        "vision_profile": None,
    }


def test_accepted_snapshot_serialization_keeps_image_purposes() -> None:
    payload = {**snapshot_payload(), "input_image_roles": ["reference", "edit_source"]}
    accepted = AcceptedContext.model_validate(payload)
    restored = AcceptedContext.model_validate(accepted.model_dump(mode="json"))
    assert restored.input_image_roles == ["reference", "edit_source"]
    assert image_source_index(restored.input_artifact_ids, restored.input_image_roles) == 1
    assert restored.input_artifact_ids == ["layout", "canvas"]


def test_an_older_snapshot_does_not_invent_explicit_image_purposes() -> None:
    accepted = AcceptedContext.model_validate(snapshot_payload())
    assert accepted.input_image_roles is None
    assert image_source_index(accepted.input_artifact_ids, accepted.input_image_roles) == 0


def test_accepted_snapshot_refuses_truncated_image_purposes() -> None:
    with pytest.raises(ValidationError, match="match the selected images"):
        AcceptedContext.model_validate({**snapshot_payload(), "input_image_roles": ["reference"]})


@pytest.mark.parametrize(
    "value", ["reference", ["unsupported"], ["reference"], ("reference", "edit_source")]
)
def test_recorded_image_purposes_refuse_changed_or_malformed_selections(value: object) -> None:
    with pytest.raises(ValueError, match="Accepted image roles are unavailable"):
        parse_input_image_roles(["layout", "canvas"], value)


def test_role_binding_preserves_reference_order_in_numbered_slots() -> None:
    assert image_role_bindings(
        ["first-reference", "canvas", "second-reference"],
        ["reference", "edit_source", "reference"],
        {
            "input_image_2": "reference",
            "input_image_0": "edit_source",
            "input_image_1": "reference",
        },
    ) == {"input_image_0": (1,), "input_image_1": (0,), "input_image_2": (2,)}


def test_a_required_reference_cannot_borrow_the_edit_source() -> None:
    with pytest.raises(ValueError, match="do not match this workflow's image purposes"):
        image_role_bindings(
            ["canvas"],
            ["edit_source"],
            {"input_image_0": "edit_source", "input_image_1": "reference"},
        )


def test_unknown_image_slots_do_not_authorize_explicit_reference_selection() -> None:
    with pytest.raises(ValueError, match="separate picture-to-edit and reference inputs"):
        image_role_bindings(["layout"], ["reference"], {"input_image": "unknown"})


def test_older_requests_keep_positional_dispatch_when_roles_are_unknown() -> None:
    assert image_role_bindings(["canvas"], None, {"input_image": "unknown"}) is None
