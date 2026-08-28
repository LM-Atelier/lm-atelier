"""Fields whose values are a closed set must say so in the schema.

A field typed `str` tells a consumer nothing, so the browser compares prose or
branches on values nobody wrote down. That has already caused a real defect
once: a server union gained `review_required`, the browser's union kept three
members, and a workflow awaiting review was reported to the user as one that
cannot run on this machine.

These seven are the group where no judgement was needed - the ORM column
already used the enum, so the vocabulary was declared and enforced one layer
down and only the API schema was throwing it away. Pinned so that cannot
quietly return.
"""

from __future__ import annotations

from enum import StrEnum

import pytest
from pydantic import BaseModel

from local_lm.domain import (
    ArtifactKind,
    JobKind,
    JobStatus,
    MessageStatus,
    PartType,
    RoutingMode,
    RunStatus,
)
from local_lm.schemas import (
    ArtifactOut,
    ChatDetail,
    ChatOut,
    JobOut,
    MessageOut,
    MessagePartOut,
    RunOut,
)

CLOSED: list[tuple[type[BaseModel], str, type[StrEnum]]] = [
    (ArtifactOut, "kind", ArtifactKind),
    (JobOut, "kind", JobKind),
    (JobOut, "status", JobStatus),
    (MessageOut, "status", MessageStatus),
    (MessagePartOut, "type", PartType),
    (ChatOut, "routing_mode", RoutingMode),
    (RunOut, "status", RunStatus),
    # ChatDetail extends ChatOut, so declaring the parent closed this one
    # too. Pinned because it is easy to undo by accident: redeclaring
    # routing_mode on the subclass as a str would silently reopen it.
    (ChatDetail, "routing_mode", RoutingMode),
]


def _schema_values(model: type[BaseModel], field: str) -> set[str]:
    """The values the exported schema admits for one field."""

    schema = model.model_json_schema()
    spec = schema["properties"][field]
    reference = spec.get("$ref") or next(
        (option.get("$ref") for option in spec.get("allOf", []) if option.get("$ref")), None
    )
    if reference:
        target = reference.rsplit("/", 1)[-1]
        spec = schema.get("$defs", {}).get(target, {})
    values = spec.get("enum") or next(
        (option.get("enum") for option in spec.get("anyOf", []) if option.get("enum")), None
    )
    return set(values or ())


@pytest.mark.parametrize(
    ("model", "field", "vocabulary"),
    CLOSED,
    ids=[f"{m.__name__}.{f}" for m, f, _ in CLOSED],
)
def test_the_schema_publishes_the_whole_vocabulary(
    model: type[BaseModel], field: str, vocabulary: type[StrEnum]
) -> None:
    """And the WHOLE vocabulary, not a hand-copied subset.

    Comparing against the enum rather than a literal list is the point: a new
    member added to the enum appears here without anyone remembering to update
    a second list, which is the failure this pins.
    """

    published = _schema_values(model, field)

    assert published, f"{model.__name__}.{field} publishes no vocabulary; it is an open string"
    assert published == {member.value for member in vocabulary}


def test_the_wire_form_is_still_a_plain_string() -> None:
    """Declaring the vocabulary must not change what consumers receive.

    A StrEnum serialises as its value, so this is a schema change and not a
    protocol change - and that is worth a test rather than a claim, because it
    is the whole reason this was safe to do without a version bump.
    """

    import json

    assert json.dumps(JobStatus.QUEUED) == json.dumps(JobStatus.QUEUED.value)
    assert isinstance(JobStatus.QUEUED.value, str)
