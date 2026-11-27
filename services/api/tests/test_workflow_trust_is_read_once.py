"""Workflow trust is read once per revision and read, and its identity encodes in one call.

A revision's trust is its review read as workflow selection reads it, which
derives the revision's identity from its graph and settings on every read. The
family list used to derive it twice for each variant, once for the variant's
readiness and once for its trusted field, and the encoding behind each
identity ran fragment by fragment in Python. The family list now reads it once
per variant, and a value whose encoding is bounded within the byte limit is
encoded in one call, which gives the same bytes as the fragments.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm import api as api_module
from local_lm import workflow_revision_reviews as reviews
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkflowRevisionReview,
)
from local_lm.workflow_revision_reviews import (
    WorkflowReviewError,
    _canonical,
    _review_subject,
    revision_identity,
    revision_is_trusted,
)

#: Values a review identity may hold, chosen for what the encoding does with
#: them: escapes, characters of every width, long and negative numbers,
#: nesting, empty containers and keys that sort differently from their order.
VALUES: list[object] = [
    None,
    True,
    False,
    0,
    -1,
    2**63,
    -(10**400),
    0.1,
    -0.0,
    1e-308,
    -1.2345678901234567e-308,
    1.7976931348623157e308,
    "",
    "plain",
    'quote " and backslash \\',
    "control \x00\x01\x1f \n\r\t\b\f",
    "accents éü and wide 字 and beyond 😀",
    "\u2028\u2029",
    [],
    {},
    [[], {}, [[]], [{}], {"a": []}],
    [{"z": 1, "a": 2}, {"b": [3, 4.5, None]}, [True, False]],
    {"b": 1, "a": 2, "A": 3, "é": 4, "10": 5, "9": 6, "": 7},
    {
        str(index): {
            "class_type": "KSampler" if index % 3 else "SaveImage",
            "inputs": {
                "seed": index,
                "cfg": 7.5,
                "sampler_name": "euler",
                "model": [str(max(index - 1, 0)), 0],
            },
        }
        for index in range(40)
    },
]


def _nested(depth: int) -> list[object]:
    """Lists nested `depth` deep."""

    value: list[object] = []
    for _ in range(depth):
        value = [value]
    return value


def _in_fragments(value: object) -> bytes:
    """The bytes the fragment-by-fragment encoding gives."""

    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return b"".join(fragment.encode("utf-8") for fragment in encoder.iterencode(value))


@pytest.mark.parametrize("value", VALUES, ids=range(len(VALUES)))
def test_a_value_encodes_to_the_same_bytes_in_one_call_as_in_fragments(value: object) -> None:
    assert _canonical(value) == _in_fragments(value)


@pytest.mark.parametrize("value", VALUES, ids=range(len(VALUES)))
def test_the_bound_is_never_below_the_encoding(value: object) -> None:
    assert reviews._encoding_bound(value, max_depth=32) >= len(_in_fragments(value))


@pytest.mark.parametrize("value", VALUES, ids=range(len(VALUES)))
def test_the_byte_limit_refuses_exactly_an_encoding_over_it(value: object) -> None:
    """The fragments stop the encoding at the limit; one call stays within it."""

    expected = _in_fragments(value)
    assert _canonical(value, max_bytes=len(expected)) == expected
    with pytest.raises(WorkflowReviewError):
        _canonical(value, max_bytes=len(expected) - 1)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        {1: "a key that is not a string"},
        {"k" * 201: "a key longer than the limit"},
        "s" * 65_537,
        "a lone surrogate \ud800",
        10**5000,
        (1, 2),
        _nested(34),
    ],
    ids=[
        "nan",
        "infinity",
        "integer-key",
        "long-key",
        "long-string",
        "lone-surrogate",
        "huge-integer",
        "tuple",
        "too-deep",
    ],
)
def test_a_value_an_identity_may_not_hold_is_still_refused(value: object) -> None:
    with pytest.raises(WorkflowReviewError):
        _canonical(value)


def _graph(seed: int) -> dict[str, Any]:
    return {
        str(index): {
            "class_type": "KSampler" if index % 3 else "SaveImage",
            "inputs": {"seed": seed + index, "steps": 20, "model": [str(max(index - 1, 0)), 0]},
        }
        for index in range(12)
    }


def _reviewed_variants(count: int) -> None:
    """Families of one reviewed image workflow each, offered for the image selector."""

    bindings = {
        "KSampler": {"kind": "core", "module": "nodes", "contract": "0" * 64},
        "SaveImage": {"kind": "core", "module": "nodes", "contract": "0" * 64},
    }
    with SessionLocal() as session:
        for number in range(count):
            family = WorkflowFamily(name=f"Neutral family {number:02d}", use_case="neutral")
            definition = WorkflowDefinition(
                family=family,
                variant_key="main",
                name=f"Neutral workflow {number:02d}",
                operation=Operation.TEXT_TO_IMAGE.value,
            )
            session.add_all(
                [
                    family,
                    definition,
                    WorkflowPreference(
                        family=family, selector_capability="image", is_default=False
                    ),
                ]
            )
            session.flush()
            revision = WorkflowRevision(
                definition=definition,
                version=1,
                engine="mock",
                api_graph_json=_graph(number),
                input_schema_json={"type": "object", "properties": {}},
                trusted=True,
            )
            session.add(revision)
            session.flush()
            identity = revision_identity(definition, revision)
            session.add(
                WorkflowRevisionReview(
                    workflow_revision_id=revision.id,
                    revision_sha256=identity,
                    subject_sha256=_review_subject(identity, bindings),
                    node_bindings_json=bindings,
                    state="approved",
                )
            )
            definition.current_revision_id = revision.id
        session.commit()


async def test_the_family_list_reads_each_variants_trust_once(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reviewed_variants(3)
    read: list[str] = []

    def counted(session: Any, revision: WorkflowRevision) -> bool:
        read.append(revision.id)
        return revision_is_trusted(session, revision)

    monkeypatch.setattr(api_module, "revision_is_trusted", counted)

    listed = await client.get("/api/workflow-families", params={"selector_capability": "image"})

    assert listed.status_code == 200, listed.text
    every_variant = [variant for family in listed.json() for variant in family["variants"]]
    neutral = [item for item in every_variant if item["name"].startswith("Neutral workflow")]
    assert len(neutral) == 3
    assert all(variant["trusted"] and variant["readiness"] == "ready" for variant in neutral)
    # A variant a model profile backs may share another variant's revision, so
    # the readings are counted per variant, not per revision.
    with_a_revision = [item for item in every_variant if item["current_revision_id"]]
    assert len(read) == len(with_a_revision), "a variant's trust was read more than once"
    assert set(read) == {item["current_revision_id"] for item in with_a_revision}
