from __future__ import annotations

import pytest

from local_lm.h3_node_observation_v1 import (
    EXACT_H3_NODE_TYPES,
    INVALID_OBSERVATION,
    MAX_NODE_COUNT,
    MAX_NODE_TYPE_BYTES,
    MAX_NODE_TYPE_CHARS,
    H3NodeObservationError,
    H3NodeObservationV1,
    observe_h3_node_types,
)


def test_exact_nodes_never_authorize() -> None:
    observation = observe_h3_node_types(
        [
            "KSampler",
            "MiniMaxH3ImageToVideo",
            "EmptyMiniMaxH3LatentAV",
            "MiniMaxH3ReferenceToVideo",
            "MiniMaxH3SigmaShift",
        ]
    )
    assert observation.family_id == "minimax-h3"
    assert observation.declaration_source == "exact_workflow_node_types"
    assert observation.evidence_node_types == (
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
    )
    assert observation.variant_hints == ("image_to_video", "reference_to_video")
    assert observation.runtime_verified is False
    assert observation.installation_authorized is False
    assert observation.execution_authorized is False
    assert observation.workflow_contract_bound is False
    assert observation.activation_probe_required is False
    assert observation.geometry_claimed is False
    assert observation.reference_slots_claimed is False
    assert not hasattr(observation, "geometry_alignment_pixels")
    assert not hasattr(observation, "length_step_frames")
    assert not hasattr(observation, "reference_image_slots_max")


def test_unknown_and_empty_have_no_family() -> None:
    empty = observe_h3_node_types([])
    assert empty.family_id is None
    assert empty.declaration_source is None
    assert empty.evidence_node_types == ()
    assert empty.variant_hints == ()
    assert empty.execution_authorized is False
    unknown = observe_h3_node_types(["KSampler", "SaveImage"])
    assert unknown.family_id is None
    assert unknown.evidence_node_types == ()


@pytest.mark.parametrize(
    "declaration",
    [
        "MiniMaxH3ImageToVideoX",
        "EmptyMiniMaxH3",
        "MiniMaxH3",
        "minimaxh3imagetovideo",
        "NotMiniMaxH3ImageToVideo",
        "MiniMaxH3TextToVideo",
    ],
)
def test_refuses_substring_and_casefold_lookalikes(declaration: str) -> None:
    observation = observe_h3_node_types([declaration])
    assert observation.family_id is None
    assert observation.evidence_node_types == ()
    assert declaration not in EXACT_H3_NODE_TYPES


def test_variant_hints_follow_exact_membership() -> None:
    image = observe_h3_node_types(["MiniMaxH3ImageToVideo"])
    assert image.variant_hints == ("image_to_video",)
    reference = observe_h3_node_types(["MiniMaxH3ReferenceToVideo"])
    assert reference.variant_hints == ("reference_to_video",)
    latent = observe_h3_node_types(["EmptyMiniMaxH3LatentAV", "MiniMaxH3SigmaShift"])
    assert latent.family_id == "minimax-h3"
    assert latent.variant_hints == ()


def test_duplicates_collapse_to_sorted_evidence() -> None:
    observation = observe_h3_node_types(
        ["MiniMaxH3SigmaShift", "MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift"]
    )
    assert observation.evidence_node_types == (
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3SigmaShift",
    )


def test_node_list_bounds() -> None:
    exact_count = [f"n{index:03d}" for index in range(MAX_NODE_COUNT)]
    built = observe_h3_node_types(exact_count)
    assert built.family_id is None
    over_count = [f"n{index:03d}" for index in range(MAX_NODE_COUNT + 1)]
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(over_count)
    exact_bytes = ["x" * MAX_NODE_TYPE_CHARS] * 102 + ["y" * 16]
    assert sum(len(item.encode("utf-8")) for item in exact_bytes) == MAX_NODE_TYPE_BYTES
    assert observe_h3_node_types(exact_bytes).evidence_node_types == ()
    over_bytes = [*exact_bytes, "z"]
    assert sum(len(item.encode("utf-8")) for item in over_bytes) == MAX_NODE_TYPE_BYTES + 1
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(over_bytes)
    hostile = [f"h{index:04d}" for index in range(10_000)]
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(hostile)


def test_invalid_tokens() -> None:
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(None)
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types({"1": {"class_type": "MiniMaxH3ImageToVideo"}})
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(["MiniMaxH3ImageToVideo", 1])
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(["bad name"])
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(["x" * (MAX_NODE_TYPE_CHARS + 1)])
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        observe_h3_node_types(["bad\ud800"])


def test_public_constructor_cannot_authorize() -> None:
    with pytest.raises(H3NodeObservationError, match=INVALID_OBSERVATION):
        H3NodeObservationV1()
    with pytest.raises(TypeError):
        H3NodeObservationV1(
            schema="lm-atelier-h3-node-observation-v1",
            schema_version=1,
            execution_authorized=True,
        )


def test_exact_closed_vocabulary() -> None:
    assert (
        frozenset(
            {
                "EmptyMiniMaxH3LatentAV",
                "MiniMaxH3ImageToVideo",
                "MiniMaxH3ReferenceToVideo",
                "MiniMaxH3SigmaShift",
            }
        )
        == EXACT_H3_NODE_TYPES
    )
