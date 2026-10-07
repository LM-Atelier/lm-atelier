"""Whether the graph that actually ran still carries the binding that was proven.

The geometry proof is over the STORED revision, and a run does not always
dispatch that document: a LoRA stack rewrites the graph on the way out. So the
proof alone is evidence about a file rather than about a generation, and the gap
between them is where a verdict about produced size would otherwise be built on
sand.

The load-bearing case here is the ordinary one - a real LoRA transform must keep
the answer True - because a confirmation that refused correct runs would be worse
than none: it would make the common case look like the broken one.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.auxiliary_assets import transform_lora_graph
from local_lm.model_planner import workflow_artifact_contract
from local_lm.workflow_output_geometry import (
    executed_graph_carries_the_proof,
    prove_workflow_output_geometry,
)


def _graph() -> dict[str, Any]:
    """The smallest graph the proof accepts, with a model path a LoRA can rewrite."""
    return {
        "loader": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "fixture.safetensors"},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": "${width}", "height": "${height}", "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"latent_image": ["latent", 0], "model": ["loader", 0]},
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}},
    }


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "width": {
                "type": "integer",
                "default": 1024,
                "minimum": 128,
                "maximum": 2048,
                "multipleOf": 64,
            },
            "height": {
                "type": "integer",
                "default": 768,
                "minimum": 128,
                "maximum": 2048,
                "multipleOf": 64,
            },
        },
    }


def _proof(graph: dict[str, Any] | None = None) -> Any:
    stored = _graph() if graph is None else graph
    schema = _schema()
    result = prove_workflow_output_geometry(
        workflow_id="wf-1",
        revision_id="rev-1",
        operation="text_to_image",
        engine="comfyui",
        api_graph=stored,
        input_schema=schema,
        dependencies={},
        artifact_sha256=workflow_artifact_contract(
            operation="text_to_image",
            engine="comfyui",
            api_graph=stored,
            input_schema=schema,
            dependencies={},
        ),
        trusted=True,
    )
    assert result.available, result.reason
    return result.proof


def test_the_graph_it_was_proven_from_carries_it() -> None:
    stored = _graph()

    assert executed_graph_carries_the_proof(_proof(stored), deepcopy(stored)) is True


def test_a_real_lora_transform_keeps_the_binding() -> None:
    """The ordinary case, through the transform the dispatcher actually applies.

    A run with a LoRA stack does not execute the stored graph. The reason the
    binding survives is structural rather than lucky - the transform rewires
    model and clip links, which are lists, while a declared dimension is a
    string - but "structural" is an argument, and this is the measurement.
    """
    stored = _graph()
    proof = _proof(stored)
    executed = transform_lora_graph(
        deepcopy(stored),
        {"model": ["loader", 0], "clip": ["loader", 1]},
        [{"comfy_name": "fixture-lora.safetensors", "model_strength": 0.8, "clip_strength": 0.8}],
    )

    assert executed != stored, "the transform must actually have rewritten the graph"
    assert executed_graph_carries_the_proof(proof, executed) is True


@pytest.mark.parametrize(
    "divergence",
    ["upscale_between_sampler_and_decode", "second_save_on_a_scaled_branch", "relocated_latent"],
)
def test_a_graph_that_diverged_does_not_carry_it(divergence: str) -> None:
    """Three ways a run can produce something other than the declared pair.

    Each leaves a graph the walk can still read; what changes is WHICH nodes form
    the spine. Comparing the whole spine rather than merely re-running the walk
    is what separates these from the proven one.
    """
    stored = _graph()
    proof = _proof(stored)
    executed = deepcopy(stored)

    if divergence == "upscale_between_sampler_and_decode":
        executed["upscale"] = {
            "class_type": "LatentUpscaleBy",
            "inputs": {"samples": ["sampler", 0], "scale_by": 2.0},
        }
        executed["decode"]["inputs"]["samples"] = ["upscale", 0]
    elif divergence == "second_save_on_a_scaled_branch":
        executed["scale"] = {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["decode", 0], "scale_by": 2.0},
        }
        executed["save-2"] = {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}}
    else:
        executed["latent-2"] = deepcopy(executed["latent"])
        del executed["latent"]
        executed["sampler"]["inputs"]["latent_image"] = ["latent-2", 0]

    assert executed_graph_carries_the_proof(proof, executed) is False


def test_it_refuses_a_proof_it_did_not_mint() -> None:
    """The answer must not be obtainable by handing it something proof-shaped.

    Everything else in this module is reserved to the verifier for the same
    reason; a confirmation that accepted a plausible object would let a caller
    manufacture the very evidence it exists to supply.
    """

    class LooksLikeAProof:
        latent_node_id = "latent"
        sampler_node_ids = ("sampler",)
        decode_node_ids = ("decode",)
        save_node_ids = ("save",)

    assert executed_graph_carries_the_proof(LooksLikeAProof(), _graph()) is False
    assert executed_graph_carries_the_proof(None, _graph()) is False


@pytest.mark.parametrize("graph", [None, "a graph", [], 7, {}, {"save": "not a node"}])
def test_it_answers_rather_than_raises_for_anything_unreadable(graph: object) -> None:
    """It runs after a generation has already succeeded, so it cannot throw."""

    assert executed_graph_carries_the_proof(_proof(), graph) is False
