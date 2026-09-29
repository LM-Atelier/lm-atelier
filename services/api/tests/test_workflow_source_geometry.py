"""Trace pixel transport without claiming a runtime output-size guarantee."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest


def graph() -> dict[str, Any]:
    return {
        "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "pad": {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": ["source", 0],
                "left": 0,
                "top": 0,
                "right": 0,
                "bottom": 0,
                "feathering": 0,
            },
        },
        "checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "${checkpoint}",
            },
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "Neutral geometric shapes",
                "clip": ["checkpoint", 1],
            },
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "Low contrast",
                "clip": ["checkpoint", 1],
            },
        },
        "encode": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["pad", 0],
                "mask": ["pad", 1],
                "vae": ["checkpoint", 2],
                "grow_mask_by": 0,
            },
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["checkpoint", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["encode", 0],
                "seed": 1,
                "steps": 2,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["sample", 0],
                "vae": ["checkpoint", 2],
            },
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["decode", 0],
                "filename_prefix": "fixture",
            },
        },
    }


@pytest.mark.parametrize("separate_vae", [False, True])
def test_trace_binds_source_mask_and_same_vae_without_mutating_graph(separate_vae: bool) -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = graph()
    if separate_vae:
        value["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": "fixture.safetensors"}}
        value["encode"]["inputs"]["vae"] = ["vae", 0]
        value["decode"]["inputs"]["vae"] = ["vae", 0]
    before = deepcopy(value)
    route = trace_source_fit_route(value, "save")
    assert route is not None
    assert route.source_node_id == "source"
    assert route.source_parameter == "input_image"
    assert route.pad_node_id == "pad"
    assert route.encode_node_id == "encode"
    assert route.sampler_node_id == "sample"
    assert route.decode_node_id == "decode"
    assert route.vae_node_id == ("vae" if separate_vae else "checkpoint")
    assert route.vae_output_index == (0 if separate_vae else 2)
    assert value == before


def inpaint_conditioning_graph() -> dict[str, Any]:
    """The official outpaint shape: conditioning carries the source, feathered, no composite."""
    value = graph()
    value["pad"]["inputs"]["feathering"] = 24
    del value["encode"]
    value["condition"] = {
        "class_type": "InpaintModelConditioning",
        "inputs": {
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "vae": ["checkpoint", 2],
            "pixels": ["pad", 0],
            "mask": ["pad", 1],
            "noise_mask": False,
        },
    }
    value["sample"]["inputs"].update(
        positive=["condition", 0], negative=["condition", 1], latent_image=["condition", 2]
    )
    return value


def test_the_inpaint_conditioning_route_is_recognized_with_feathering() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = inpaint_conditioning_graph()
    before = deepcopy(value)
    route = trace_source_fit_route(value, "save")
    assert route is not None
    assert route.encode_node_id == "condition"
    assert route.pad_node_id == "pad"
    assert route.composite_node_id is None
    assert value == before


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("positive", ["positive", 0]),
        ("negative", ["condition", 0]),
        ("latent_image", ["condition", 0]),
    ],
)
def test_a_sampler_that_bypasses_the_inpaint_conditioning_is_refused(
    field: str, replacement: Any
) -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = inpaint_conditioning_graph()
    value["sample"]["inputs"][field] = replacement
    assert trace_source_fit_route(value, "save") is None


def test_a_second_padding_step_leaves_no_single_place_for_the_margins() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = graph()
    value["other_pad"] = deepcopy(value["pad"])
    assert trace_source_fit_route(value, "save") is None


@pytest.mark.parametrize(
    "node,field,replacement",
    [
        ("save", "images", ["source", 0]),
        ("decode", "samples", ["encode", 0]),
        ("decode", "vae", ["checkpoint", 1]),
        ("sample", "latent_image", ["source", 0]),
        ("sample", "denoise", 0),
        ("sample", "denoise", True),
        ("sample", "denoise", "1"),
        ("sample", "denoise", "$" + "{}"),
        ("sample", "denoise", "$" + "{checkpoint}"),
        ("sample", "denoise", "$" + "{input_image}"),
        ("encode", "pixels", ["source", 0]),
        ("encode", "mask", ["pad", 0]),
        ("encode", "mask", ["pad", True]),
        ("pad", "image", ["source", 1]),
        ("pad", "left", "$" + "{outpaint_left_px}"),
        ("pad", "top", True),
        ("pad", "right", 1.5),
        ("source", "image", "unrelated.png"),
        ("source", "image", "${input_image_1}"),
    ],
)
def test_ambiguous_or_unbound_transport_is_refused(node: str, field: str, replacement: Any) -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = graph()
    value[node]["inputs"][field] = replacement
    assert trace_source_fit_route(value, "save") is None


def test_a_strength_mapped_to_its_own_setting_is_named_for_the_run_to_pin() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    written = trace_source_fit_route(graph(), "save")
    assert written is not None and written.strength_parameter is None
    value = graph()
    value["sample"]["inputs"]["denoise"] = "$" + "{denoise}"
    mapped = trace_source_fit_route(value, "save")
    assert mapped is not None and mapped.strength_parameter == "denoise"
    # A setting that another input also reads cannot be pinned for this sampler alone.
    value["negative"]["inputs"]["text"] = "$" + "{denoise}"
    assert trace_source_fit_route(value, "save") is None


def test_different_vae_instances_are_not_treated_as_the_same_binding() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = graph()
    value["other"] = deepcopy(value["checkpoint"])
    value["decode"]["inputs"]["vae"] = ["other", 2]
    assert trace_source_fit_route(value, "save") is None


@pytest.mark.parametrize(
    "value,save", [(None, "save"), ({}, "save"), ([], "save"), (graph(), True)]
)
def test_malformed_graph_or_save_identity_is_refused(value: Any, save: Any) -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    assert trace_source_fit_route(value, save) is None


def composited_graph() -> dict[str, Any]:
    value = graph()
    value["composite"] = {
        "class_type": "ImageCompositeMasked",
        "inputs": {
            "destination": ["pad", 0],
            "source": ["decode", 0],
            "mask": ["pad", 1],
            "x": 0,
            "y": 0,
            "resize_source": True,
        },
    }
    value["save"]["inputs"]["images"] = ["composite", 0]
    return value


def test_trace_identifies_canvas_and_source_restoring_composite() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = composited_graph()
    before = deepcopy(value)
    route = trace_source_fit_route(value, "save")
    assert route is not None
    assert route.composite_node_id == "composite"
    assert route.pad_node_id == "pad"
    assert route.decode_node_id == "decode"
    assert value == before


def test_direct_decode_route_does_not_claim_composited_source() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    route = trace_source_fit_route(graph(), "save")
    assert route is not None
    assert route.composite_node_id is None


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("destination", ["decode", 0]),
        ("destination", ["source", 0]),
        ("source", ["pad", 0]),
        ("mask", ["pad", 0]),
        ("mask", ["unrelated", 0]),
        ("x", 1),
        ("x", False),
        ("y", 1),
        ("resize_source", False),
        ("resize_source", 1),
    ],
)
def test_composite_cannot_change_canvas_placement_or_repaint_source(
    field: str, replacement: Any
) -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = composited_graph()
    value["composite"]["inputs"][field] = replacement
    assert trace_source_fit_route(value, "save") is None


def test_a_composite_without_the_pad_mask_is_refused() -> None:
    from local_lm.workflow_source_geometry import trace_source_fit_route

    value = composited_graph()
    del value["composite"]["inputs"]["mask"]
    assert trace_source_fit_route(value, "save") is None


def object_info(value: dict[str, Any] | None = None) -> dict[str, Any]:
    arities = {
        "InpaintModelConditioning": 3,
        "LoadImage": 2,
        "ImagePadForOutpaint": 2,
        "CheckpointLoaderSimple": 3,
        "CLIPTextEncode": 1,
        "VAEEncodeForInpaint": 1,
        "KSampler": 1,
        "VAEDecode": 1,
        "ImageCompositeMasked": 1,
        "SaveImage": 0,
    }
    return {
        node["class_type"]: {
            "python_module": "nodes",
            "input": {"required": {key: ["*", {}] for key in node["inputs"]}},
            "output": ["*"] * arities[node["class_type"]],
        }
        for node in (composited_graph() if value is None else value).values()
    }
