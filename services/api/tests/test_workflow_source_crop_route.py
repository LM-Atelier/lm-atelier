"""Trace the picture-to-picture path a cropped source can be uploaded to."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.workflow_source_geometry import trace_source_crop_route


def graph(*, separate_vae: bool = False) -> dict[str, Any]:
    vae = ["vae", 0] if separate_vae else ["checkpoint", 2]
    nodes: dict[str, Any] = {
        "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "${checkpoint}"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "Neutral geometric shapes", "clip": ["checkpoint", 1]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "Low contrast", "clip": ["checkpoint", 1]},
        },
        "encode": {"class_type": "VAEEncode", "inputs": {"pixels": ["source", 0], "vae": vae}},
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
                "denoise": "${denoise}",
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": vae}},
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": "fixture"},
        },
    }
    if separate_vae:
        nodes["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": "${vae}"}}
    return nodes


@pytest.mark.parametrize("separate_vae", [False, True])
def test_a_plain_encode_sample_and_decode_is_a_crop_route(separate_vae: bool) -> None:
    api_graph = graph(separate_vae=separate_vae)
    before = deepcopy(api_graph)

    route = trace_source_crop_route(api_graph, "save")

    assert route is not None
    assert (
        route.source_node_id,
        route.encode_node_id,
        route.sampler_node_id,
        route.decode_node_id,
    ) == (
        "source",
        "encode",
        "sample",
        "decode",
    )
    assert (route.vae_node_id, route.vae_output_index) == (
        ("vae", 0) if separate_vae else ("checkpoint", 2)
    )
    # The edit strength is the person's to choose: a crop is an ordinary edit.
    assert api_graph == before


def test_the_built_in_picture_to_picture_shape_is_a_crop_route() -> None:
    # The shape a standard whole-image edit is derived into: the empty latent
    # replaced by the encoded source, with the decode's own VAE.
    api_graph = graph()
    api_graph["lma-load-image"] = api_graph.pop("source")
    api_graph["lma-vae-encode"] = api_graph.pop("encode")
    api_graph["lma-vae-encode"]["inputs"]["pixels"] = ["lma-load-image", 0]
    api_graph["sample"]["inputs"]["latent_image"] = ["lma-vae-encode", 0]

    route = trace_source_crop_route(api_graph, "save")

    assert route is not None and route.source_node_id == "lma-load-image"


def _scaled_before_encode(api_graph: dict[str, Any]) -> None:
    api_graph["scale"] = {
        "class_type": "ImageScaleBy",
        "inputs": {"image": ["source", 0], "upscale_method": "lanczos", "scale_by": 0.5},
    }
    api_graph["encode"]["inputs"]["pixels"] = ["scale", 0]


def _scaled_before_save(api_graph: dict[str, Any]) -> None:
    api_graph["scale"] = {
        "class_type": "ImageScaleBy",
        "inputs": {"image": ["decode", 0], "upscale_method": "lanczos", "scale_by": 2.0},
    }
    api_graph["save"]["inputs"]["images"] = ["scale", 0]


def _empty_latent(api_graph: dict[str, Any]) -> None:
    api_graph["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    }
    api_graph["sample"]["inputs"]["latent_image"] = ["latent", 0]


def _other_decoder(api_graph: dict[str, Any]) -> None:
    api_graph["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": "${vae}"}}
    api_graph["decode"]["inputs"]["vae"] = ["vae", 0]


def _fixed_source(api_graph: dict[str, Any]) -> None:
    api_graph["source"]["inputs"]["image"] = "fixture.png"


def _outpainting(api_graph: dict[str, Any]) -> None:
    api_graph["encode"]["class_type"] = "VAEEncodeForInpaint"


@pytest.mark.parametrize(
    "change",
    [
        _scaled_before_encode,
        _scaled_before_save,
        _empty_latent,
        _other_decoder,
        _fixed_source,
        _outpainting,
    ],
    ids=[
        "the source is resized before it is encoded",
        "the result is resized before it is saved",
        "the sampler starts from an empty latent",
        "the result is decoded by another VAE",
        "the loaded picture is not the uploaded source",
        "the encode is an outpainting one",
    ],
)
def test_a_path_that_does_not_keep_the_sources_size_is_refused(change: Any) -> None:
    api_graph = graph()
    change(api_graph)

    assert trace_source_crop_route(api_graph, "save") is None


@pytest.mark.parametrize(
    ("api_graph", "save"),
    [
        (None, "save"),
        ({}, "save"),
        (graph(), "missing"),
        (graph(), ""),
        (graph(), 7),
    ],
)
def test_malformed_graph_or_save_identity_is_refused(api_graph: Any, save: Any) -> None:
    assert trace_source_crop_route(api_graph, save) is None
