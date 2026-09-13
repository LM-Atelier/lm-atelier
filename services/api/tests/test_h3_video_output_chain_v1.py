"""Recognizing the MiniMax H3 video output chain, and refusing everything near it.

The graphs here are constructed by hand in the shape the published H3 nodes and
their example workflows use: one sampler output decoded twice, once for frames
and once for sound, muxed and saved. Every refusal case changes exactly one hop,
because the property being protected is that nothing between the root and the
saved video can change the size the root declares.
"""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest

from local_lm.h3_video_output_chain_v1 import (
    MAX_GRAPH_NODES,
    H3VideoOutputChain,
    H3VideoOutputChainError,
    RootClass,
    recognize_h3_video_output_chain,
)


def _graph(root_class: str = "MiniMaxH3ImageToVideo", latent_output: int = 1) -> dict[str, Any]:
    root_inputs: dict[str, Any] = {"width": "${width}", "height": "${height}", "length": 124}
    if root_class != "EmptyMiniMaxH3LatentAV":
        root_inputs.update({"clip": ["2", 0], "vae": ["3", 0], "prompt": "a neutral test scene"})
    return {
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "text-encoder.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "video-vae.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "audio-vae.safetensors"}},
        "10": {"class_type": root_class, "inputs": root_inputs},
        "20": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["30", 0],
                "guider": ["31", 0],
                "sampler": ["32", 0],
                "sigmas": ["33", 0],
                "latent_image": ["10", latent_output],
            },
        },
        "30": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "31": {"class_type": "BasicGuider", "inputs": {}},
        "32": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "33": {"class_type": "BasicScheduler", "inputs": {"steps": 8}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["3", 0]}},
        "41": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["20", 0], "vae": ["4", 0]}},
        "50": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["40", 0], "audio": ["41", 0], "fps": 24},
        },
        "60": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["50", 0], "filename_prefix": "video/out"},
        },
    }


def _refused(graph: object) -> None:
    with pytest.raises(H3VideoOutputChainError):
        recognize_h3_video_output_chain(graph)


# ---- what is recognized --------------------------------------------------------


@pytest.mark.parametrize(
    ("root_class", "latent_output"),
    [
        ("MiniMaxH3ImageToVideo", 1),
        ("MiniMaxH3ReferenceToVideo", 1),
        ("EmptyMiniMaxH3LatentAV", 0),
    ],
)
def test_each_root_is_recognized_on_its_own_latent_output(
    root_class: str, latent_output: int
) -> None:
    chain = recognize_h3_video_output_chain(_graph(root_class, latent_output))

    assert chain == H3VideoOutputChain(
        root_id="10",
        root_class=cast(RootClass, root_class),
        sampler_id="20",
        video_decode_id="40",
        audio_decode_id="41",
        create_video_id="50",
        save_ids=("60",),
    )


def test_a_video_saved_without_sound_is_still_recognized() -> None:
    """Sound does not decide the frame size, so its absence changes nothing here."""

    graph = _graph()
    del graph["50"]["inputs"]["audio"]
    del graph["41"]

    chain = recognize_h3_video_output_chain(graph)

    assert chain.audio_decode_id is None


def test_a_classic_ksampler_is_accepted_in_place_of_the_custom_sampler() -> None:
    graph = _graph()
    graph["20"] = {"class_type": "KSampler", "inputs": {"latent_image": ["10", 1]}}

    assert recognize_h3_video_output_chain(graph).sampler_id == "20"


def test_two_saves_of_the_same_video_share_one_chain() -> None:
    graph = _graph()
    graph["61"] = {"class_type": "SaveVideo", "inputs": {"video": ["50", 0]}}

    assert recognize_h3_video_output_chain(graph).save_ids == ("60", "61")


# ---- what is refused: anything between the root and the saved video -----------


def test_an_upscale_between_decoding_and_the_video_is_refused() -> None:
    """The root would still say one size while a different size arrived."""

    graph = _graph()
    graph["45"] = {
        "class_type": "ImageScale",
        "inputs": {"image": ["40", 0], "width": 1920, "height": 1080},
    }
    graph["50"]["inputs"]["images"] = ["45", 0]

    _refused(graph)


def test_a_latent_resize_before_sampling_is_refused() -> None:
    graph = _graph()
    graph["15"] = {
        "class_type": "LatentUpscale",
        "inputs": {"samples": ["10", 1], "width": 1920, "height": 1080},
    }
    graph["20"]["inputs"]["latent_image"] = ["15", 0]

    _refused(graph)


def test_a_decoder_the_chain_does_not_know_is_refused_even_on_the_same_sampler() -> None:
    """Only the class check can refuse this: every other hop is wired correctly.

    The upscale case above also fails at the next hop, because an image scaler
    has no latent input. This one does not, so it is the case that shows the
    decoder itself is checked.
    """

    graph = _graph()
    graph["40"]["class_type"] = "VAEDecodeTiled"

    _refused(graph)


def test_a_latent_resize_carrying_the_declared_size_is_still_refused() -> None:
    """Only the root check can refuse this: its size inputs are the placeholders.

    A resize with literal sizes is also refused by the placeholder check, so it
    cannot show that the resize is rejected for not being a root.
    """

    graph = _graph()
    graph["15"] = {
        "class_type": "LatentUpscale",
        "inputs": {"samples": ["10", 1], "width": "${width}", "height": "${height}"},
    }
    graph["20"]["inputs"]["latent_image"] = ["15", 0]

    _refused(graph)


def test_a_sampler_fed_from_the_conditioning_output_is_refused() -> None:
    """Output 0 of the image-to-video node is conditioning, not the latent."""

    _refused(_graph("MiniMaxH3ImageToVideo", 0))


def test_the_empty_latent_root_on_a_slot_it_does_not_have_is_refused() -> None:
    _refused(_graph("EmptyMiniMaxH3LatentAV", 1))


def test_an_image_latent_root_is_not_an_h3_root() -> None:
    graph = _graph()
    graph["10"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": "${width}", "height": "${height}"},
    }
    graph["20"]["inputs"]["latent_image"] = ["10", 0]

    _refused(graph)


def test_sound_decoded_from_a_different_sampling_pass_is_refused() -> None:
    graph = _graph()
    graph["21"] = copy.deepcopy(graph["20"])
    graph["41"]["inputs"]["samples"] = ["21", 0]

    _refused(graph)


def test_frames_and_sound_wired_to_each_others_ports_are_refused() -> None:
    graph = _graph()
    graph["50"]["inputs"]["images"] = ["41", 0]
    graph["50"]["inputs"]["audio"] = ["40", 0]

    _refused(graph)


def test_saves_from_two_different_videos_are_refused() -> None:
    graph = _graph()
    graph["51"] = copy.deepcopy(graph["50"])
    graph["61"] = {"class_type": "SaveVideo", "inputs": {"video": ["51", 0]}}

    _refused(graph)


@pytest.mark.parametrize(
    "size",
    [
        {"width": 1344, "height": "${height}"},
        {"width": "${width}", "height": 768},
        {"width": "${height}", "height": "${width}"},
    ],
    ids=["literal-width", "literal-height", "swapped-placeholders"],
)
def test_a_root_whose_size_is_not_the_declared_inputs_is_refused(size: dict[str, Any]) -> None:
    graph = _graph()
    graph["10"]["inputs"].update(size)

    _refused(graph)


def test_an_image_graph_is_not_a_video_chain() -> None:
    _refused(
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": "${width}", "height": "${height}"},
            },
            "2": {"class_type": "KSampler", "inputs": {"latent_image": ["1", 0]}},
            "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0]}},
            "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        }
    )


# ---- what is refused: malformed or oversized graphs -----------------------------


@pytest.mark.parametrize(
    "graph",
    [
        None,
        [],
        {},
        {"1": "not a node"},
        {"1": {"class_type": "SaveVideo"}},
        {"1": {"class_type": "", "inputs": {}}},
    ],
    ids=["none", "list", "empty", "node-not-object", "no-inputs", "empty-class"],
)
def test_a_malformed_graph_is_refused(graph: object) -> None:
    _refused(graph)


def test_a_link_to_a_node_that_does_not_exist_is_refused() -> None:
    graph = _graph()
    graph["60"]["inputs"]["video"] = ["999", 0]

    _refused(graph)


def test_a_graph_past_the_node_limit_is_refused() -> None:
    graph = _graph()
    for index in range(MAX_GRAPH_NODES):
        graph[f"extra-{index}"] = {"class_type": "PrimitiveInt", "inputs": {}}

    _refused(graph)
