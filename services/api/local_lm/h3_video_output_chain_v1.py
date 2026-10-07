"""Recognize the exact output chain of a MiniMax H3 video graph, or refuse.

The image size prover walks a save, a decode, a sampler and a latent root, every
link on output slot 0. An H3 video graph shares none of those hops, so that walk
refuses it before any size arithmetic is reached. This module recognizes H3's
own chain so a later proof can bind to it:

    SaveVideo.video                    <- CreateVideo, output 0
    CreateVideo.images                 <- VAEDecode, output 0
    CreateVideo.audio (when present)   <- VAEDecodeAudio, output 0
    VAEDecode.samples                  <- the sampler, output 0
    VAEDecodeAudio.samples             <- the SAME sampler, output 0
    sampler.latent_image               <- the root, on the root's latent output

One sampler output carries the packed video and audio latent, and it fans out to
one decoder for frames and one for sound.

WHY EXACT HOPS AND NOTHING BETWEEN THEM. The root builds a latent of
``height // 16`` by ``width // 16`` from its own width and height inputs, so the
frames that arrive are exactly the size the root declares - unless something
sits between the root and the saved video. An upscale after decoding, a crop, or
a latent resize before sampling would each deliver a different size while the
root still declared the original. Requiring each link to come straight from the
expected node is what makes the root's inputs a statement about the output.

WHY THE LATENT OUTPUT SLOT IS PART OF THE ROOT. The image-to-video and
reference-to-video nodes return conditioning on output 0 and the latent on
output 1; the empty-latent node returns only a latent, on output 0. A sampler
fed from the conditioning slot is not this chain, however the node is named.

Recognition is structural and says nothing about what is installed or which
runtime runs it. Node names in a graph are a claim about the graph, not proof
that a runtime providing those names will deliver these pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Never, cast

H3_VIDEO_OUTPUT_CHAIN_VERSION: Final = 1

MAX_GRAPH_NODES: Final = 512
MAX_NODE_INPUTS: Final = 128
MAX_IDENTIFIER_LENGTH: Final = 256
MAX_SAVE_NODES: Final = 32

RootClass = Literal["MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "EmptyMiniMaxH3LatentAV"]

#: The output slot each root delivers its latent on.
ROOT_LATENT_OUTPUT: Final[dict[str, int]] = {
    "MiniMaxH3ImageToVideo": 1,
    "MiniMaxH3ReferenceToVideo": 1,
    "EmptyMiniMaxH3LatentAV": 0,
}

#: Samplers that take the latent on ``latent_image`` and return it on output 0.
SAMPLERS: Final = frozenset({"SamplerCustomAdvanced", "KSampler"})

WIDTH_PLACEHOLDER: Final = "${width}"
HEIGHT_PLACEHOLDER: Final = "${height}"


class H3VideoOutputChainError(ValueError):
    """The graph is not exactly the recognized H3 video output chain."""

    def __init__(self) -> None:
        super().__init__("workflow graph is not a recognized H3 video output chain")


@dataclass(frozen=True, slots=True)
class H3VideoOutputChain:
    """The node ids of one recognized chain, in the order a frame travels back through them."""

    root_id: str
    root_class: RootClass
    sampler_id: str
    video_decode_id: str
    audio_decode_id: str | None
    create_video_id: str
    save_ids: tuple[str, ...]


def recognize_h3_video_output_chain(graph: object) -> H3VideoOutputChain:
    """The single H3 chain every saved video in ``graph`` comes from.

    ``graph`` is an API-format graph: node id to ``{"class_type", "inputs"}``,
    with links written as ``[source_id, output_index]``. Every saved video must
    trace back through the same create, decode, sampler and root nodes, and the
    root's width and height must be the workflow's declared placeholders.
    """

    nodes = _nodes(graph)
    save_ids = tuple(sorted(node_id for node_id, (kind, _) in nodes.items() if kind == "SaveVideo"))
    if not save_ids or len(save_ids) > MAX_SAVE_NODES:
        _refuse()

    creates: set[str] = set()
    for save_id in save_ids:
        creates.add(_source(nodes, nodes[save_id][1].get("video"), frozenset({"CreateVideo"}), 0))
    if len(creates) != 1:
        _refuse()
    create_id = next(iter(creates))
    create_inputs = nodes[create_id][1]

    video_decode_id = _source(nodes, create_inputs.get("images"), frozenset({"VAEDecode"}), 0)
    sampler_id = _source(nodes, nodes[video_decode_id][1].get("samples"), SAMPLERS, 0)

    audio_decode_id: str | None = None
    if "audio" in create_inputs:
        audio_decode_id = _source(
            nodes, create_inputs.get("audio"), frozenset({"VAEDecodeAudio"}), 0
        )
        audio_sampler_id = _source(nodes, nodes[audio_decode_id][1].get("samples"), SAMPLERS, 0)
        # Sound decoded from a different sampling pass is a different chain,
        # even when the frames are exactly right.
        if audio_sampler_id != sampler_id:
            _refuse()

    latent_link = nodes[sampler_id][1].get("latent_image")
    root_id = _root(nodes, latent_link)
    root_class = cast(RootClass, nodes[root_id][0])
    root_inputs = nodes[root_id][1]
    if (
        root_inputs.get("width") != WIDTH_PLACEHOLDER
        or root_inputs.get("height") != HEIGHT_PLACEHOLDER
    ):
        _refuse()

    return H3VideoOutputChain(
        root_id=root_id,
        root_class=root_class,
        sampler_id=sampler_id,
        video_decode_id=video_decode_id,
        audio_decode_id=audio_decode_id,
        create_video_id=create_id,
        save_ids=save_ids,
    )


def _nodes(graph: object) -> dict[str, tuple[str, dict[str, object]]]:
    if type(graph) is not dict or not graph or len(graph) > MAX_GRAPH_NODES:
        _refuse()
    nodes: dict[str, tuple[str, dict[str, object]]] = {}
    for raw_id, raw_node in cast(dict[object, object], graph).items():
        node_id = _identifier(raw_id)
        if type(raw_node) is not dict:
            _refuse()
        node = cast(dict[str, object], raw_node)
        kind = _identifier(node.get("class_type"))
        inputs = node.get("inputs")
        if type(inputs) is not dict or len(inputs) > MAX_NODE_INPUTS:
            _refuse()
        nodes[node_id] = (kind, cast(dict[str, object], inputs))
    return nodes


def _source(
    nodes: dict[str, tuple[str, dict[str, object]]],
    link: object,
    expected: frozenset[str],
    output: int,
) -> str:
    source_id, source_output = _link(nodes, link)
    if source_output != output or nodes[source_id][0] not in expected:
        _refuse()
    return source_id


def _root(nodes: dict[str, tuple[str, dict[str, object]]], link: object) -> str:
    source_id, source_output = _link(nodes, link)
    expected_output = ROOT_LATENT_OUTPUT.get(nodes[source_id][0])
    if expected_output is None or source_output != expected_output:
        _refuse()
    return source_id


def _link(nodes: dict[str, tuple[str, dict[str, object]]], link: object) -> tuple[str, int]:
    if type(link) is not list or len(link) != 2:
        _refuse()
    source_id = _identifier(link[0])
    output = link[1]
    if type(output) is not int or output < 0 or source_id not in nodes:
        _refuse()
    return source_id, output


def _identifier(value: object) -> str:
    if type(value) is not str or not value or len(value) > MAX_IDENTIFIER_LENGTH:
        _refuse()
    return value


def _refuse() -> Never:
    raise H3VideoOutputChainError()
