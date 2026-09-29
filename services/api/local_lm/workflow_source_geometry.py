"""Identify source-fit pixel transport in one ComfyUI output path.

A route describes graph connections, not authorization or output dimensions.
The caller still needs trusted revision and source identity, a pixel plan,
runtime VAE compression, and verified output before claiming a fit was applied.
The accepted margins go into the one padding step that receives the source,
the step the Studio's Extend also writes. That step feeds an inpainting encode
or inpaint conditioning, and the sampler runs at full strength: written into
the graph, or mapped to a setting no other input reads, which the run pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Never, cast

from .outpaint_workflows import source_pad_node

MAX_OUTPAINT_PIXELS = 16384
# The inpainting encode leaves the new area empty, so only a full-strength
# sample fills it.
FULL_STRENGTH = 1
_VAE_OUTPUTS = {"CheckpointLoaderSimple": 2, "VAELoader": 0}


@dataclass(frozen=True, slots=True)
class SourceFitRoute:
    source_node_id: str
    source_parameter: Literal["input_image"]
    pad_node_id: str
    encode_node_id: str
    sampler_node_id: str
    decode_node_id: str
    save_node_id: str
    composite_node_id: str | None
    vae_node_id: str
    vae_output_index: int
    strength_parameter: str | None


class _UnboundRoute(ValueError):
    pass


def _refuse() -> Never:
    raise _UnboundRoute


def _identifier(value: object) -> str:
    if type(value) is not str or not value or len(value) > 256:
        _refuse()
    return value


def _node(graph: dict[str, object], node_id: str) -> dict[str, object]:
    value = graph.get(node_id)
    if type(value) is not dict:
        _refuse()
    return cast(dict[str, object], value)


def _inputs(graph: dict[str, object], node_id: str, kind: str) -> dict[str, object]:
    node = _node(graph, node_id)
    value = node.get("inputs")
    if node.get("class_type") != kind or type(value) is not dict or len(value) > 128:
        _refuse()
    if any(type(key) is not str for key in value):
        _refuse()
    return cast(dict[str, object], value)


def _edge(inputs: dict[str, object], key: str) -> tuple[str, int]:
    value = inputs.get(key)
    if type(value) is not list or len(value) != 2:
        _refuse()
    items = cast(list[object], value)
    node_id = _identifier(items[0])
    port = items[1]
    if type(port) is not int or port < 0:
        _refuse()
    return node_id, port


def _image_edge(inputs: dict[str, object], key: str) -> str:
    node_id, port = _edge(inputs, key)
    if port != 0:
        _refuse()
    return node_id


def _full_strength(graph: dict[str, object], value: object) -> str | None:
    """Name the setting a mapped strength reads, or accept a written full strength."""
    if type(value) in (int, float):
        if value != FULL_STRENGTH:
            _refuse()
        return None
    if type(value) is not str or not value.startswith("$" + "{") or not value.endswith("}"):
        _refuse()
    parameter = _identifier(value[2:-1])
    # Pinning a setting that another input also reads would change that input too.
    readers = 0
    for node in graph.values():
        inputs = cast(dict[str, object], node).get("inputs") if type(node) is dict else None
        if type(inputs) is dict:
            readers += sum(item == value for item in cast(dict[str, object], inputs).values())
    if readers != 1:
        _refuse()
    return parameter


def _zero(value: object) -> None:
    if type(value) is not int or value != 0:
        _refuse()


def trace_source_fit_route(api_graph: object, save_node_id: object) -> SourceFitRoute | None:
    """Trace one selected SaveImage path; return no route for unknown structure."""
    try:
        return _trace(api_graph, save_node_id)
    except _UnboundRoute:
        return None


def _trace(api_graph: object, save_node_id: object) -> SourceFitRoute:
    if type(api_graph) is not dict or not 1 <= len(api_graph) <= 512:
        _refuse()
    graph = cast(dict[str, object], api_graph)
    for key in graph:
        _identifier(key)
    save_id = _identifier(save_node_id)
    save = _inputs(graph, save_id, "SaveImage")
    decode_id = _image_edge(save, "images")
    composite_id: str | None = None
    composite: dict[str, object] | None = None
    if _node(graph, decode_id).get("class_type") == "ImageCompositeMasked":
        composite_id = decode_id
        composite = _inputs(graph, composite_id, "ImageCompositeMasked")
        _zero(composite.get("x"))
        _zero(composite.get("y"))
        if composite.get("resize_source") is not True:
            _refuse()
        decode_id = _image_edge(composite, "source")
    decode = _inputs(graph, decode_id, "VAEDecode")
    sampler_id = _image_edge(decode, "samples")
    sampler = _inputs(graph, sampler_id, "KSampler")
    strength_parameter = _full_strength(graph, sampler.get("denoise"))
    encode_id, latent_port = _edge(sampler, "latent_image")
    encode_kind = _node(graph, encode_id).get("class_type")
    if encode_kind == "VAEEncodeForInpaint" and latent_port == 0:
        encode = _inputs(graph, encode_id, "VAEEncodeForInpaint")
    elif encode_kind == "InpaintModelConditioning" and latent_port == 2:
        encode = _inputs(graph, encode_id, "InpaintModelConditioning")
        # The fill model sees the source only through this node's conditioning.
        if _edge(sampler, "positive") != (encode_id, 0) or _edge(sampler, "negative") != (
            encode_id,
            1,
        ):
            _refuse()
    else:
        _refuse()
    vae_id, vae_port = _edge(encode, "vae")
    if _edge(decode, "vae") != (vae_id, vae_port):
        _refuse()
    vae_kind = _node(graph, vae_id).get("class_type")
    if not isinstance(vae_kind, str) or _VAE_OUTPUTS.get(vae_kind) != vae_port:
        _refuse()
    _inputs(graph, vae_id, vae_kind)
    pad_id = _image_edge(encode, "pixels")
    if _edge(encode, "mask") != (pad_id, 1):
        _refuse()
    # The accepted margins are written into exactly this step, so it must be
    # the graph's only padding and take the source straight from LoadImage.
    if source_pad_node(graph) != pad_id:
        _refuse()
    pad = _inputs(graph, pad_id, "ImagePadForOutpaint")
    if composite is not None:
        # A workflow's own composite may only put the decode over the padded
        # source through the pad's mask, resizing a decode the VAE rounded.
        if _image_edge(composite, "destination") != pad_id:
            _refuse()
        if _edge(composite, "mask") != (pad_id, 1):
            _refuse()
    source_id = _image_edge(pad, "image")
    source = _inputs(graph, source_id, "LoadImage")
    if source.get("image") != "$" + "{input_image}":
        _refuse()
    return SourceFitRoute(
        source_node_id=source_id,
        source_parameter="input_image",
        pad_node_id=pad_id,
        encode_node_id=encode_id,
        sampler_node_id=sampler_id,
        decode_node_id=decode_id,
        save_node_id=save_id,
        composite_node_id=composite_id,
        vae_node_id=vae_id,
        vae_output_index=vae_port,
        strength_parameter=strength_parameter,
    )


@dataclass(frozen=True, slots=True)
class SourceFitPixels:
    """Integer transport, without source identity or output-size authority."""

    left: int
    top: int
    right: int
    bottom: int
    canvas_width: int
    canvas_height: int

    def margins(self) -> dict[str, int]:
        """The four sides, named as the source padding step names them."""
        return {"top": self.top, "right": self.right, "bottom": self.bottom, "left": self.left}
