"""Proving the output size of a MiniMax H3 video revision, and refusing near misses.

The image cases live beside the image prover. These are the video half: the
operation decides which output chain is required, the stored digest binds the
operation, H3's own size grid bounds what may be offered, and an executed graph
has to carry the whole recognized chain rather than the flat id lists.

Graphs are built by hand in the shape of the public H3 nodes, with neutral
values only.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.h3_video_output_chain_v1 import recognize_h3_video_output_chain
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import WorkflowRevision
from local_lm.workflow_output_geometry import (
    WorkflowOutputGeometryResult,
    executed_graph_carries_the_proof,
    prove_workflow_output_geometry,
    resolve_workflow_output_geometry,
    workflow_output_geometry_payload,
)


def _h3_graph() -> dict[str, Any]:
    return {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "encoder.safetensors"}},
        "video-vae": {"class_type": "VAELoader", "inputs": {"vae_name": "video.safetensors"}},
        "audio-vae": {"class_type": "VAELoader", "inputs": {"vae_name": "audio.safetensors"}},
        "root": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["video-vae", 0],
                "width": "${width}",
                "height": "${height}",
                "length": 124,
            },
        },
        "sampler": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"latent_image": ["root", 1]},
        },
        "frames": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["video-vae", 0]},
        },
        "sound": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sampler", 0], "vae": ["audio-vae", 0]},
        },
        "mux": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["frames", 0], "audio": ["sound", 0], "fps": 24},
        },
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["mux", 0]}},
    }


def _image_graph() -> dict[str, Any]:
    return {
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": "${width}", "height": "${height}", "batch_size": 1},
        },
        "sampler": {"class_type": "KSampler", "inputs": {"latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}},
    }


def _dimension(default: int, multiple: int = 32) -> dict[str, object]:
    return {
        "type": "integer",
        "default": default,
        "minimum": 256,
        "maximum": 1920,
        "multipleOf": multiple,
    }


def _schema(multiple: int = 32) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "width": _dimension(1344, multiple),
            "height": _dimension(768, multiple),
        },
    }


def _arguments(
    operation: str = "text_to_video",
    graph: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
    digest_operation: str | None = None,
) -> dict[str, Any]:
    api_graph = graph if graph is not None else _h3_graph()
    input_schema = schema if schema is not None else _schema()
    return {
        "workflow_id": "workflow-video",
        "revision_id": "revision-video",
        "operation": operation,
        "engine": "comfyui",
        "api_graph": api_graph,
        "input_schema": input_schema,
        "dependencies": {},
        "artifact_sha256": workflow_artifact_contract(
            operation=digest_operation or operation,
            engine="comfyui",
            api_graph=api_graph,
            input_schema=input_schema,
            dependencies={},
        ),
        "trusted": True,
    }


def _prove(**overrides: Any) -> WorkflowOutputGeometryResult:
    return prove_workflow_output_geometry(**_arguments(**overrides))


# ---- what is proven -------------------------------------------------------------


@pytest.mark.parametrize("operation", ["text_to_video", "image_to_video"])
def test_an_h3_revision_is_proven_on_its_own_chain(operation: str) -> None:
    result = _prove(operation=operation)

    assert result.available is True
    proof = result.proof
    assert proof is not None
    assert proof.operation == operation
    assert proof.video_chain == recognize_h3_video_output_chain(_h3_graph())
    assert proof.latent_node_id == "root"
    assert proof.sampler_node_ids == ("sampler",)
    assert proof.decode_node_ids == ("frames", "sound")
    assert proof.save_node_ids == ("save",)
    # The size inputs are the root's, which is what makes them a statement
    # about the frames that arrive.
    assert (proof.width.node_id, proof.height.node_id) == ("root", "root")
    assert proof.capability.allowed_modes == ("video",)

    payload = workflow_output_geometry_payload(result)
    assert payload["operation"] == operation
    assert payload["decode_node_ids"] == ["frames", "sound"]
    # Every ratio the 32 grid can express exactly inside these bounds.
    assert payload["preset_ids"] == ["16:9", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16"]


def test_a_picture_resized_before_it_reaches_the_root_does_not_block_the_proof() -> None:
    """The root resizes its start frame to the declared size, so the source cannot
    change what arrives. Only nodes between the root and the saved video can."""

    graph = _h3_graph()
    graph["picture"] = {"class_type": "LoadImage", "inputs": {"image": "neutral.png"}}
    graph["scaled"] = {
        "class_type": "ImageScale",
        "inputs": {"image": ["picture", 0], "width": 640, "height": 640},
    }
    graph["root"]["inputs"]["start_image"] = ["scaled", 0]

    assert _prove(operation="image_to_video", graph=graph).available is True


def test_resolving_a_video_request_returns_the_pixels_the_root_will_make() -> None:
    result = _prove()

    resolution = resolve_workflow_output_geometry(
        result, {"mode": "video", "size_mode": "preset", "preset_id": "9:16"}
    )

    assert resolution is not None
    assert resolution.operation == "text_to_video"
    # Worked by hand: 9:16 on a 32 grid is 288c by 512c, and c = 3 gives the area
    # nearest the 1344 x 768 default. That default is not itself 9:16 or 16:9,
    # which is why the preset is a different pair.
    assert (resolution.geometry.mode, resolution.geometry.width) == ("video", 864)
    assert resolution.geometry.height == 1536
    # A picture request is not what this revision makes.
    assert (
        resolve_workflow_output_geometry(
            result, {"mode": "image", "size_mode": "preset", "preset_id": "9:16"}
        )
        is None
    )


# ---- the operation decides the chain --------------------------------------------


def test_an_h3_graph_stored_as_a_picture_workflow_is_not_proven() -> None:
    assert _prove(operation="text_to_image").available is False


@pytest.mark.parametrize("operation", ["text_to_video", "image_to_video"])
def test_a_picture_graph_stored_as_a_video_workflow_is_not_proven(operation: str) -> None:
    schema = _schema(multiple=64)

    assert _prove(operation=operation, graph=_image_graph(), schema=schema).available is False


def test_the_same_picture_graph_is_still_proven_as_a_picture() -> None:
    """The control for the case above: the refusal is the operation's, not the graph's."""

    schema = _schema(multiple=64)

    assert _prove(operation="text_to_image", graph=_image_graph(), schema=schema).available


def test_a_digest_taken_for_another_operation_does_not_prove_a_video() -> None:
    """Everything else here is a valid H3 revision; only the digest's operation differs."""

    assert _prove(digest_operation="text_to_image").available is False
    assert _prove(operation="image_to_video", digest_operation="text_to_video").available is False


@pytest.mark.parametrize("operation", ["image_to_image", "video_to_video", "", "TEXT_TO_VIDEO"])
def test_an_operation_with_no_proof_is_refused(operation: str) -> None:
    """The digest is taken for the same operation, so only the operation can refuse."""

    assert _prove(operation=operation).available is False


# ---- H3's size grid --------------------------------------------------------------


@pytest.mark.parametrize(("multiple", "available"), [(16, False), (32, True), (64, True)])
def test_only_sizes_on_the_roots_own_grid_are_offered(multiple: int, available: bool) -> None:
    """Off the 32 grid the root would round a requested size down to another one."""

    assert _prove(schema=_schema(multiple)).available is available


def test_a_width_off_the_grid_is_refused_even_when_the_height_is_on_it() -> None:
    schema = _schema()
    schema["properties"]["width"]["multipleOf"] = 16
    schema["properties"]["width"]["default"] = 1344

    assert _prove(schema=schema).available is False


# ---- the schema around the size ---------------------------------------------------


def _with_length_contract(schema: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    schema["properties"]["frames"] = {
        "type": "integer",
        "default": 124,
        "minimum": 124,
        "maximum": 362,
    }
    schema["properties"]["fps"] = {"type": "number", "const": 24}
    schema["x-lm-atelier-video-length"] = {
        "version": 1,
        "frames_parameter": "frames",
        "fps_parameter": "fps",
        "fps_numerator": 24,
        "fps_denominator": 1,
        "frame_alignment": 17,
        "frame_offset": 5,
    }
    return schema


def test_a_video_length_contract_beside_the_size_does_not_block_the_proof() -> None:
    assert _prove(schema=_with_length_contract(_schema())).available is True


def test_a_malformed_video_length_contract_is_still_refused() -> None:
    schema = _with_length_contract(_schema())
    schema["x-lm-atelier-video-length"]["frame_offset"] = 17

    assert _prove(schema=schema).available is False


def test_a_picture_schema_carrying_a_video_length_contract_is_refused() -> None:
    """A picture has no frames to bind. Admitting the contract for videos does not
    admit it for pictures."""

    schema = _with_length_contract(_schema(multiple=64))

    assert _prove(operation="text_to_image", graph=_image_graph(), schema=schema).available is False


def test_any_other_top_level_keyword_is_still_refused_for_a_video() -> None:
    schema = _schema()
    schema["allOf"] = [{"properties": {"width": {"maximum": 512}}}]

    assert _prove(schema=schema).available is False


# ---- the executed graph ------------------------------------------------------------


def test_an_executed_video_graph_carries_the_proof_when_its_chain_is_unchanged() -> None:
    proof = _prove().proof

    assert executed_graph_carries_the_proof(proof, _h3_graph()) is True


def test_a_renamed_mux_breaks_the_proof_though_every_listed_id_is_unchanged() -> None:
    """The flat lists hold the root, sampler, decoders and saves, but not the mux.

    Only comparing the whole chain can tell this run from the proven one.
    """

    proof = _prove().proof
    graph = _h3_graph()
    graph["remux"] = graph.pop("mux")
    graph["save"]["inputs"]["video"] = ["remux", 0]

    assert proof is not None
    assert executed_graph_carries_the_proof(proof, graph) is False


def test_an_upscale_added_before_the_mux_breaks_the_proof() -> None:
    proof = _prove().proof
    graph = _h3_graph()
    graph["upscale"] = {
        "class_type": "ImageScale",
        "inputs": {"image": ["frames", 0], "width": 1920, "height": 1080},
    }
    graph["mux"]["inputs"]["images"] = ["upscale", 0]

    assert executed_graph_carries_the_proof(proof, graph) is False


def test_proofs_of_one_kind_are_not_carried_by_a_graph_of_the_other() -> None:
    video = _prove().proof
    picture = _prove(operation="text_to_image", graph=_image_graph(), schema=_schema(64)).proof

    assert executed_graph_carries_the_proof(video, _image_graph()) is False
    assert executed_graph_carries_the_proof(picture, _h3_graph()) is False


# ---- through the application -------------------------------------------------------


async def test_a_stored_h3_revision_reports_its_video_geometry(client: AsyncClient) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral video geometry",
            "operation": "text_to_video",
            "engine": "comfyui",
            "api_graph": _h3_graph(),
            "input_schema": _schema(),
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["revisions"][0]["id"]
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()

    capability = await client.get(f"/api/workflow-revisions/{revision_id}/output-geometry")
    assert capability.status_code == 200, capability.text
    body = capability.json()
    assert body["available"] is True
    assert body["operation"] == "text_to_video"
    assert body["capability"]["allowed_modes"] == ["video"]

    resolved = await client.post(
        f"/api/workflow-revisions/{revision_id}/output-geometry/resolve",
        json={"mode": "video", "size_mode": "preset", "preset_id": "16:9"},
    )
    assert resolved.status_code == 200, resolved.text
    assert (resolved.json()["mode"], resolved.json()["width"]) == ("video", 1536)
    assert resolved.json()["height"] == 864
