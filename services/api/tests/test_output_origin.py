"""Which part of a workflow produced a file, kept as the engine stated it.

The record's whole value is that it is first-party and unrecoverable: the engine
says which node wrote each file exactly once, in its response. These cover the
vocabulary itself, and the adapter that has to carry the key rather than drop it.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.adapters.contracts import media_event_contract_errors
from local_lm.db import SessionLocal
from local_lm.models import Job, Run
from local_lm.output_origin import MAX_IDENTIFIER, record_for, stated_origin, usable_identifier
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest


def _png(width: int, height: int) -> bytes:
    """A PNG an ordinary encoder wrote, so each asset has distinct real bytes."""
    pixels = bytes(
        (row * 7 + column * 13) % 256 for row in range(height) for column in range(width * 3)
    )
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, "PNG")
    return buffer.getvalue()


def test_an_engine_that_names_nothing_is_not_the_same_as_a_name_we_cannot_use() -> None:
    """Two absences that a reader must be able to tell apart.

    An engine that does not attribute its outputs will never attribute any of
    them, and that is worth knowing once. A key that arrived unusable is a fact
    about a single response. Collapsing them would make both uninformative.
    """
    silent = record_for(None, "mock")
    unusable = record_for(stated_origin(" \x01bad", "output", "images"), "comfyui")

    assert silent["state"] == "unattributed"
    assert silent["reason"] == "engine_does_not_attribute"
    assert unusable["state"] == "unattributed"
    assert unusable["reason"] == "node_id_unusable"
    assert silent["engine"] == "mock" and unusable["engine"] == "comfyui"


def test_a_named_node_is_recorded_with_what_kind_of_output_it_was() -> None:
    record = record_for(stated_origin("save", "output", "images"), "comfyui")

    assert record == {
        "v": 1,
        "state": "attributed",
        "engine": "comfyui",
        "node_id": "save",
        "output_type": "output",
        "collection": "images",
    }


def test_a_preview_is_attributed_too_and_is_not_judged_here() -> None:
    """`temp` is what a preview node writes and the engine does not keep.

    Recording it as an output of a named node is the whole point: this module
    tells files apart and says nothing about which one somebody asked for.
    """
    record = record_for(stated_origin("preview", "temp", "images"), "comfyui")

    assert record["state"] == "attributed"
    assert record["output_type"] == "temp"
    assert "requested" not in repr(record)


@pytest.mark.parametrize(
    "node_id",
    ["", " ", "x" * (MAX_IDENTIFIER + 1), "line\nbreak", 7, None, ["save"]],
)
def test_a_key_that_is_not_an_identifier_never_reaches_the_record(node_id: object) -> None:
    """The response is somebody else's data, so it is bounded before it is kept."""
    assert usable_identifier(node_id) is False
    assert stated_origin(node_id, "output", "images")["node_id"] is None
    assert record_for(stated_origin(node_id, "output", "images"), "comfyui")["state"] == (
        "unattributed"
    )


def test_a_type_outside_the_engine_s_own_vocabulary_is_dropped_not_carried() -> None:
    origin = stated_origin("save", "something-else", "sprites")

    assert origin["output_type"] is None
    assert origin["collection"] is None
    record = record_for(origin, "comfyui")
    assert record["state"] == "attributed", "the node is still named"
    assert record["output_type"] is None and record["collection"] is None


def test_an_adapter_cannot_hand_over_an_origin_it_invented() -> None:
    """The contract admits the engine's own three values, or nothing.

    A shape outside that means an adapter built the record itself rather than
    passing through what it was told, which is the tautology this whole area
    keeps having to refuse.
    """
    invented = GeneratedAsset(
        content=b"x",
        media_type="image/png",
        kind="image",
        name="a.png",
        origin={"node_id": "save", "raster_width": 1024},
    )
    honest = GeneratedAsset(
        content=b"x",
        media_type="image/png",
        kind="image",
        name="a.png",
        origin=stated_origin("save", "output", "images"),
    )

    refused = media_event_contract_errors(
        MediaEvent(type="complete", assets=[invented]), max_output_bytes=1024
    )
    accepted = media_event_contract_errors(
        MediaEvent(type="complete", assets=[honest]), max_output_bytes=1024
    )

    assert refused != [] and accepted == []


async def test_the_engine_says_which_node_wrote_each_file_and_we_keep_it(
    tmp_path: Path,
) -> None:
    """Driven through the real adapter against a real history shape.

    The outputs mapping is keyed by graph node. Two nodes here: the save branch
    the person asked for, and a preview branch whose file the engine marks
    `temp` and does not keep. Before this, both arrived as indistinguishable
    assets - same kind, same media type, nothing to tell them apart - and the
    key that separates them was dropped while reading the response.
    """
    prompt_id = "prompt-origin"
    output_root = tmp_path / "output"
    kept = output_root / "LMAtelier" / "kept.png"
    throwaway = output_root / "LMAtelier" / "throwaway.png"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"kept")
    throwaway.write_bytes(b"throwaway")

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "save": {
                                "images": [
                                    {
                                        "filename": kept.name,
                                        "subfolder": "LMAtelier",
                                        "type": "output",
                                    }
                                ]
                            },
                            "preview": {
                                "images": [
                                    {
                                        "filename": throwaway.name,
                                        "subfolder": "LMAtelier",
                                        "type": "temp",
                                    }
                                ]
                            },
                        }
                    }
                },
            )
        if request.url.path == "/view":
            filename = request.url.params["filename"]
            content = (output_root / "LMAtelier" / filename).read_bytes()
            return httpx.Response(200, content=content, headers={"content-type": "image/png"})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    adapter = ComfyUIAdapter("http://comfy.test", managed_output_root=output_root)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        outputs = await adapter._collect_outputs(prompt_id, "text_to_image")
    finally:
        await adapter.close()

    # Bound to the bytes each asset actually carries, so this cannot pass with
    # the two origins swapped onto the wrong files.
    origins = {asset.content: record_for(asset.origin, "comfyui") for asset in outputs}
    assert origins[b"kept"]["node_id"] == "save"
    assert origins[b"kept"]["output_type"] == "output"
    assert origins[b"throwaway"]["node_id"] == "preview"
    assert origins[b"throwaway"]["output_type"] == "temp"


async def test_the_record_reaches_the_run_beside_the_measurement(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real execution path, with the two files told apart.

    A helper returning the right record proves nothing about the run: what
    matters is that the orchestrator carries each asset's own origin to its own
    provenance entry. Two assets with different origins and different bytes make
    a swap fatal rather than invisible.
    """
    orchestrator = app.state.services.orchestrator
    chat = (await client.post("/api/chats", json={"title": "Named outputs"})).json()
    assets = [
        GeneratedAsset(
            content=_png(96, 64),
            kind="image",
            media_type="image/png",
            name="kept.png",
            origin=stated_origin("save", "output", "images"),
        ),
        GeneratedAsset(
            content=_png(32, 128),
            kind="image",
            media_type="image/png",
            name="throwaway.png",
            origin=stated_origin("preview", "temp", "images"),
        ),
        GeneratedAsset(
            content=_png(64, 64),
            kind="image",
            media_type="image/png",
            name="from-a-silent-engine.png",
        ),
    ]

    async def generate(request: MediaRequest) -> Any:
        yield MediaEvent(type="complete", assets=assets)

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)

    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(text="A picture of something neutral.", mode="image"),
                freeze_context=True,
                activate_branch=False,
            )
            run_id = accepted.run.id
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            job.status = "running"
            claim = JobClaim(token="named-output-attempt", attempt=1)
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()

        await orchestrator._execute_media(job_id, run_id, claim)

    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        outputs = run.provenance_json["outputs"]

    by_digest = {entry["sha256"]: entry["output_origin"] for entry in outputs}
    named = by_digest[hashlib.sha256(assets[0].content).hexdigest()]
    preview = by_digest[hashlib.sha256(assets[1].content).hexdigest()]
    silent = by_digest[hashlib.sha256(assets[2].content).hexdigest()]

    assert (named["state"], named["node_id"], named["output_type"]) == (
        "attributed",
        "save",
        "output",
    )
    assert (preview["state"], preview["node_id"], preview["output_type"]) == (
        "attributed",
        "preview",
        "temp",
    )
    assert silent["state"] == "unattributed"
    assert silent["reason"] == "engine_does_not_attribute"
    # The engine name is stamped by the orchestrator from what this execution
    # selected, not claimed by whatever answered.
    assert {record["engine"] for record in by_digest.values()} == {"mock"}
