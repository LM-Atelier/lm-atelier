"""What a generation produced, recorded against the artifact it produced.

The measurement module is a pure function over bytes and says nothing about a
run. These are about the one place that calls it: that every produced file gets
its own measurement, that the measurement lands on the row those exact bytes
address, and that a file we cannot measure is recorded as unmeasured rather than
stopping a run that otherwise succeeded.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, Run
from local_lm.output_measurement import Budget
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest


def _png(width: int, height: int) -> bytes:
    """A PNG an ordinary encoder wrote, so the record is about real bytes."""
    pixels = bytes(
        (row * 7 + column * 13) % 256 for row in range(height) for column in range(width * 3)
    )
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, "PNG")
    return buffer.getvalue()


async def _run_media(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    assets: list[GeneratedAsset],
) -> list[Artifact]:
    """Drive one real media execution that produces these assets."""
    orchestrator = app.state.services.orchestrator
    chat = (await client.post("/api/chats", json={"title": "Measured output"})).json()

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
            claim = JobClaim(token="measured-output-attempt", attempt=1)
            job.claim_owner = claim.token
            # Ownership is asserted at the database against BOTH the owner and
            # the captured attempt, so the row has to carry each of them.
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()

        await orchestrator._execute_media(job_id, run_id, claim)

    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None, "the run must survive its own execution"
        return [
            artifact
            for artifact in session.scalars(select(Artifact))
            if artifact.metadata_json.get("run_id") == run_id
        ]


async def test_a_produced_picture_is_recorded_at_the_size_it_actually_is(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = await _run_media(
        app,
        client,
        monkeypatch,
        [
            GeneratedAsset(
                content=_png(96, 64), kind="image", media_type="image/png", name="neutral.png"
            )
        ],
    )

    assert len(outputs) == 1
    record = outputs[0].metadata_json["output_measurement"]
    assert record["state"] == "measured"
    assert (record["raster_width"], record["raster_height"]) == (96, 64)


async def test_each_output_carries_its_own_measurement(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attribution, which is the whole risk at this seam.

    A run can store several files of different sizes. One measurement applied to
    all of them, or the right measurements paired with the wrong rows, is the
    false success this feature exists to prevent - and it would look correct in
    any test that produced a single output.

    Each answer is bound to the digest of the bytes it is about, never to the
    order they came back in or to a set of dimensions collected from all of
    them. A set would accept the two answers swapped, which is precisely the
    fault worth catching.
    """
    wide, tall = _png(96, 64), _png(32, 128)
    outputs = await _run_media(
        app,
        client,
        monkeypatch,
        [
            GeneratedAsset(content=wide, kind="image", media_type="image/png", name="wide.png"),
            GeneratedAsset(content=tall, kind="image", media_type="image/png", name="tall.png"),
        ],
    )

    measured = {
        artifact.sha256: (
            artifact.metadata_json["output_measurement"]["raster_width"],
            artifact.metadata_json["output_measurement"]["raster_height"],
        )
        for artifact in outputs
    }
    assert measured == {
        hashlib.sha256(wide).hexdigest(): (96, 64),
        hashlib.sha256(tall).hexdigest(): (32, 128),
    }


async def test_a_file_we_cannot_measure_is_recorded_rather_than_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measuring is evidence, not a gate.

    Nothing here may refuse a run. A produced file that cannot be measured -
    truncated, or a container this version has no method for - is recorded as
    unmeasured, with `about` saying whether that is a fact about the file or a
    limit of ours, and the run completes exactly as it would have before.
    """
    whole = _png(96, 64)
    outputs = await _run_media(
        app,
        client,
        monkeypatch,
        [
            GeneratedAsset(
                content=whole[: len(whole) // 2],
                kind="image",
                media_type="image/png",
                name="cut-short.png",
            ),
            GeneratedAsset(
                content=b"\xff\xd8\xff\xe0" + bytes(64),
                kind="image",
                media_type="image/jpeg",
                name="another-container.jpg",
            ),
        ],
    )

    assert len(outputs) == 2, "both files were still stored"
    records = {
        artifact.metadata_json["output_measurement"]["about"]: artifact.metadata_json[
            "output_measurement"
        ]
        for artifact in outputs
    }
    assert set(records) == {"file", "scope"}
    assert records["file"]["reason"] == "container_incomplete"
    assert records["scope"]["reason"] == "unsupported_container"
    assert all(record["state"] == "unmeasured" for record in records.values())
    assert all("raster_width" not in record for record in records.values())


async def test_a_second_run_over_identical_bytes_cannot_erase_a_measurement(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The consequence of storing by content rather than by run.

    Identical output lands on the identical row, so a later generation writes
    where an earlier one already did. Its budget is shared across everything it
    produced and can be spent, and a starved look says nothing about the file -
    so if it overwrote, a busy run would quietly turn a measurement we hold
    into "we could not look".
    """
    content = _png(96, 64)
    first = await _run_media(
        app,
        client,
        monkeypatch,
        [GeneratedAsset(content=content, kind="image", media_type="image/png", name="same.png")],
    )
    assert len(first) == 1
    artifact_id = first[0].id
    assert first[0].metadata_json["output_measurement"]["state"] == "measured"

    # The second generation has nothing left to spend by the time it looks.
    monkeypatch.setattr(orchestrator_module, "Budget", lambda: Budget(steps=0))
    await _run_media(
        app,
        client,
        monkeypatch,
        [GeneratedAsset(content=content, kind="image", media_type="image/png", name="same.png")],
    )

    with SessionLocal() as session:
        stored = session.get(Artifact, artifact_id)
        assert stored is not None, "the content-addressed row is the same row"
        record = stored.metadata_json["output_measurement"]

    assert record["state"] == "measured", "our exhausted budget is not evidence about the file"
    assert (record["raster_width"], record["raster_height"]) == (96, 64)


async def test_one_budget_is_shared_across_everything_a_run_produced(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is the generation's, not each file's.

    A file can be built to be expensive to read, so a ceiling applied per file
    multiplies by however many files a run produced - which is the whole reason
    one Budget is created for the generation and handed to every measurement.
    Nothing about a single output can show that, so this uses two: each fits on
    its own, the pair does not, and the second is refused by our own ceiling.

    The two differ in bytes as well as in shape. Identical content would share
    one content-addressed row and hide the second answer entirely.
    """
    first, second = _png(96, 64), _png(64, 96)
    # 64 rows of 1 + 96*3 is 18,496 bytes; 96 rows of 1 + 64*3 is 18,528. Room
    # for either alone, and not for both.
    monkeypatch.setattr(orchestrator_module, "Budget", lambda: Budget(decoded=20_000))

    outputs = await _run_media(
        app,
        client,
        monkeypatch,
        [
            GeneratedAsset(content=first, kind="image", media_type="image/png", name="first.png"),
            GeneratedAsset(content=second, kind="image", media_type="image/png", name="second.png"),
        ],
    )

    records = {
        artifact.sha256: artifact.metadata_json["output_measurement"] for artifact in outputs
    }
    measured = records[hashlib.sha256(first).hexdigest()]
    refused = records[hashlib.sha256(second).hexdigest()]

    assert (measured["raster_width"], measured["raster_height"]) == (96, 64)
    assert refused["state"] == "unmeasured"
    assert (refused["about"], refused["reason"]) == ("budget", "over_decode_budget")
