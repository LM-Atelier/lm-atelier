"""A produced video is measured where it is stored, through the real media execution.

The measurer's own tests show what it answers. This shows the orchestrator
actually asks it for a video - and still asks the picture path for a picture in
the same run - because a measurer that nothing dispatches to records nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_output_measurement_seam import _png, _run_media
from test_video_output_measurement import _neutral_video

from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import GeneratedAsset
from local_lm.output_measurement import Budget
from local_lm.video_output_measurement import measure_video_output

pytestmark = pytest.mark.asyncio


async def test_a_produced_video_is_recorded_at_the_size_of_its_decoded_frame(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = _neutral_video(tmp_path, 96, 64)
    picture = _png(32, 128)

    outputs = await _run_media(
        app,
        client,
        monkeypatch,
        [
            GeneratedAsset(content=video, kind="video", media_type="video/mp4", name="clip.mp4"),
            GeneratedAsset(content=picture, kind="image", media_type="image/png", name="still.png"),
        ],
    )

    # A video also stores a poster frame for the browser. That poster is not a
    # produced output and carries no measurement, so records are looked up by
    # the digest of the bytes the run produced.
    records = {
        artifact.sha256: artifact.metadata_json.get("output_measurement") for artifact in outputs
    }
    clip = records[hashlib.sha256(video).hexdigest()]
    still = records[hashlib.sha256(picture).hexdigest()]
    assert clip is not None and still is not None
    assert clip["state"] == "measured"
    assert clip["method"] == "video_first_frame_png"
    assert (clip["raster_width"], clip["raster_height"]) == (96, 64)
    assert still["method"] == "png_idat_consumed"
    assert (still["raster_width"], still["raster_height"]) == (32, 128)


async def test_every_video_in_a_run_draws_on_one_budget(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dispatch passes the run's single budget, not a fresh one per video."""

    budgets: list[int] = []
    measure = measure_video_output

    async def recording(content: bytes, budget: Budget) -> dict[str, object]:
        budgets.append(id(budget))
        return await measure(content, budget)

    monkeypatch.setattr(orchestrator_module, "measure_video_output", recording)
    first = _neutral_video(tmp_path, 96, 64)
    second = _neutral_video(tmp_path, 48, 128)

    await _run_media(
        app,
        client,
        monkeypatch,
        [
            GeneratedAsset(content=first, kind="video", media_type="video/mp4", name="a.mp4"),
            GeneratedAsset(content=second, kind="video", media_type="video/mp4", name="b.mp4"),
        ],
    )

    assert len(budgets) == 2
    assert budgets[0] == budgets[1]
