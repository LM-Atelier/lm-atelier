"""An uploaded video is described over HTTP with what the video utilities may do with it.

This case reaches the application only through its HTTP surface and imports
nothing of the probe itself, so it states the same contract against any build:
one that cannot describe a stored video fails it on its answer.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from httpx2 import AsyncClient


async def test_an_uploaded_video_is_described_with_what_the_utilities_may_do(
    client: AsyncClient, tmp_path: Path
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "this case needs a real ffmpeg on PATH, as hosted CI provides"
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        timeout=60,
    )
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")}
    )
    assert uploaded.status_code == 201, uploaded.text
    artifact_id = uploaded.json()["id"]

    response = await client.get(f"/api/artifacts/{artifact_id}/video-probe")

    assert response.status_code == 200, response.text
    probe = response.json()
    assert (probe["artifact_id"], probe["container"]) == (artifact_id, "mp4")
    video = probe["video"]
    assert (video["codec"], video["width"], video["height"]) == ("h264", 64, 48)
    assert (video["frame_rate"], video["frame_rate_form"]) == ("10/1", "constant")
    assert (probe["can_save_frame"], probe["can_trim"]) == (True, True)
    assert probe["limits"] == []
    assert (probe["tool"]["name"], probe["tool"]["origin"]) == ("ffprobe", "system")
