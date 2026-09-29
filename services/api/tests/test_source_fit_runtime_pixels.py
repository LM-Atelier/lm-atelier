"""Opt-in real ComfyUI pad/composite pixel controls over a neutral CPU graph.

The paint is a constant image, not a sampler or VAE. This verifies core image
transforms and saved pixels, not generated outpainting or the full fit feature.
"""

from __future__ import annotations

import io
import os
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from run_waits import wait_for_terminal_status

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Run
from local_lm.source_fit_image import prepare_source_fit_image


@pytest.fixture
def settings(settings: Settings) -> Settings:
    executable = os.environ.get("LM_ATELIER_TEST_COMFY_EXECUTABLE")
    directory = os.environ.get("LM_ATELIER_TEST_COMFY_DIRECTORY")
    if not executable or not directory:
        pytest.skip("An isolated CPU ComfyUI runtime is required")
    settings.comfy_executable = Path(executable)
    settings.comfy_directory = Path(directory)
    assert settings.comfy_executable.is_file()
    assert (settings.comfy_directory / "main.py").is_file()
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    settings.comfy_url = f"http://127.0.0.1:{port}"
    settings.media_engine = "comfyui"
    settings.worker_startup_seconds = 120
    return settings


def source_png(orientation: int) -> tuple[bytes, bytes, tuple[int, int]]:
    raw = bytes(
        component
        for index in range(16 * 24)
        for component in (index % 256, (index * 37) % 256, 255 - index % 256)
    )
    with Image.frombytes("RGB", (16, 24), raw) as source:
        exif = Image.Exif()
        exif[274] = orientation
        output = io.BytesIO()
        source.save(output, format="PNG", exif=exif)
    if orientation == 1:
        return output.getvalue(), raw, (16, 24)
    # Enumerate clockwise rotation independently of both runtimes' EXIF helper.
    oriented = b"".join(
        raw[((23 - x) * 16 + y) * 3 : ((23 - x) * 16 + y) * 3 + 3]
        for y in range(16)
        for x in range(24)
    )
    return output.getvalue(), oriented, (24, 16)


def assert_paint(content: bytes) -> None:
    # Actual bilinear resampling uses floating point, then SaveImage truncates
    # to uint8. Only this synthetic paint permits a one-level downward error.
    # The preserved source rectangle is always compared byte-for-byte.
    target = (51, 102, 204)
    assert all(
        target[index % 3] - 1 <= value <= target[index % 3] for index, value in enumerate(content)
    )


@pytest.mark.parametrize("orientation,inverted", [(1, False), (6, False), (6, True)])
async def test_real_pad_composite_preserves_source_pixels(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    orientation: int,
    inverted: bool,
) -> None:
    services = app.state.services
    replace = services.processes._replace
    directories = {
        name: settings.data_dir / name for name in ("comfy-input", "comfy-temp", "comfy-user")
    }
    for path in directories.values():
        path.mkdir()

    async def cpu_replace(name: str, command: list[str], *args: Any, **kwargs: Any) -> Any:
        if name == "media":
            command = [
                *command,
                "--cpu",
                "--input-directory",
                str(directories["comfy-input"]),
                "--temp-directory",
                str(directories["comfy-temp"]),
                "--user-directory",
                str(directories["comfy-user"]),
            ]
        return await replace(name, command, *args, **kwargs)

    monkeypatch.setattr(services.processes, "_replace", cpu_replace)
    async with services.scheduler.lease("primary"):
        await services.processes.start_media()
    worker = next(item for item in services.processes.statuses() if item.name == "media")
    assert worker.managed and worker.running and worker.state == "ready" and worker.pid
    record_property("worker_pid", worker.pid)
    record_property("worker_port", int(settings.comfy_url.rsplit(":", 1)[1]))
    try:
        original, expected, (width, height) = source_png(orientation)
        upload = await client.post(
            "/api/artifacts", files={"file": ("pixel-grid.png", original, "image/png")}
        )
        assert upload.status_code == 201, upload.text
        source_id = upload.json()["id"]
        with SessionLocal() as session:
            source = session.get(Artifact, source_id)
            assert source is not None
            canonical = prepare_source_fit_image(services.artifacts, source)
            with Image.open(io.BytesIO(canonical.content)) as prepared:
                assert prepared.size == (width, height)
                assert prepared.tobytes() == expected
        # Odd margins and smaller paint exercise the actual resize branch;
        # no universal VAE compression factor is assumed.
        graph: dict[str, Any] = {
            "load": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
            "pad": {
                "class_type": "ImagePadForOutpaint",
                "inputs": {
                    "image": ["load", 0],
                    "left": 2,
                    "top": 3,
                    "right": 3,
                    "bottom": 4,
                    "feathering": 0,
                },
            },
            "paint": {
                "class_type": "EmptyImage",
                "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0x3366CC},
            },
            "composite": {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": ["pad", 0],
                    "source": ["paint", 0],
                    "mask": ["invert", 0] if inverted else ["pad", 1],
                    "x": 0,
                    "y": 0,
                    "resize_source": True,
                },
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {"images": ["composite", 0], "filename_prefix": "pixel-control"},
            },
        }
        if inverted:
            graph["invert"] = {"class_type": "InvertMask", "inputs": {"mask": ["pad", 1]}}
        created = await client.post(
            "/api/workflows",
            json={
                "name": "Neutral composite pixel control",
                "operation": "image_to_image",
                "engine": "comfyui",
                "api_graph": graph,
                "input_schema": {
                    "type": "object",
                    "properties": {"input_image": {"type": "string"}},
                },
            },
        )
        assert created.status_code == 201, created.text
        workflow = created.json()
        revision_id = workflow["current_revision_id"]
        review_url = f"/api/workflows/{workflow['id']}/revisions/{revision_id}/review"
        preview = await client.get(review_url)
        assert preview.status_code == 200, preview.text
        review = await client.post(
            review_url,
            json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]},
        )
        assert review.status_code == 200 and review.json()["trusted"] is True, review.text
        profile = await client.post(
            "/api/profiles",
            json={"name": "Neutral CPU pixels", "role": "image", "engine": "comfyui"},
        )
        assert profile.status_code == 201, profile.text
        chat = await client.post("/api/chats", json={"title": "Neutral pixel control"})
        assert chat.status_code == 201, chat.text
        accepted = await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={
                "text": "Extend the neutral pixel grid",
                "mode": "image",
                "input_artifact_ids": [source_id],
                "workflow_revision_id": revision_id,
                "profile_id": profile.json()["id"],
            },
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]

        async def read() -> Mapping[str, Any] | None:
            with SessionLocal() as session:
                run = session.get(Run, run_id)
                return {"status": run.status} if run else None

        await wait_for_terminal_status(read, what="real composite pixel control")
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            outputs = run.provenance_json["outputs"]
            assert len(outputs) == 1
            output_id = outputs[0]["artifact_id"]
        response = await client.get(f"/api/artifacts/{output_id}/content")
        assert response.status_code == 200, response.text
        with Image.open(io.BytesIO(response.content)) as rendered:
            assert rendered.mode == "RGB"
            assert rendered.size == (width + 5, height + 7)
            kept = rendered.crop((2, 3, width + 2, height + 3))
            if inverted:
                assert kept.tobytes() != expected
                assert_paint(kept.tobytes())
            else:
                assert kept.tobytes() == expected
            # Inspect every added pixel, not just corner samples.
            raster = rendered.tobytes()
            for y in range(height + 7):
                for x in range(width + 5):
                    if 2 <= x < width + 2 and 3 <= y < height + 3:
                        continue
                    offset = (y * (width + 5) + x) * 3
                    pixel = raster[offset : offset + 3]
                    if inverted:
                        assert pixel == bytes((127, 127, 127))
                    else:
                        assert_paint(pixel)
        original_response = await client.get(f"/api/artifacts/{source_id}/content")
        assert original_response.status_code == 200
        assert original_response.content == original
        record_property("orientation", orientation)
        record_property("inverted_mask_control", inverted)
        record_property("source_pixels", width * height)
        record_property("canvas", f"{width + 5}x{height + 7}")
    finally:
        stopped = await services.processes.stop("media")
        assert stopped.running is False
