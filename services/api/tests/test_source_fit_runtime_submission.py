"""Opt-in source-fit HTTP submission through a real managed CPU ComfyUI."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from run_waits import wait_for_terminal_status
from test_source_fit_output_pixels import records
from test_source_fit_runtime_pixels import settings as settings
from test_workflow_source_geometry import composited_graph

from local_lm.accepted_turn_context import accepted_context
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Run


def oriented_source(width: int, height: int) -> tuple[bytes, bytes]:
    raw = bytes(
        component
        for index in range(width * height)
        for component in (index % 256, (index * 37) % 256, 255 - index % 256)
    )
    # Stored axes are reversed. Derive clockwise output independently of
    # Pillow's transpose and the application's canonical decoder.
    expected = b"".join(
        raw[((width - 1 - x) * height + y) * 3 : ((width - 1 - x) * height + y) * 3 + 3]
        for y in range(height)
        for x in range(width)
    )
    with Image.frombytes("RGB", (height, width), raw) as image:
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="PNG", exif=exif)
    return stream.getvalue(), expected


async def build_checkpoint(settings: Settings) -> Path:
    assert settings.comfy_executable and settings.comfy_directory
    destination = settings.data_dir / "neutral-weights"
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = await asyncio.to_thread(
        subprocess.run,
        [
            str(settings.comfy_executable),
            str(Path(__file__).with_name("source_fit_checkpoint_fixture.py")),
            "--runtime",
            str(settings.comfy_directory),
            "--output",
            str(destination),
            "--text-encoder",
        ],
        cwd=settings.comfy_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["execution"]["text_encoder"] is True
    return destination


def isolated_worker(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch, weights: Path
) -> None:
    replace = app.state.services.processes._replace
    directories = {name: settings.data_dir / name for name in ("input", "temp", "user")}
    for directory in directories.values():
        directory.mkdir()
    model_config = settings.data_dir / "neutral-model-paths.json"
    model_config.write_text(
        json.dumps(
            {"neutral": {"base_path": str(weights), "checkpoints": ".", "text_encoders": "."}}
        ),
        encoding="utf-8",
    )

    async def cpu_replace(name: str, command: list[str], *args: Any, **kwargs: Any) -> Any:
        if name == "media":
            command = list(command)
            command[command.index("--extra-model-paths-config") + 1] = str(model_config)
            command.extend(
                [
                    "--cpu",
                    "--input-directory",
                    str(directories["input"]),
                    "--temp-directory",
                    str(directories["temp"]),
                    "--user-directory",
                    str(directories["user"]),
                ]
            )
            kwargs["environment_overrides"] = {
                **(kwargs.get("environment_overrides") or {}),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        return await replace(name, command, *args, **kwargs)

    monkeypatch.setattr(app.state.services.processes, "_replace", cpu_replace)


async def create_workflow(client: AsyncClient) -> str:
    graph = composited_graph()
    graph["clip"] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "neutral-text.safetensors",
            "type": "stable_diffusion",
            "device": "cpu",
        },
    }
    for name in ("positive", "negative"):
        graph[name]["inputs"]["clip"] = ["clip", 0]
    graph["sample"]["inputs"]["seed"] = "${seed}"
    graph["positive"]["inputs"]["text"] = "${prompt}"
    properties: dict[str, Any] = {
        "input_image": {"type": "string"},
        "prompt": {"type": "string"},
        "checkpoint": {"type": "string", "default": "neutral-untrained.safetensors"},
        "seed": {"type": "integer", "default": 731},
        **{
            f"outpaint_{side}_px": {"type": "integer", "default": 0}
            for side in ("left", "top", "right", "bottom")
        },
    }
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral source canvas runtime",
            "operation": "image_to_image",
            "engine": "comfyui",
            "api_graph": graph,
            "input_schema": {"type": "object", "properties": properties},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    revision = workflow["current_revision_id"]
    review_url = f"/api/workflows/{workflow['id']}/revisions/{revision}/review"
    preview = await client.get(review_url)
    assert preview.status_code == 200, preview.text
    approved = await client.post(
        review_url,
        json={
            "action": "approve",
            "subject_sha256": preview.json()["subject_sha256"],
        },
    )
    assert approved.status_code == 200 and approved.json()["trusted"] is True, approved.text
    assert isinstance(revision, str)
    return revision


@pytest.mark.parametrize("source_size,canvas", [((48, 36), (64, 36)), ((64, 36), (72, 128))])
async def test_source_fit_preview_and_submission_reach_real_generated_pixels(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    source_size: tuple[int, int],
    canvas: tuple[int, int],
) -> None:
    weights = await build_checkpoint(settings)
    isolated_worker(app, settings, monkeypatch, weights)
    services = app.state.services
    try:
        async with services.scheduler.lease("primary"):
            await services.processes.start_media()
        worker = next(item for item in services.processes.statuses() if item.name == "media")
        assert worker.managed and worker.running and worker.state == "ready" and worker.pid
        record_property("worker_pid", worker.pid)
        revision = await create_workflow(client)
        original, expected = oriented_source(*source_size)
        uploaded = await client.post(
            "/api/artifacts", files={"file": ("neutral-grid.png", original, "image/png")}
        )
        assert uploaded.status_code == 201, uploaded.text
        source_id = uploaded.json()["id"]
        intent = {"mode": "extend", "width": canvas[0], "height": canvas[1]}
        fit_url = f"/api/workflow-revisions/{revision}/source-fit"
        capability = await client.get(fit_url)
        assert capability.status_code == 200 and capability.json()["available"] is True
        preview_response = await client.post(
            fit_url + "/preview", json={"source_artifact_id": source_id, "source_fit": intent}
        )
        assert preview_response.status_code == 200, preview_response.text
        preview = preview_response.json()
        left = (canvas[0] - source_size[0]) // 2
        top = (canvas[1] - source_size[1]) // 2
        assert preview["source_rectangle"] == {
            "x": left,
            "y": top,
            "width": source_size[0],
            "height": source_size[1],
        }
        profile = await client.post(
            "/api/profiles",
            json={"name": "Neutral CPU source canvas", "role": "image", "engine": "comfyui"},
        )
        assert profile.status_code == 201, profile.text
        chat = await client.post("/api/chats", json={"title": "Neutral runtime source canvas"})
        assert chat.status_code == 201, chat.text
        turns_url = f"/api/chats/{chat.json()['id']}/turns"
        rendered_pixels = []
        for seed in (731, 732):
            request = {
                "text": "Extend the neutral geometric grid",
                "mode": "image",
                "input_artifact_ids": [source_id],
                "workflow_revision_id": revision,
                "profile_id": profile.json()["id"],
                "source_fit": intent,
                "settings": {"seed": seed},
                "idempotency_key": f"neutral-source-canvas-{seed}",
            }
            accepted = await client.post(turns_url, json=request)
            assert accepted.status_code == 202, accepted.text
            run_id = accepted.json()["run"]["id"]
            replay = await client.post(turns_url, json=request)
            assert replay.status_code == 202 and replay.json()["run"]["id"] == run_id

            async def read(selected_run_id: str = run_id) -> Mapping[str, Any] | None:
                with SessionLocal() as session:
                    run = session.get(Run, selected_run_id)
                    return {"status": run.status} if run else None

            await wait_for_terminal_status(read, what="real source-fit generation")
            with SessionLocal() as session:
                run = session.get(Run, run_id)
                assert run is not None
                context = accepted_context(session, run)
                assert context is not None and context.source_fit is not None
                assert context.workflow is not None and context.workflow.id == revision
                assert context.source_fit.image.source_artifact_id == source_id
                assert (context.source_fit.canvas_width, context.source_fit.canvas_height) == canvas
            outputs, parts = records(run_id)
            assert len(outputs) == len(parts) == 1
            assert (
                outputs[0]["source_fit_agreement"]
                == parts[0]["source_fit_agreement"]
                == {"v": 1, "state": "preserved"}
            )
            output_id = outputs[0]["artifact_id"]
            with SessionLocal() as session:
                artifact = session.get(Artifact, output_id)
                assert artifact is not None and "source_fit_agreement" not in artifact.metadata_json
            response = await client.get(f"/api/artifacts/{output_id}/content")
            assert response.status_code == 200
            with Image.open(io.BytesIO(response.content)) as image:
                assert image.size == canvas and image.mode == "RGB"
                assert (
                    image.crop((left, top, left + source_size[0], top + source_size[1])).tobytes()
                    == expected
                )
                pixels = image.tobytes()
                added = {
                    pixels[(y * canvas[0] + x) * 3 : (y * canvas[0] + x) * 3 + 3]
                    for y in range(canvas[1])
                    for x in range(canvas[0])
                    if not (left <= x < left + source_size[0] and top <= y < top + source_size[1])
                }
                assert len(added) > 1 and added != {bytes((127, 127, 127))}
                rendered_pixels.append(pixels)
        assert rendered_pixels[0] != rendered_pixels[1]
        original_response = await client.get(f"/api/artifacts/{source_id}/content")
        assert original_response.status_code == 200 and original_response.content == original
        record_property("source", source_size)
        record_property("canvas", canvas)
        record_property("source_preserved", True)
        record_property("generated_extension_varies_with_seed", True)
    finally:
        stopped = await services.processes.stop("media")
        assert stopped.running is False
