import math
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_output_measurement_seam import _png
from test_workflow_package_import_endpoint import _object_info, _ui_graph
from workflow_fixtures import seed_workflow_trust

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, Run
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest
from local_lm.workflow_graph_settings import bind_compiled_workflow_settings


@pytest.mark.parametrize(
    ("ratio", "budget", "multiple"), [("3:4", 0.125, 16), ("4:3", 0.25, 32), ("1:1", 0.125, 64)]
)
async def test_native_geometry_controls_reach_rendering_and_measured_provenance(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    ratio: str,
    budget: float,
    multiple: int,
) -> None:
    info, graph = _object_info(), _ui_graph()
    declarations = {
        "aspect_ratio": [["1:1", "3:4", "4:3"]],
        "megapixel_budget": ["FLOAT", {"min": 0.0625, "max": 1}],
        "round_to_multiple": ["INT", {"min": 16, "max": 64}],
    }
    info["Source"]["input"]["required"].update(declarations)
    info["Source"]["input_order"]["required"].extend(declarations)
    graph["nodes"][0]["widgets_values"].extend(["1:1", 0.0625, 16])
    compiled = compile_comfyui_ui_graph(graph, info)
    bound = bind_compiled_workflow_settings(compiled, compiled.api_graph, {})
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Native size fixture",
            "operation": "text_to_image",
            "engine": "mock",
            "ui_graph": graph,
            "api_graph": bound.api_graph,
            "input_schema": bound.input_schema,
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    chat = (await client.post("/api/chats", json={"title": "Native size execution"})).json()
    observed: list[tuple[dict[str, Any], int, int]] = []

    async def generate(request: MediaRequest) -> Any:
        inputs = ComfyUIAdapter._compile(request.workflow, request.parameters)["1"]["inputs"]
        assert "width" not in inputs and "height" not in inputs
        # This neutral fixture implements its own documented size contract.
        # A third-party selector needs its own implementation evidence.
        rw, rh = map(int, inputs["aspect_ratio"].split(":"))
        step = inputs["round_to_multiple"]
        unit = math.lcm(step // math.gcd(rw, step), step // math.gcd(rh, step))
        pixels = int(inputs["megapixel_budget"] * 1_000_000)
        scale = math.isqrt(pixels // (rw * rh * unit * unit)) * unit
        width, height = rw * scale, rh * scale
        observed.append((inputs, width, height))
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=_png(width, height),
                    kind="image",
                    media_type="image/png",
                    name="native-size.png",
                )
            ],
        )

    orchestrator = app.state.services.orchestrator
    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    selected = {"aspect_ratio": ratio, "megapixels": budget, "round_to_multiple": multiple}
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="A neutral geometric pattern.",
                    mode="image",
                    settings=selected,
                    workflow_revision_id=revision_id,
                ),
                freeze_context=True,
                activate_branch=False,
            )
            run_id = accepted.run.id
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="native-size-attempt", attempt=1)
            job.status, job.claim_owner, job.attempt = "running", claim.token, claim.attempt
            job_id = job.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)
    assert len(observed) == 1
    inputs, width, height = observed[0]
    assert inputs["aspect_ratio"] == ratio and inputs["megapixel_budget"] == budget
    assert inputs["round_to_multiple"] == multiple
    rw, rh = map(int, ratio.split(":"))
    assert width * rh == height * rw
    assert width % multiple == height % multiple == 0
    assert width * height <= budget * 1_000_000
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete"
        assert run.workflow_revision_id == revision_id
        for key, value in selected.items():
            assert run.settings_json[key] == value
            assert run.provenance_json["resolved_settings"][key] == value
        outputs = [
            artifact
            for artifact in session.scalars(select(Artifact))
            if artifact.metadata_json.get("run_id") == run_id
        ]
        assert len(outputs) == 1
        measured = outputs[0].metadata_json["output_measurement"]
        assert measured["state"] == "measured"
        assert (measured["raster_width"], measured["raster_height"]) == (width, height)
