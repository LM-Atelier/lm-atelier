"""Accepted turns retain the chosen purposes of their exact image inputs."""

import io

from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import Run, RunContextSnapshot
from local_lm.schemas import TurnRequest


async def test_image_purposes_reach_the_durable_accepted_snapshot(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    artifact_ids: list[str] = []
    for index in range(2):
        content = io.BytesIO()
        Image.new("RGB", (32 + index, 32), (40, 80 + index, 120)).save(content, format="PNG")
        uploaded = await client.post(
            "/api/artifacts",
            files={"file": (f"garden-{index}.png", content.getvalue(), "image/png")},
        )
        assert uploaded.status_code == 201
        artifact_ids.append(uploaded.json()["id"])
    chat = (await client.post("/api/chats", json={"title": "Garden picture roles"})).json()
    request = TurnRequest.model_validate(
        {
            "text": "Describe the garden layouts",
            "mode": "text",
            "input_artifact_ids": artifact_ids,
            "input_image_roles": ["reference", "edit_source"],
        }
    )
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await app.state.services.orchestrator.create_turn(
                session,
                chat["id"],
                request,
            )
            run_id = accepted.run.id
        request.input_image_roles = ["edit_source", "reference"]
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            stored = session.get(RunContextSnapshot, run_id)
            assert run is not None and stored is not None
            assert run.provenance_json["input_image_roles"] == ["reference", "edit_source"]
            assert stored.payload_json["input_image_roles"] == ["reference", "edit_source"]
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            assert snapshot.input_image_roles == ["reference", "edit_source"]
            assert snapshot.input_artifact_ids == artifact_ids
            run.provenance_json = {
                **run.provenance_json,
                "input_image_roles": ["edit_source", "reference"],
            }
            session.commit()
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            assert snapshot.input_image_roles == ["reference", "edit_source"]
            assert snapshot.input_artifact_ids == artifact_ids
