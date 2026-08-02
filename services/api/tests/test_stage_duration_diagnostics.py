"""Diagnostics summarize where each job kind actually spends its time."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Job

pytestmark = pytest.mark.asyncio


def _finished_media_job(identifier: str, sampling_ms: int, writing_ms: int) -> Job:
    return Job(
        id=identifier,
        kind="image",
        status="complete",
        progress_json={
            "version": 2,
            "stage": "Writing outputs",
            "stage_elapsed_ms": writing_ms,
            "completed_stages": [
                {"stage": "Loading model", "duration_ms": 4_000},
                {"stage": "Sampling", "duration_ms": sampling_ms},
            ],
        },
    )


async def test_stage_durations_are_totaled_by_kind_and_stage(client: AsyncClient) -> None:
    with SessionLocal() as session:
        session.add(_finished_media_job("job_stage_one", 20_000, 3_000))
        session.add(_finished_media_job("job_stage_two", 30_000, 5_000))
        session.commit()

    created = await client.post("/api/diagnostics")
    assert created.status_code == 201
    archive = await client.get(created.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        payload = json.loads(bundle.read("diagnostics.json"))

    stages = payload["job_stages"]["image"]
    assert stages["Loading model"] == {"jobs": 2, "total_ms": 8_000, "mean_ms": 4_000}
    assert stages["Sampling"] == {"jobs": 2, "total_ms": 50_000, "mean_ms": 25_000}
    # The stage a job finished in counts too - it is where the tail lives.
    assert stages["Writing outputs"] == {"jobs": 2, "total_ms": 8_000, "mean_ms": 4_000}


async def test_jobs_without_recorded_stages_are_simply_absent(client: AsyncClient) -> None:
    with SessionLocal() as session:
        session.add(Job(id="job_stage_bare", kind="download", status="complete", progress_json={}))
        session.commit()

    created = await client.post("/api/diagnostics")
    archive = await client.get(created.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        payload = json.loads(bundle.read("diagnostics.json"))

    assert "download" not in payload["job_stages"]
