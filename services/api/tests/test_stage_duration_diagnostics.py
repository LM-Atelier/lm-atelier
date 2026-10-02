"""Diagnostics summarize where each job kind actually spends its time."""

from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest
from httpx2 import AsyncClient

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.filesystem_links import is_link_or_reparse
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


async def test_a_linked_log_is_not_measured(client: AsyncClient, settings: Settings) -> None:
    outside = settings.data_dir / "outside-log"
    outside.write_bytes(b"x" * 1_000_000)
    link = settings.log_dir / "linked.log"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    created = await client.post("/api/diagnostics")
    assert created.status_code == 201
    archive = await client.get(created.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        logs = json.loads(bundle.read("diagnostics.json"))["logs"]

    ordinary = [
        path for path in settings.log_dir.iterdir() if path.is_file() and not path.is_symlink()
    ]
    assert logs["file_count"] == len(ordinary)
    assert logs["total_bytes"] < outside.stat().st_size
    assert outside.read_bytes() == b"x" * 1_000_000
    assert link.is_symlink()


def _make_link_dir(link: Path, target: Path) -> bool:
    """Create a directory-shaped redirection, or False without privileges."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


async def test_a_linked_log_directory_is_not_measured(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = settings.data_dir / "outside-logs"
    outside.mkdir()
    secret = outside / "secret.log"
    secret.write_bytes(b"x" * 1_000_000)
    log_dir = settings.data_dir / "linked-logs"
    if not _make_link_dir(log_dir, outside):
        pytest.skip("directory links are unavailable")
    monkeypatch.setattr(Settings, "log_dir", property(lambda _self: log_dir))

    created = await client.post("/api/diagnostics")
    assert created.status_code == 201, created.text
    archive = await client.get(created.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        logs = json.loads(bundle.read("diagnostics.json"))["logs"]

    assert logs["file_count"] == 0
    assert logs["total_bytes"] == 0
    assert secret.read_bytes() == b"x" * 1_000_000
    assert is_link_or_reparse(log_dir, missing="raise", unreadable="raise")
