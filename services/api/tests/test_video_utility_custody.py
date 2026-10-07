"""What a video utility job holds, and when it lets go.

A job holds the video it reads while it is waiting, running or interrupted, so
the video cannot be deleted under it. Once it has finished, and only after its
work has ended, it names that video and what it made without holding either,
so both can be deleted like any other file. Running it again holds the video
again, and only while that video is still stored and not in Recently Deleted.
"""

from __future__ import annotations

import asyncio
import importlib
import shutil
import subprocess
import threading
import time
from collections.abc import Set as AbstractSet
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.artifact_library import _job_ids, ensure_library_entry, referenced_artifact_ids
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind, JobStatus, utcnow
from local_lm.models import Artifact, Job

PATTERN = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10"]
H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "5"]


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    assert executable, "these cases need a real ffmpeg on PATH, as hosted CI provides"
    return executable


def _video(directory: Path, seconds: int) -> bytes:
    target = directory / f"walk-{seconds}.mp4"
    subprocess.run(
        [
            *(_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"),
            *(*PATTERN, "-t", str(seconds), *H264, str(target)),
        ],
        check=True,
        timeout=60,
    )
    return target.read_bytes()


def _utilities() -> Any:
    """The manager's module, imported where a case uses it rather than at collection."""

    return importlib.import_module("local_lm.video_utilities")


def _store(app: FastAPI, directory: Path, seconds: int = 1) -> str:
    """A stored video with no library entry, so only references keep it."""

    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session,
            _video(directory, seconds),
            kind=ArtifactKind.VIDEO,
            media_type="video/mp4",
            original_name="walk.mp4",
        )
        session.commit()
        return str(artifact.id)


def _stage(app: FastAPI, source: str, at: float = 0.0) -> str:
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = app.state.services.video_utilities.stage_frame(session, artifact, at)
        session.commit()
        return str(job.id)


async def _finished(client: AsyncClient, job_id: str) -> dict[str, Any]:
    async with asyncio.timeout(60):
        while True:
            job = (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()
            if job["status"] in {"complete", "failed", "cancelled"}:
                return dict(job)
            await asyncio.sleep(0.05)


def _held(job_id: str) -> set[str]:
    """The files a job's own payload and result hold, as retention reads them."""

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        return _job_ids(job.payload_json) | _job_ids(job.result_json)


def _kept(*, apart_from_library: AbstractSet[str] = frozenset()) -> set[str]:
    with SessionLocal() as session:
        return set(
            referenced_artifact_ids(session, exclude_library_membership_for=apart_from_library)
        )


async def test_a_finished_job_holds_neither_its_video_nor_what_it_made(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, tmp_path)
    job_id = _stage(app, source, 0.2)
    assert _held(job_id) == {source}

    app.state.services.video_utilities.start(job_id)
    job = await _finished(client, job_id)

    assert job["status"] == "complete", job
    result = job["result_json"]
    picture = result["made_artifact_id"]
    assert (result["from_artifact_id"], job["payload_json"]["from_artifact_id"]) == (source, source)
    assert "source_artifact_id" not in job["payload_json"]
    assert {"result_artifact_id", "source_artifact_id"}.isdisjoint(result)
    assert _held(job_id) == set()
    # The picture is kept by its library entry alone, and the video by nothing.
    assert picture in _kept()
    assert {source, picture}.isdisjoint(_kept(apart_from_library={picture}))
    with SessionLocal() as session:
        made = session.get(Artifact, picture)
        assert made is not None
        # The picture's own record still names where it came from.
        assert made.metadata_json["video_frame"]["source_artifact_id"] == source

    deleted = await client.delete(f"/api/artifacts/{source}")

    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["removed_count"] == 1
    with SessionLocal() as session:
        assert session.get(Artifact, source) is None


async def test_a_job_holds_its_video_while_it_waits_and_runs(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, tmp_path)
    utilities = _utilities()
    entered = asyncio.Event()
    go = asyncio.Event()
    decode = utilities.decode_frame

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await go.wait()
        return await decode(*args, **kwargs)

    monkeypatch.setattr(utilities, "decode_frame", held)
    job_id = _stage(app, source)
    assert source in _kept()
    app.state.services.video_utilities.start(job_id)
    await asyncio.wait_for(entered.wait(), timeout=30)

    assert _held(job_id) == {source}
    refused = await client.delete(f"/api/artifacts/{source}")
    assert (refused.status_code, refused.json()["code"]) == (409, "artifact-in-use")

    go.set()
    assert (await _finished(client, job_id))["status"] == "complete"
    assert source not in _kept()


async def test_a_failed_job_lets_its_video_go(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, tmp_path)
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = app.state.services.video_utilities.stage_frame(session, artifact, 0.0)
        # The request names a digest the stored video does not have.
        job.payload_json = {**job.payload_json, "source_sha256": "0" * 64}
        session.commit()
        job_id = job.id
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)

    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-utility-source-changed"},
    )
    assert failed["payload_json"]["from_artifact_id"] == source
    assert _held(job_id) == set()
    assert source not in _kept()


async def test_a_cancelled_job_lets_its_video_go_only_once_its_work_has_ended(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, tmp_path)
    utilities = _utilities()
    entered = asyncio.Event()
    unwound: list[bool] = []
    decode = utilities.decode_frame

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # The job still holds its video while its work unwinds.
            unwound.append(_held(job_id) == {source})
            raise
        return await decode(*args, **kwargs)

    monkeypatch.setattr(utilities, "decode_frame", held)
    job_id = _stage(app, source)
    app.state.services.video_utilities.start(job_id)
    await asyncio.wait_for(entered.wait(), timeout=30)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    assert unwound == [True]
    assert (await _finished(client, job_id))["status"] == "cancelled"
    assert _held(job_id) == set()
    assert source not in _kept()


async def test_a_job_cancelled_while_it_saves_finishes_the_save_and_then_lets_go(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, tmp_path)
    utilities = _utilities()
    saving = threading.Event()
    finish = utilities.VideoUtilityManager._finish
    holding: list[set[str]] = []

    def slow_finish(self: Any, job_id: str, *args: Any) -> None:
        saving.set()
        time.sleep(0.5)
        holding.append(_held(job_id))
        finish(self, job_id, *args)

    monkeypatch.setattr(utilities.VideoUtilityManager, "_finish", slow_finish)
    job_id = _stage(app, source)
    app.state.services.video_utilities.start(job_id)
    assert await asyncio.to_thread(saving.wait, 30)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert (cancelled.status_code, cancelled.json()["code"]) == (409, "job-not-cancellable")
    assert holding == [{source}]
    job = await _finished(client, job_id)
    assert job["status"] == "complete", job
    assert _held(job_id) == set()


async def test_a_displaced_attempt_writes_nothing_and_its_job_keeps_holding(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, tmp_path)
    utilities = _utilities()
    entered = asyncio.Event()
    go = asyncio.Event()
    decode = utilities.decode_frame

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await go.wait()
        return await decode(*args, **kwargs)

    monkeypatch.setattr(utilities, "decode_frame", held)
    job_id = _stage(app, source)
    app.state.services.video_utilities.start(job_id)
    await asyncio.wait_for(entered.wait(), timeout=30)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        # Another dispatcher has taken the job over.
        job.claim_owner = "another-dispatcher_claim"
        session.commit()

    go.set()
    async with asyncio.timeout(30):
        while app.state.services.video_utilities._tasks.get(job_id) is not None:
            await asyncio.sleep(0.05)

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert (job.status, job.result_json) == (JobStatus.RUNNING.value, {})
    assert _held(job_id) == {source}
    assert source in _kept()


async def test_a_job_left_running_by_a_hard_exit_keeps_holding_until_it_runs_again(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, tmp_path)
    job_id = _stage(app, source, 0.2)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        # As a dispatcher that exited without writing anything left it.
        job.status = JobStatus.RUNNING.value
        job.claim_owner = "gone-dispatcher_claim"
        job.claim_expires_at = utcnow() - timedelta(minutes=5)
        session.commit()

    expired = app.state.services.scheduler._expire_foreign_claims("cpu_media")

    assert job_id in expired
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.status == JobStatus.INTERRUPTED.value
    assert _held(job_id) == {source}

    app.state.services.video_utilities.recover()

    finished = await _finished(client, job_id)
    assert finished["status"] == "complete", finished
    assert _held(job_id) == set()


def _failed_once(app: FastAPI, source: str) -> str:
    """A job that failed on a digest its video does not have, its request then put right."""

    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = app.state.services.video_utilities.stage_frame(session, artifact, 0.0)
        job.payload_json = {**job.payload_json, "source_sha256": "0" * 64}
        session.commit()
        return str(job.id)


def _put_right(job_id: str, source: str) -> None:
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        stored = session.get(Job, job_id)
        assert artifact is not None and stored is not None
        stored.payload_json = {**stored.payload_json, "source_sha256": artifact.sha256}
        session.commit()


async def _trash(client: AsyncClient, artifact_id: str) -> None:
    """The video moved to Recently Deleted, as the Media Library moves one there."""

    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        entry = ensure_library_entry(session, artifact)
        session.commit()
        assert entry is not None
        entry_id = entry.id
    impact = (await client.get(f"/api/artifact-library/{entry_id}/deletion-impact")).json()
    trashed = await client.post(
        f"/api/artifact-library/{entry_id}/trash",
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": "trash-the-video",
        },
    )
    assert trashed.status_code == 200, trashed.text


async def test_a_retry_holds_the_video_again_while_it_is_stored(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, tmp_path)
    job_id = _failed_once(app, source)
    app.state.services.video_utilities.start(job_id)
    assert (await _finished(client, job_id))["status"] == "failed"
    assert _held(job_id) == set()
    _put_right(job_id, source)

    retried = await client.post(f"/api/jobs/{job_id}/retry")

    assert retried.status_code == 200, retried.text
    assert retried.json()["payload_json"]["source_artifact_id"] == source
    assert "from_artifact_id" not in retried.json()["payload_json"]
    assert (await _finished(client, job_id))["status"] == "complete"
    assert _held(job_id) == set()


async def test_a_retry_of_a_job_whose_video_is_in_recently_deleted_queues_nothing(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, tmp_path, seconds=2)
    job_id = _failed_once(app, source)
    app.state.services.video_utilities.start(job_id)
    assert (await _finished(client, job_id))["status"] == "failed"
    _put_right(job_id, source)
    await _trash(client, source)

    refused = await client.post(f"/api/jobs/{job_id}/retry")

    assert (refused.status_code, refused.json()["code"]) == (409, "video-utility-source-gone")
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None and stored.status == JobStatus.FAILED.value
    assert _held(job_id) == set()


async def test_a_retry_of_a_job_whose_video_is_gone_queues_nothing(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, tmp_path)
    job_id = _stage(app, source)
    app.state.services.video_utilities.start(job_id)
    assert (await _finished(client, job_id))["status"] == "complete"
    assert (await client.delete(f"/api/artifacts/{source}")).status_code == 200
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        # As if the job had failed, so that it may be run again.
        stored.status = JobStatus.FAILED.value
        session.commit()

    refused = await client.post(f"/api/jobs/{job_id}/retry")

    assert (refused.status_code, refused.json()["code"]) == (409, "video-utility-source-gone")
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None and stored.status == JobStatus.FAILED.value
    assert _held(job_id) == set()


def test_an_unfinished_job_that_cannot_hold_its_video_ends_instead_of_waiting(app: FastAPI) -> None:
    with SessionLocal() as session:
        job = Job(
            kind="media_utility",
            status=JobStatus.INTERRUPTED.value,
            queue_resource="cpu_media",
            queue_group="cpu_media",
            queue_ticket="ticket-neutral",
            enqueued_at=utcnow(),
            payload_json={
                "version": 1,
                "action": "extract_frame",
                "from_artifact_id": "sha256:" + "1" * 64,
                "source_sha256": "1" * 64,
                "requested_seconds": 0.0,
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id

    app.state.services.video_utilities.recover()

    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        assert (stored.status, stored.result_json) == (
            JobStatus.FAILED.value,
            {"failure_code": "video-utility-source-gone"},
        )
        assert stored.claim_owner is None
