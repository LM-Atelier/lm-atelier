from __future__ import annotations

from datetime import UTC, datetime, timedelta

from local_lm.models import Job
from local_lm.progress import completed_progress, update_job_progress


def test_progress_v2_is_monotonic_within_a_stage_and_derives_legacy_progress() -> None:
    job = Job(progress=0, phase="queued", progress_json={})
    first = update_job_progress(job, stage="downloading", stage_progress=0.4)
    second = update_job_progress(job, stage="downloading", stage_progress=0.2)

    assert first["version"] == 2
    assert second["stage_progress"] == 0.4
    assert job.progress == 0.4
    assert job.phase == "downloading"


def test_progress_v2_reports_exact_bytes_speed_eta_and_stale_samples() -> None:
    job = Job(progress=0, phase="queued", progress_json={})
    started = datetime(2026, 7, 26, tzinfo=UTC)
    update_job_progress(
        job,
        stage="downloading",
        completed_units=100,
        total_units=1_000,
        unit="bytes",
        now=started,
    )
    active = update_job_progress(
        job,
        stage="downloading",
        completed_units=300,
        total_units=1_000,
        unit="bytes",
        now=started + timedelta(seconds=2),
    )
    stale = update_job_progress(
        job,
        stage="downloading",
        completed_units=400,
        total_units=1_000,
        unit="bytes",
        now=started + timedelta(seconds=10),
    )

    assert active["stage_progress"] == 0.3
    assert active["rate_bytes_per_second"] == 100
    assert active["eta_seconds"] == 7
    assert stale["rate_bytes_per_second"] is None
    assert stale["eta_seconds"] is None


def test_indeterminate_progress_has_no_plausible_percentage_or_rate() -> None:
    job = Job(progress=0, phase="queued", progress_json={})
    snapshot = update_job_progress(
        job,
        stage="testing model",
        stage_progress=0.75,
        overall_progress=0.8,
        completed_units=800,
        total_units=1_000,
        unit="bytes",
        indeterminate=True,
    )

    assert snapshot["indeterminate"] is True
    assert snapshot["stage_progress"] is None
    assert snapshot["overall_progress"] is None
    assert snapshot["rate_bytes_per_second"] is None
    assert snapshot["eta_seconds"] is None


def test_completed_progress_reaches_exactly_one() -> None:
    job = Job(progress=0.6, phase="activating", progress_json={})
    snapshot = completed_progress(job)

    assert snapshot["stage_progress"] == 1
    assert snapshot["overall_progress"] == 1
    assert job.progress == 1


def test_progress_tracks_current_and_completed_stage_timings() -> None:
    job = Job(progress=0, phase="queued", progress_json={})
    started = datetime(2026, 7, 29, tzinfo=UTC)

    first = update_job_progress(
        job,
        stage="preparing chat model",
        indeterminate=True,
        now=started,
    )
    active = update_job_progress(
        job,
        stage="preparing chat model",
        indeterminate=True,
        now=started + timedelta(seconds=2, milliseconds=250),
    )
    transitioned = update_job_progress(
        job,
        stage="waiting for first token",
        indeterminate=True,
        now=started + timedelta(seconds=3),
    )

    assert first["stage_started_at"] == "2026-07-29T00:00:00Z"
    assert first["stage_elapsed_ms"] == 0
    assert first["completed_stages"] == []
    assert active["stage_started_at"] == first["stage_started_at"]
    assert active["stage_elapsed_ms"] == 2_250
    assert transitioned["stage_elapsed_ms"] == 0
    assert transitioned["completed_stages"] == [
        {"stage": "preparing chat model", "duration_ms": 3_000}
    ]


def test_progress_stage_history_is_bounded_and_legacy_safe() -> None:
    started = datetime(2026, 7, 29, tzinfo=UTC)
    job = Job(
        progress=0,
        phase="legacy",
        progress_json={
            "version": 2,
            "stage": "legacy",
            "stage_progress": None,
            "overall_progress": None,
            "indeterminate": True,
            "updated_at": started.isoformat(),
            "completed_stages": [
                {"stage": "valid", "duration_ms": 5},
                {"stage": "", "duration_ms": 7},
                {"stage": "invalid", "duration_ms": True},
            ],
        },
    )

    snapshot = update_job_progress(job, stage="stage-0", now=started)
    for index in range(1, 30):
        snapshot = update_job_progress(
            job,
            stage=f"stage-{index}",
            now=started + timedelta(seconds=index),
        )

    assert snapshot["stage_started_at"] == "2026-07-29T00:00:29Z"
    assert snapshot["stage_elapsed_ms"] == 0
    assert len(snapshot["completed_stages"]) == 24
    assert snapshot["completed_stages"][0] == {
        "stage": "stage-5",
        "duration_ms": 1_000,
    }
    assert snapshot["completed_stages"][-1] == {
        "stage": "stage-28",
        "duration_ms": 1_000,
    }
