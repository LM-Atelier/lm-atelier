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
