from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .models import Job
from .schemas import ProgressStageTiming, ProgressV2

_RATE_STALE_SECONDS = 5.0
_MAX_COMPLETED_STAGES = 24


def update_job_progress(
    job: Job,
    *,
    stage: str,
    stage_progress: float | None = None,
    overall_progress: float | None = None,
    completed_units: int | None = None,
    total_units: int | None = None,
    unit: str | None = None,
    bytes_reused: int = 0,
    file_index: int | None = None,
    file_count: int | None = None,
    queue_resource: str | None = None,
    queue_position: int | None = None,
    queue_length: int | None = None,
    blocked_by: list[str] | None = None,
    indeterminate: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Persist one validated Progress v2 snapshot and derive legacy fields."""

    previous = job.progress_json if isinstance(job.progress_json, dict) else {}
    snapshot = reduce_progress(
        previous,
        stage=stage,
        stage_progress=stage_progress,
        overall_progress=overall_progress,
        completed_units=completed_units,
        total_units=total_units,
        unit=unit,
        bytes_reused=max(0, bytes_reused),
        file_index=file_index,
        file_count=file_count,
        queue_resource=queue_resource,
        queue_position=queue_position,
        queue_length=queue_length,
        blocked_by=blocked_by,
        indeterminate=indeterminate,
        now=now,
    )
    job.progress_json = snapshot
    job.phase = stage
    normalized_overall = _optional_float(snapshot.get("overall_progress"))
    normalized_stage_progress = _optional_float(snapshot.get("stage_progress"))
    if normalized_overall is not None:
        job.progress = normalized_overall
    elif normalized_stage_progress is not None:
        job.progress = max(job.progress, normalized_stage_progress)
    return snapshot


def reduce_progress(
    previous: dict[str, Any] | None,
    *,
    stage: str,
    stage_progress: float | None = None,
    overall_progress: float | None = None,
    completed_units: int | None = None,
    total_units: int | None = None,
    unit: str | None = None,
    bytes_reused: int = 0,
    file_index: int | None = None,
    file_count: int | None = None,
    queue_resource: str | None = None,
    queue_position: int | None = None,
    queue_length: int | None = None,
    blocked_by: list[str] | None = None,
    indeterminate: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Reduce any operation into the same validated, versioned snapshot."""

    prior = previous or {}
    current_time = now or datetime.now(UTC)
    previous_stage = prior.get("stage")
    normalized_stage_progress = _fraction(stage_progress)
    normalized_overall = _fraction(overall_progress)
    if previous_stage == stage:
        previous_stage_progress = _optional_float(prior.get("stage_progress"))
        if previous_stage_progress is not None and normalized_stage_progress is not None:
            normalized_stage_progress = max(previous_stage_progress, normalized_stage_progress)
    previous_overall = _optional_float(prior.get("overall_progress"))
    if previous_overall is not None and normalized_overall is not None:
        normalized_overall = max(previous_overall, normalized_overall)

    normalized_completed = max(0, completed_units) if completed_units is not None else None
    normalized_total = max(0, total_units) if total_units is not None else None
    if normalized_total is not None and normalized_completed is not None:
        normalized_completed = min(normalized_completed, normalized_total)
        normalized_stage_progress = (
            normalized_completed / normalized_total if normalized_total > 0 else 1.0
        )

    rate, eta = _byte_rate(
        prior,
        stage=stage,
        completed_units=normalized_completed,
        total_units=normalized_total,
        unit=unit,
        now=current_time,
        indeterminate=indeterminate,
    )
    stage_started_at, stage_elapsed_ms, completed_stages = _stage_timings(
        prior,
        stage=stage,
        now=current_time,
    )
    return ProgressV2(
        stage=stage,
        stage_progress=None if indeterminate else normalized_stage_progress,
        overall_progress=None if indeterminate else normalized_overall,
        completed_units=normalized_completed,
        total_units=normalized_total,
        unit=unit,
        bytes_reused=max(0, bytes_reused),
        rate_bytes_per_second=rate,
        eta_seconds=eta,
        file_index=file_index,
        file_count=file_count,
        queue_resource=queue_resource,
        queue_position=queue_position,
        queue_length=queue_length,
        blocked_by=blocked_by or [],
        indeterminate=indeterminate,
        stage_started_at=stage_started_at,
        stage_elapsed_ms=stage_elapsed_ms,
        completed_stages=completed_stages,
        updated_at=current_time,
    ).model_dump(mode="json")


def completed_progress(job: Job, *, now: datetime | None = None) -> dict[str, Any]:
    return update_job_progress(
        job,
        stage="complete",
        stage_progress=1,
        overall_progress=1,
        now=now,
    )


def _fraction(value: float | None) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, float(value)))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _stage_timings(
    previous: dict[str, Any],
    *,
    stage: str,
    now: datetime,
) -> tuple[datetime, int, list[ProgressStageTiming]]:
    previous_stage = previous.get("stage")
    previous_started_at = _optional_datetime(previous.get("stage_started_at"))
    completed_stages = _completed_stage_timings(previous.get("completed_stages"))

    if previous_stage == stage:
        stage_started_at = previous_started_at or now
    else:
        if isinstance(previous_stage, str) and previous_started_at is not None:
            completed_stages.append(
                ProgressStageTiming(
                    stage=previous_stage,
                    duration_ms=_elapsed_ms(previous_started_at, now),
                )
            )
            completed_stages = completed_stages[-_MAX_COMPLETED_STAGES:]
        stage_started_at = now
    return (
        stage_started_at,
        _elapsed_ms(stage_started_at, now),
        completed_stages,
    )


def _completed_stage_timings(value: object) -> list[ProgressStageTiming]:
    if not isinstance(value, list):
        return []
    timings: list[ProgressStageTiming] = []
    for item in value[-_MAX_COMPLETED_STAGES:]:
        if not isinstance(item, dict):
            continue
        stage = item.get("stage")
        duration_ms = item.get("duration_ms")
        if (
            not isinstance(stage, str)
            or not stage
            or isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
        ):
            continue
        timings.append(ProgressStageTiming(stage=stage, duration_ms=max(0, duration_ms)))
    return timings


def _optional_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _elapsed_ms(started_at: datetime, now: datetime) -> int:
    return max(0, int(round((now - started_at).total_seconds() * 1_000)))


def _byte_rate(
    previous: dict[str, Any],
    *,
    stage: str,
    completed_units: int | None,
    total_units: int | None,
    unit: str | None,
    now: datetime,
    indeterminate: bool,
) -> tuple[float | None, int | None]:
    if indeterminate or unit != "bytes" or completed_units is None:
        return None, None
    if previous.get("stage") != stage or previous.get("unit") != "bytes":
        return None, None
    previous_units = previous.get("completed_units")
    previous_updated = previous.get("updated_at")
    if (
        isinstance(previous_units, bool)
        or not isinstance(previous_units, int)
        or not isinstance(previous_updated, str)
    ):
        return None, None
    try:
        previous_time = datetime.fromisoformat(previous_updated.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    elapsed = (now - previous_time).total_seconds()
    if elapsed <= 0 or elapsed > _RATE_STALE_SECONDS:
        return None, None
    transferred = completed_units - previous_units
    if transferred <= 0:
        return None, None
    rate = transferred / elapsed
    eta = None
    if total_units is not None and total_units >= completed_units and rate > 0:
        eta = int(round((total_units - completed_units) / rate))
    return rate, eta
