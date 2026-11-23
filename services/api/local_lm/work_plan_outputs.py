"""What each output of a work plan was made with, read back from what the plan recorded."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from .domain import ArtifactKind
from .models import Run, WorkPlan
from .output_origin import names_a_preview
from .output_recipe import recorded_seed

_KINDS = frozenset(kind.value for kind in ArtifactKind)
_BATCH_SIZE_LIMIT = 16


def plan_outputs(session: Session, plan: WorkPlan) -> list[dict[str, Any]]:
    """Each of the plan's outputs in order, with its seed and the files it kept.

    One entry per step, in the order the plan made them. The seed is the one
    the step's run recorded, with whether the workflow read it. The files are
    the ones the run kept, previews left out; files one engine call made
    together share that call's seed, and batch_index says where each came.
    Odd or missing provenance gives fewer files, never an error.
    """

    entries: list[dict[str, Any]] = []
    steps = sorted(plan.steps, key=lambda step: (step.ordinal, step.id))
    for index, step in enumerate(steps, start=1):
        run = session.get(Run, step.run_id) if step.run_id else None
        if run is None:
            seed: dict[str, Any] = {"value": None, "binding": "not_recorded"}
            settings: dict[str, Any] = {}
        else:
            seed, settings = recorded_seed(session, run)
        entries.append(
            {
                "index": index,
                "work_step_id": step.id,
                "run_id": run.id if run is not None else None,
                "message_id": run.assistant_message_id if run is not None else None,
                "operation": step.operation,
                "status": step.status,
                "seed": seed["value"],
                "seed_binding": seed["binding"],
                "batch_size": _batch_size(settings),
                "artifacts": _kept_files(run) if run is not None else [],
            }
        )
    return entries


def _batch_size(settings: dict[str, Any]) -> int:
    value = settings.get("batch_size")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _BATCH_SIZE_LIMIT:
        return 1
    return value


def _kept_files(run: Run) -> list[dict[str, Any]]:
    """The files the run kept, in the order the engine gave them.

    batch_index counts files from the same output node and collection, so the
    pictures one save node wrote in a single call are numbered from zero.
    """

    provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
    outputs = provenance.get("outputs")
    places: dict[tuple[str, str], int] = {}
    files: list[dict[str, Any]] = []
    for item in outputs if isinstance(outputs, list) else []:
        if not isinstance(item, dict):
            continue
        origin = item.get("output_origin")
        if names_a_preview(origin):
            continue
        artifact_id, sha256 = item.get("artifact_id"), item.get("sha256")
        kind, media_type = item.get("kind"), item.get("media_type")
        if not (
            isinstance(artifact_id, str)
            and isinstance(sha256, str)
            and isinstance(media_type, str)
            and kind in _KINDS
        ):
            continue
        place = origin if isinstance(origin, dict) else {}
        key = (str(place.get("node_id", "")), str(place.get("collection", "")))
        batch_index = places.get(key, 0)
        places[key] = batch_index + 1
        files.append(
            {
                "artifact_id": artifact_id,
                "sha256": sha256,
                "kind": kind,
                "media_type": media_type,
                "batch_index": batch_index,
            }
        )
    return files
