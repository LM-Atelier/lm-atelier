"""Comparing a blind comparison's pictures without knowing which choice made which.

A viewing has its own random order of the pictures and names them only by
where they are shown. Its answers carry no step, run, job, artifact or label,
and each picture is served re-encoded, with nothing embedded that a workflow
wrote into the file and under no name of its own, so neither the page nor what
it fetches says which picture is which. Saying a preference in the viewing
keeps it as a blind saying, ends the viewing and returns the reveal; from then
on the comparison reads as one that names its choices.
"""

from __future__ import annotations

import io
import secrets
from typing import Any

from sqlalchemy.orm import Session

from . import generation_experiment_store as store
from .artifacts import ArtifactStore
from .generation_experiments_v1 import (
    BlindPositionOut,
    BlindRevealOut,
    GenerationExperimentBlindEvaluationCreate,
    GenerationExperimentBlindViewOut,
    GenerationExperimentEvaluationCreate,
    GenerationExperimentEvaluationMode,
    GenerationExperimentState,
)
from .models import (
    Artifact,
    GenerationExperiment,
    GenerationExperimentArm,
    GenerationExperimentBlindView,
    GenerationExperimentTrial,
    Run,
)
from .studio_region_edit import MAX_BLEND_READ_BYTES, RegionEditError, decode_picture


class BlindViewRefused(Exception):
    """A blind view step that cannot be taken; the code is one of the comparison API's own."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def open_view(session: Session, experiment: GenerationExperiment) -> GenerationExperimentBlindView:
    """Begin a viewing with its own order of the pictures, drawn at random."""

    _require_blind_pending(experiment)
    order = [arm.id for arm in experiment.arms]
    if secrets.randbelow(2):
        order.reverse()
    view = GenerationExperimentBlindView(experiment_id=experiment.id, order_json=order)
    session.add(view)
    session.commit()
    return view


def find_view(
    session: Session, experiment: GenerationExperiment, view_id: str
) -> GenerationExperimentBlindView:
    view = session.get(GenerationExperimentBlindView, view_id)
    if view is None or view.experiment_id != experiment.id:
        raise BlindViewRefused("generation-experiment-blind-view-not-found")
    arms = {arm.id for arm in experiment.arms}
    if sorted(view.order_json) != sorted(arms) or len(view.order_json) != len(arms):
        raise BlindViewRefused("generation-experiment-record-invalid")
    return view


def view_out(
    session: Session, experiment: GenerationExperiment, view: GenerationExperimentBlindView
) -> GenerationExperimentBlindViewOut:
    """The viewing by position: each picture's progress, and the reveal once it is said."""

    progress = store.read_trial_progress(session, experiment)
    positions = []
    for position, trial in enumerate(_trials_in_order(experiment, view), start=1):
        status = progress[trial.id].status if trial.id in progress else None
        positions.append(
            BlindPositionOut(
                position=position,
                status=status,
                ready=status == "complete" and _picture_artifact(session, trial) is not None,
            )
        )
    said = next((item for item in experiment.evaluations if item.id == view.evaluation_id), None)
    reveal = None
    evaluation = None
    if said is not None:
        arms = _arms_in_order(experiment, view)
        reveal = [
            BlindRevealOut(position=position, arm_ordinal=arm.ordinal, label=arm.label)
            for position, arm in enumerate(arms, start=1)
        ]
        evaluation = store.out(experiment).evaluation
    return GenerationExperimentBlindViewOut(
        id=view.id,
        experiment_id=experiment.id,
        positions=positions,
        evaluation=evaluation,
        reveal=reveal,
    )


def view_picture(
    session: Session,
    artifacts: ArtifactStore,
    experiment: GenerationExperiment,
    view: GenerationExperimentBlindView,
    position: int,
) -> bytes:
    """The picture at a position as a plain PNG, with nothing a workflow embedded in it."""

    trials = _trials_in_order(experiment, view)
    if not 1 <= position <= len(trials):
        raise BlindViewRefused("generation-experiment-blind-picture-not-ready")
    artifact = _picture_artifact(session, trials[position - 1])
    if artifact is None:
        raise BlindViewRefused("generation-experiment-blind-picture-not-ready")
    try:
        picture = decode_picture(
            artifacts.verified_bytes(artifact, maximum_bytes=MAX_BLEND_READ_BYTES), "picture"
        )
    except (RegionEditError, ValueError, OSError):
        raise BlindViewRefused("generation-experiment-blind-picture-unreadable") from None
    plain = picture.convert("RGBA" if picture.mode in {"RGBA", "LA", "PA"} else "RGB")
    # Saved without the source's text chunks, profile or other metadata: a
    # workflow can write its own name and settings into the file it makes.
    plain.info = {}
    buffer = io.BytesIO()
    plain.save(buffer, format="PNG")
    return buffer.getvalue()


def say_blind(
    session: Session,
    experiment: GenerationExperiment,
    view: GenerationExperimentBlindView,
    payload: GenerationExperimentBlindEvaluationCreate,
) -> GenerationExperiment:
    """Keep a preference said in a viewing as the comparison's blind saying."""

    _require_blind_pending(experiment)
    arms = _arms_in_order(experiment, view)
    said = GenerationExperimentEvaluationCreate(
        preference=payload.preference,
        arm_ordinal=arms[payload.position - 1].ordinal if payload.position is not None else None,
        note=payload.note,
    )
    # As when the choices are named: nothing is said before a picture is made,
    # and only a picture that is made can be preferred.
    made = store.choices_with_a_picture(session, experiment)
    if not made or (said.arm_ordinal is not None and said.arm_ordinal not in made):
        raise BlindViewRefused("generation-experiment-picture-not-ready")
    try:
        updated = store.evaluate(
            session,
            experiment.id,
            said,
            GenerationExperimentEvaluationMode.BLIND,
            blind_view_id=view.id,
        )
    except ValueError:
        raise BlindViewRefused("generation-experiment-revealed") from None
    if updated is None:
        raise BlindViewRefused("generation-experiment-not-found")
    return updated


def _require_blind_pending(experiment: GenerationExperiment) -> None:
    if experiment.common_json.get("evaluation_mode") != GenerationExperimentEvaluationMode.BLIND:
        raise BlindViewRefused("generation-experiment-not-blind")
    if experiment.state != GenerationExperimentState.STARTED.value:
        raise BlindViewRefused("generation-experiment-not-started")
    if not store.blind_pending(experiment):
        raise BlindViewRefused("generation-experiment-revealed")


def _arms_in_order(
    experiment: GenerationExperiment, view: GenerationExperimentBlindView
) -> list[GenerationExperimentArm]:
    by_id = {arm.id: arm for arm in experiment.arms}
    return [by_id[arm_id] for arm_id in view.order_json]


def _trials_in_order(
    experiment: GenerationExperiment, view: GenerationExperimentBlindView
) -> list[GenerationExperimentTrial]:
    # One picture per choice: a comparison makes exactly one each.
    return [arm.trials[0] for arm in _arms_in_order(experiment, view)]


def _picture_artifact(session: Session, trial: GenerationExperimentTrial) -> Artifact | None:
    """The trial's kept picture, chosen as the comparison page chooses it."""

    run = session.get(Run, trial.run_id) if trial.run_id else None
    if run is None or run.work_step_id != trial.work_step_id:
        return None
    provenance: dict[str, Any] = run.provenance_json or {}
    witness = provenance.get("generation_experiment") or {}
    if witness.get("trial_id") != trial.id:
        return None
    for output in provenance.get("outputs") or []:
        if not isinstance(output, dict):
            continue
        origin = output.get("output_origin") or {}
        # A preview the engine does not keep is never the picture shown.
        if origin.get("state") == "attributed" and origin.get("output_type") == "temp":
            continue
        artifact_id = output.get("artifact_id")
        if output.get("kind") == "image" and isinstance(artifact_id, str) and artifact_id:
            return session.get(Artifact, artifact_id)
    return None
