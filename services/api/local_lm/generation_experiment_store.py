"""Accept a resolved comparison once, and read it back only as it was accepted.

Accepting writes one comparison, its two choices and one planned picture per
choice, in one commit with nothing awaited between the first write and the
commit. Reading recomputes every digest from what is stored and refuses a
record that no longer matches them.
"""

from __future__ import annotations

import copy
import secrets
from dataclasses import dataclass
from typing import Any, cast, get_args

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import __version__
from .domain import utcnow
from .generation_experiment_preflight import (
    ExperimentResolution,
    ResolvedArm,
    common_inputs,
    preflight_digest,
)
from .generation_experiments_v1 import (
    CONTRACT_VERSION,
    ExperimentArmOut,
    ExperimentEvaluationOut,
    ExperimentTrialOut,
    GenerationExperimentCreate,
    GenerationExperimentEvaluationCreate,
    GenerationExperimentEvaluationMode,
    GenerationExperimentOut,
    GenerationExperimentStart,
    GenerationExperimentState,
    GenerationExperimentTrialState,
    SeedPolicy,
    SeedPolicyKind,
    TrialWorkStatus,
    canonical_sha256,
)
from .models import (
    GenerationExperiment,
    GenerationExperimentArm,
    GenerationExperimentEvaluation,
    GenerationExperimentTrial,
    Job,
    Run,
    WorkStep,
)
from .orchestrator import MEDIA_SEED_SPACE


class GenerationExperimentRecordError(Exception):
    """A stored comparison does not match its own digests."""


class GenerationExperimentKeyConflict(Exception):
    """One idempotency key was already used for a different request."""


def _work_status(value: str) -> TrialWorkStatus | None:
    """A step's status as a picture's, or None for a value a step should never hold."""

    return cast(TrialWorkStatus, value) if value in get_args(TrialWorkStatus) else None


@dataclass(frozen=True)
class TrialProgress:
    """Where one started picture's work stands, read from its own step, run and job."""

    work_step_id: str | None
    run_id: str | None
    job_id: str | None
    status: TrialWorkStatus


def start_request_digest(payload: GenerationExperimentStart) -> str:
    """Everything a start asked for except its key, so a retry is recognised as one."""

    return canonical_sha256(payload.model_dump(mode="json", exclude={"idempotency_key"}))


def link_started_work(
    experiment: GenerationExperiment,
    payload: GenerationExperimentStart,
    work_plan_id: str,
    links: dict[str, tuple[str, str]],
) -> None:
    """Record, once, the work plan and each picture's step and run."""

    if experiment.state != GenerationExperimentState.READY.value or experiment.work_plan_id:
        raise GenerationExperimentRecordError
    experiment.state = GenerationExperimentState.STARTED.value
    experiment.work_plan_id = work_plan_id
    experiment.start_idempotency_key = payload.idempotency_key
    experiment.start_request_sha256 = start_request_digest(payload)
    experiment.started_at = utcnow()
    for arm in experiment.arms:
        for trial in arm.trials:
            if trial.work_step_id or trial.run_id:
                raise GenerationExperimentRecordError
            trial.work_step_id, trial.run_id = links[trial.id]
            trial.state = GenerationExperimentTrialState.STARTED.value


def read_trial_progress(
    session: Session, experiment: GenerationExperiment
) -> dict[str, TrialProgress]:
    """Each started picture's status, after checking its rows still describe that picture.

    The step, run and job are rows of their own, written when the comparison
    started; a link that no longer leads back to this picture's choice and
    seed is a damaged record, not a status. Work deleted since reads as removed.
    """

    progress: dict[str, TrialProgress] = {}
    if experiment.state != GenerationExperimentState.STARTED.value:
        return progress
    for arm in experiment.arms:
        for trial in arm.trials:
            if trial.work_step_id is None or trial.run_id is None:
                progress[trial.id] = TrialProgress(
                    trial.work_step_id, trial.run_id, None, "removed"
                )
                continue
            step = session.get(WorkStep, trial.work_step_id)
            run = session.get(Run, trial.run_id)
            if step is None or run is None:
                progress[trial.id] = TrialProgress(None, None, None, "removed")
                continue
            witness = (run.provenance_json or {}).get("generation_experiment") or {}
            status = _work_status(step.status)
            if (
                step.plan_id != experiment.work_plan_id
                or step.run_id != run.id
                or run.work_step_id != step.id
                or run.profile_id != arm.profile_id
                or run.workflow_revision_id != arm.workflow_revision_id
                or (run.settings_json or {}).get("seed") != trial.seed
                or witness.get("trial_id") != trial.id
                or status is None
            ):
                raise GenerationExperimentRecordError
            job = session.scalar(select(Job).where(Job.work_step_id == step.id))
            progress[trial.id] = TrialProgress(
                step.id, run.id, job.id if job is not None else None, status
            )
    return progress


def choices_with_a_picture(session: Session, experiment: GenerationExperiment) -> set[int]:
    """The ordinals of the choices whose picture is made, for saying which is preferred."""

    progress = read_trial_progress(session, experiment)
    return {
        arm.ordinal
        for arm in experiment.arms
        if any(
            trial.id in progress and progress[trial.id].status == "complete" for trial in arm.trials
        )
    }


def request_digest(payload: GenerationExperimentCreate) -> str:
    """Everything a create asked for except its key, so a retry is recognised as one."""

    return canonical_sha256(payload.model_dump(mode="json", exclude={"idempotency_key"}))


def draw_seeds(policy: SeedPolicy, arms: int) -> list[int]:
    """The seed each choice's picture runs with; never the -1 that leaves the choice to a run."""

    if policy.kind == SeedPolicyKind.FIXED_NUMERIC:
        if policy.seed is None:
            raise ValueError("A fixed seed needs its number.")
        return [policy.seed] * arms
    if policy.kind == SeedPolicyKind.SAME_RECORDED_NUMBER:
        seed = policy.seed if policy.seed is not None else secrets.randbelow(MEDIA_SEED_SPACE)
        return [seed] * arms
    if policy.kind == SeedPolicyKind.INDEPENDENT_DETERMINISTIC:
        base = policy.seed if policy.seed is not None else secrets.randbelow(MEDIA_SEED_SPACE)
        return [(base + offset) % MEDIA_SEED_SPACE for offset in range(arms)]
    return [secrets.randbelow(MEDIA_SEED_SPACE) for _ in range(arms)]


def _record_digest(
    experiment: dict[str, str],
    preflight_sha256: str,
    arm_digests: list[str],
    trials: list[tuple[int, int, int]],
    seed_equivalence: str,
    estimate: list[dict[str, Any]],
) -> str:
    """The accepted comparison: what the preflight covered, plus its name and what was added."""

    return canonical_sha256(
        {
            "experiment": experiment,
            "preflight_sha256": preflight_sha256,
            "arms": arm_digests,
            "trials": [list(trial) for trial in trials],
            "seed_equivalence": seed_equivalence,
            "estimate": estimate,
        }
    )


def _identity(name: str, operation: str, seed_policy: str) -> dict[str, str]:
    return {"name": name, "operation": operation, "seed_policy": seed_policy}


def find(session: Session, idempotency_key: str) -> GenerationExperiment | None:
    return session.scalar(
        select(GenerationExperiment).where(GenerationExperiment.idempotency_key == idempotency_key)
    )


def _arm_row(arm: ResolvedArm, seed: int) -> GenerationExperimentArm:
    return GenerationExperimentArm(
        ordinal=arm.ordinal,
        label=arm.label,
        profile_id=arm.profile_id,
        workflow_revision_id=arm.workflow_revision_id,
        workflow_activation_id=arm.workflow_activation_id,
        model_family=arm.model_family,
        requested_settings_json=copy.deepcopy(arm.snapshot["requested_settings"]),
        effective_settings_json=copy.deepcopy(arm.effective_settings),
        snapshot_json=copy.deepcopy(arm.snapshot),
        snapshot_sha256=arm.snapshot_sha256,
        trials=[
            GenerationExperimentTrial(
                ordinal=1, seed=seed, state=GenerationExperimentTrialState.PLANNED.value
            )
        ],
    )


def create(
    session: Session,
    payload: GenerationExperimentCreate,
    resolution: ExperimentResolution,
    seeds: list[int],
) -> tuple[GenerationExperiment, bool]:
    """Write the accepted comparison in one commit, or refuse a reused key.

    Two creates with one key can both get this far. The unique key decides:
    the loser rolls back and returns the winner's record, saying it did not
    create it, or refuses if the winner asked for something else.
    """

    arms = [arm for arm in resolution.arms if arm is not None]
    if not resolution.compatible or resolution.preflight_sha256 is None or len(arms) != len(seeds):
        raise ValueError("Only a compatible resolution with one seed per choice is accepted.")
    trials = [(arm.ordinal, 1, seed) for arm, seed in zip(arms, seeds, strict=True)]
    estimate = [item.model_dump(mode="json") for item in resolution.estimate]
    experiment = GenerationExperiment(
        name=payload.name,
        state=GenerationExperimentState.READY.value,
        operation=payload.operation,
        contract_version=CONTRACT_VERSION,
        app_version=__version__,
        seed_policy=payload.seed_policy.kind.value,
        seed_equivalence=resolution.seed_equivalence,
        common_json=common_inputs(payload),
        estimate_json=estimate,
        preflight_sha256=resolution.preflight_sha256,
        snapshot_sha256=_record_digest(
            _identity(payload.name, payload.operation, payload.seed_policy.kind.value),
            resolution.preflight_sha256,
            [arm.snapshot_sha256 for arm in arms],
            trials,
            resolution.seed_equivalence,
            estimate,
        ),
        idempotency_key=payload.idempotency_key,
        request_sha256=request_digest(payload),
        arms=[_arm_row(arm, seed) for arm, seed in zip(arms, seeds, strict=True)],
    )
    session.add(experiment)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = find(session, payload.idempotency_key)
        if existing is None:
            raise
        if existing.request_sha256 != request_digest(payload):
            raise GenerationExperimentKeyConflict from None
        return existing, False
    return experiment, True


def _arm_out(arm: GenerationExperimentArm, progress: dict[str, TrialProgress]) -> ExperimentArmOut:
    snapshot: dict[str, Any] = arm.snapshot_json
    if canonical_sha256(snapshot) != arm.snapshot_sha256:
        raise GenerationExperimentRecordError
    if snapshot.get("effective_settings") != arm.effective_settings_json:
        raise GenerationExperimentRecordError
    profile = snapshot.get("profile") or {}
    workflow = snapshot.get("workflow") or {}
    activation = snapshot.get("activation") or {}
    # The columns are what other reads join on; each must still say what its snapshot says.
    if (
        arm.profile_id != profile.get("id")
        or arm.workflow_revision_id != workflow.get("id")
        or arm.workflow_activation_id != activation.get("id")
        or arm.model_family != snapshot.get("model_family")
        or arm.requested_settings_json != snapshot.get("requested_settings")
    ):
        raise GenerationExperimentRecordError
    geometry = snapshot.get("geometry") or {}
    # A stored value the answer cannot hold, such as a state no release wrote,
    # is a record that changed: refused like any other, never a server error.
    try:
        return ExperimentArmOut(
            id=arm.id,
            ordinal=arm.ordinal,
            label=arm.label,
            profile_id=arm.profile_id,
            profile_name=profile.get("name"),
            workflow_revision_id=arm.workflow_revision_id,
            workflow_version=workflow.get("version"),
            workflow_activation_id=arm.workflow_activation_id,
            model_family=arm.model_family,
            width=geometry["width"],
            height=geometry["height"],
            effective_settings=copy.deepcopy(arm.effective_settings_json),
            trigger_words_applied=list(
                (snapshot.get("trigger_words") or {}).get("trigger_words_applied", [])
            ),
            snapshot_sha256=arm.snapshot_sha256,
            trials=[
                ExperimentTrialOut(
                    id=trial.id,
                    ordinal=trial.ordinal,
                    seed=trial.seed,
                    state=GenerationExperimentTrialState(trial.state),
                    work_step_id=progress[trial.id].work_step_id if trial.id in progress else None,
                    run_id=progress[trial.id].run_id if trial.id in progress else None,
                    job_id=progress[trial.id].job_id if trial.id in progress else None,
                    status=progress[trial.id].status if trial.id in progress else None,
                )
                for trial in arm.trials
            ],
        )
    except (KeyError, TypeError, ValueError):
        raise GenerationExperimentRecordError from None


def out(
    experiment: GenerationExperiment, progress: dict[str, TrialProgress] | None = None
) -> GenerationExperimentOut:
    """The comparison as it was accepted, or a refusal if any stored part has changed."""

    arms = [_arm_out(arm, progress or {}) for arm in experiment.arms]
    common = experiment.common_json
    shared = preflight_digest(
        common, [(arm.ordinal, arm.label, arm.snapshot_sha256) for arm in arms]
    )
    trials = [(arm.ordinal, trial.ordinal, trial.seed) for arm in arms for trial in arm.trials]
    recomputed = _record_digest(
        _identity(experiment.name, experiment.operation, experiment.seed_policy),
        experiment.preflight_sha256,
        [arm.snapshot_sha256 for arm in arms],
        trials,
        experiment.seed_equivalence,
        experiment.estimate_json,
    )
    if shared != experiment.preflight_sha256 or recomputed != experiment.snapshot_sha256:
        raise GenerationExperimentRecordError
    try:
        return GenerationExperimentOut.model_validate(
            {
                "id": experiment.id,
                "name": experiment.name,
                "state": experiment.state,
                "operation": experiment.operation,
                "prompt": common["prompt"],
                "negative_prompt": common["negative_prompt"],
                "geometry": common["geometry"],
                "seed_policy": common["seed_policy"],
                "seed_equivalence": experiment.seed_equivalence,
                "preflight_sha256": experiment.preflight_sha256,
                "snapshot_sha256": experiment.snapshot_sha256,
                "estimate": experiment.estimate_json,
                "created_at": experiment.created_at,
                "work_plan_id": experiment.work_plan_id,
                "started_at": experiment.started_at,
                "arms": arms,
                "evaluation": _latest_evaluation(experiment),
            }
        )
    except (KeyError, TypeError, ValueError):
        raise GenerationExperimentRecordError from None


def _latest_evaluation(experiment: GenerationExperiment) -> ExperimentEvaluationOut | None:
    if not experiment.evaluations:
        return None
    latest = experiment.evaluations[-1]
    ordinals = {arm.id: arm.ordinal for arm in experiment.arms}
    return ExperimentEvaluationOut.model_validate(
        {
            "preference": latest.preference,
            "mode": latest.mode,
            "arm_ordinal": (
                ordinals[latest.preferred_arm_id] if latest.preferred_arm_id is not None else None
            ),
            "note": latest.note,
            "created_at": latest.created_at,
        }
    )


def evaluate(
    session: Session,
    experiment_id: str,
    payload: GenerationExperimentEvaluationCreate,
    mode: GenerationExperimentEvaluationMode,
) -> GenerationExperiment | None:
    """Keep one more thing said of a comparison's pictures, after everything said before.

    The write lock is taken before the next place in order is read, so two
    sayings at once are kept one after the other rather than refused. None when
    there is no such comparison.
    """

    session.execute(text("BEGIN IMMEDIATE"))
    experiment = session.get(GenerationExperiment, experiment_id, populate_existing=True)
    if experiment is None:
        session.rollback()
        return None
    preferred = next(
        (arm.id for arm in experiment.arms if arm.ordinal == payload.arm_ordinal), None
    )
    if payload.arm_ordinal is not None and preferred is None:
        session.rollback()
        raise ValueError("A preference names one of the comparison's own choices.")
    sequence = max((said.sequence for said in experiment.evaluations), default=0) + 1
    experiment.evaluations.append(
        GenerationExperimentEvaluation(
            sequence=sequence,
            mode=mode.value,
            preference=payload.preference.value,
            preferred_arm_id=preferred,
            note=payload.note or None,
        )
    )
    session.commit()
    return experiment
