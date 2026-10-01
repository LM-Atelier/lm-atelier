"""Accept a resolved comparison once, and read it back only as it was accepted.

Accepting writes one comparison, its two choices and one planned picture per
choice, in one commit with nothing awaited between the first write and the
commit. Reading recomputes every digest from what is stored and refuses a
record that no longer matches them.
"""

from __future__ import annotations

import copy
import secrets
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import __version__
from .generation_experiment_preflight import (
    ExperimentResolution,
    ResolvedArm,
    common_inputs,
    preflight_digest,
)
from .generation_experiments_v1 import (
    CONTRACT_VERSION,
    ExperimentArmOut,
    ExperimentTrialOut,
    GenerationExperimentCreate,
    GenerationExperimentOut,
    GenerationExperimentState,
    GenerationExperimentTrialState,
    SeedPolicy,
    SeedPolicyKind,
    canonical_sha256,
)
from .models import GenerationExperiment, GenerationExperimentArm, GenerationExperimentTrial
from .orchestrator import MEDIA_SEED_SPACE


class GenerationExperimentRecordError(Exception):
    """A stored comparison does not match its own digests."""


class GenerationExperimentKeyConflict(Exception):
    """One idempotency key was already used for a different request."""


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


def _arm_out(arm: GenerationExperimentArm) -> ExperimentArmOut:
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
            )
            for trial in arm.trials
        ],
    )


def out(experiment: GenerationExperiment) -> GenerationExperimentOut:
    """The comparison as it was accepted, or a refusal if any stored part has changed."""

    arms = [_arm_out(arm) for arm in experiment.arms]
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
                "arms": arms,
            }
        )
    except (KeyError, TypeError, ValueError):
        raise GenerationExperimentRecordError from None
