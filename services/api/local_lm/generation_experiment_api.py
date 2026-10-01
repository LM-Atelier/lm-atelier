"""Check two generation choices against one frozen request, accept it, and read it back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from . import generation_experiment_store as store
from .api_errors import api_error
from .db import get_session
from .generation_experiment_preflight import resolve_generation_experiment
from .generation_experiments_v1 import (
    GenerationExperimentCreate,
    GenerationExperimentOut,
    GenerationExperimentPreflightOut,
    GenerationExperimentRequest,
)
from .models import GenerationExperiment

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()

REFUSALS: dict[str, tuple[int, str]] = {
    "generation-experiment-refused": (
        422,
        "A choice in this comparison cannot run as asked. Check it again for the reasons.",
    ),
    "generation-experiment-preflight-changed": (
        409,
        "A choice would now run differently from when it was checked. Check it again.",
    ),
    "generation-experiment-idempotency-conflict": (
        409,
        "This request key was already used for a different comparison.",
    ),
    "generation-experiment-not-found": (404, "This comparison no longer exists."),
    "generation-experiment-record-invalid": (
        409,
        "This comparison's stored record no longer matches what was accepted.",
    ),
}


def _refuse(code: str, **extra: object) -> Exception:
    status, message = REFUSALS[code]
    return api_error(status, code, message, **extra)


def _out(experiment: GenerationExperiment) -> GenerationExperimentOut:
    try:
        return store.out(experiment)
    except store.GenerationExperimentRecordError:
        raise _refuse("generation-experiment-record-invalid") from None


@router.post("/generation-experiments/preflight", response_model=GenerationExperimentPreflightOut)
async def preflight_generation_experiment(
    payload: GenerationExperimentRequest, request: Request, session: SessionDep
) -> GenerationExperimentPreflightOut:
    """Say whether both choices can run as asked, and exactly how, writing nothing.

    A refusal is an answer, not an error: the response names every choice that
    cannot run and why, with one fixed sentence per reason.
    """

    services = cast("Services", request.app.state.services)
    resolution = await resolve_generation_experiment(
        services.orchestrator, services.settings, session, payload
    )
    return resolution.out()


@router.post("/generation-experiments", response_model=GenerationExperimentOut, status_code=201)
async def create_generation_experiment(
    payload: GenerationExperimentCreate,
    request: Request,
    response: Response,
    session: SessionDep,
) -> GenerationExperimentOut:
    """Accept a checked comparison: freeze both choices and give each picture its seed.

    The choices are resolved again here, and the comparison is accepted only
    if they come to the digest the preflight returned. A retry with the same
    key and request returns the comparison already accepted.
    """

    existing = store.find(session, payload.idempotency_key)
    if existing is not None:
        if existing.request_sha256 != store.request_digest(payload):
            raise _refuse("generation-experiment-idempotency-conflict")
        response.status_code = 200
        return _out(existing)
    services = cast("Services", request.app.state.services)
    resolution = await resolve_generation_experiment(
        services.orchestrator, services.settings, session, payload
    )
    if not resolution.compatible:
        raise _refuse(
            "generation-experiment-refused",
            refusals=[refusal.model_dump(mode="json") for refusal in resolution.refusals],
        )
    if resolution.preflight_sha256 != payload.preflight_sha256:
        raise _refuse("generation-experiment-preflight-changed")
    seeds = store.draw_seeds(payload.seed_policy, len(payload.arms))
    try:
        experiment, created = store.create(session, payload, resolution, seeds)
    except store.GenerationExperimentKeyConflict:
        raise _refuse("generation-experiment-idempotency-conflict") from None
    if not created:
        response.status_code = 200
    return _out(experiment)


@router.get("/generation-experiments/{experiment_id}", response_model=GenerationExperimentOut)
def read_generation_experiment(experiment_id: str, session: SessionDep) -> GenerationExperimentOut:
    """The comparison exactly as it was accepted, checked against its digests."""

    experiment = session.get(GenerationExperiment, experiment_id)
    if experiment is None:
        raise _refuse("generation-experiment-not-found")
    return _out(experiment)
