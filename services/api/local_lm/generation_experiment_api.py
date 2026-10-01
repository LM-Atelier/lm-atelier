"""Check two generation choices against one frozen request before any work is accepted."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from .db import get_session
from .generation_experiment_preflight import resolve_generation_experiment
from .generation_experiments_v1 import (
    GenerationExperimentPreflightOut,
    GenerationExperimentRequest,
)

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()


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
