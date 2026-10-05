"""Check, accept, start and read a comparison of two generation choices; keep which is preferred."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from . import generation_experiment_blind as blind
from . import generation_experiment_store as store
from .api_errors import api_error
from .db import get_session
from .generation_experiment_preflight import resolve_generation_experiment
from .generation_experiment_recipe import RecipeDraftRefused, recipe_draft
from .generation_experiment_start import StartRefused, start_generation_experiment
from .generation_experiments_v1 import (
    GenerationExperimentBlindEvaluationCreate,
    GenerationExperimentBlindViewOut,
    GenerationExperimentCreate,
    GenerationExperimentEvaluationCreate,
    GenerationExperimentEvaluationMode,
    GenerationExperimentOut,
    GenerationExperimentPreflightOut,
    GenerationExperimentRecipeDraftOut,
    GenerationExperimentRequest,
    GenerationExperimentStart,
    GenerationExperimentState,
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
    "generation-experiment-arm-not-found": (404, "This comparison has no such choice."),
    "generation-experiment-not-started": (
        409,
        "Make the pictures before saying which you prefer.",
    ),
    "generation-experiment-picture-not-ready": (
        409,
        "Wait for the picture before saying which you prefer.",
    ),
    "generation-experiment-blind": (
        409,
        "Say which picture you prefer before seeing which choice made it.",
    ),
    "generation-experiment-not-blind": (
        409,
        "This comparison names its choices, so there is nothing to compare blind.",
    ),
    "generation-experiment-revealed": (
        409,
        "These pictures were already compared blind.",
    ),
    "generation-experiment-blind-view-not-found": (404, "This blind viewing no longer exists."),
    "generation-experiment-blind-picture-not-ready": (409, "This picture is not ready yet."),
    "generation-experiment-blind-picture-unreadable": (
        409,
        "This picture could not be read.",
    ),
    "generation-experiment-recipe-unavailable": (
        409,
        "The workflow this choice ran on cannot take a recipe now.",
    ),
    "generation-experiment-recipe-not-for-changes": (
        409,
        "A way of changing a picture cannot be kept as a recipe yet.",
    ),
    "generation-experiment-record-invalid": (
        409,
        "This comparison's stored record no longer matches what was accepted.",
    ),
    "generation-experiment-already-started": (409, "This comparison has already been started."),
    "generation-experiment-snapshot-changed": (
        409,
        "This comparison changed since it was read. Read it again before starting it.",
    ),
    "generation-experiment-confirmation-required": (
        409,
        "These pictures are large. Confirm that they should be made.",
    ),
    "generation-experiment-storage-insufficient": (
        409,
        "There is not enough free storage to make these pictures.",
    ),
    "generation-experiment-unavailable": (
        503,
        "New work is not being accepted right now. Try again in a moment.",
    ),
}


def _refuse(code: str, **extra: object) -> Exception:
    status, message = REFUSALS[code]
    return api_error(status, code, message, **extra)


def _out(session: Session, experiment: GenerationExperiment) -> GenerationExperimentOut:
    try:
        return store.out(experiment, store.read_trial_progress(session, experiment))
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
        return _out(session, existing)
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
    except IntegrityError as error:
        if str(error.orig) != "workflow-recovery-write-refused":
            raise
        session.rollback()
        raise _refuse("generation-experiment-preflight-changed") from None
    if not created:
        response.status_code = 200
    return _out(session, experiment)


@router.get("/generation-experiments/{experiment_id}", response_model=GenerationExperimentOut)
def read_generation_experiment(experiment_id: str, session: SessionDep) -> GenerationExperimentOut:
    """The comparison exactly as it was accepted, checked against its digests."""

    experiment = session.get(GenerationExperiment, experiment_id)
    if experiment is None:
        raise _refuse("generation-experiment-not-found")
    return _out(session, experiment)


@router.post(
    "/generation-experiments/{experiment_id}/start",
    response_model=GenerationExperimentOut,
    status_code=202,
)
async def start_generation_experiment_route(
    experiment_id: str,
    payload: GenerationExperimentStart,
    request: Request,
    response: Response,
    session: SessionDep,
) -> GenerationExperimentOut:
    """Queue one picture per choice, exactly as accepted, as independent work.

    A retry with the same key and request returns the comparison already
    started. Starting checks each choice's accepted identities again and
    refuses, writing nothing, if either would now run differently.
    """

    services = cast("Services", request.app.state.services)
    orchestrator = services.orchestrator
    try:
        outcome = start_generation_experiment(
            orchestrator, services.settings, session, experiment_id, payload
        )
    except StartRefused as refused:
        raise _refuse(refused.code, **refused.extra) from None
    if outcome.created:
        try:
            await orchestrator.events.publish(
                "work_plan.created", outcome.work_plan_id, outcome.announcement
            )
        finally:
            for job_id, run_id in outcome.dispatch:
                orchestrator.start(job_id, run_id)
    else:
        response.status_code = 200
    experiment = session.get(GenerationExperiment, experiment_id, populate_existing=True)
    if experiment is None:
        raise _refuse("generation-experiment-not-found")
    return _out(session, experiment)


@router.get(
    "/generation-experiments/{experiment_id}/arms/{arm_ordinal}/recipe-draft",
    response_model=GenerationExperimentRecipeDraftOut,
)
async def draft_generation_experiment_recipe(
    experiment_id: str, arm_ordinal: int, request: Request, session: SessionDep
) -> GenerationExperimentRecipeDraftOut:
    """A recipe to review from one choice: the settings it ran with that a recipe can hold.

    Nothing is saved. The answer names the choice's model and workflow beside
    the settings, and says why any setting it ran with was left out.
    """

    services = cast("Services", request.app.state.services)
    try:
        return await recipe_draft(services.orchestrator, session, experiment_id, arm_ordinal)
    except RecipeDraftRefused as refused:
        raise _refuse(refused.code) from None


@router.post(
    "/generation-experiments/{experiment_id}/evaluations",
    response_model=GenerationExperimentOut,
    status_code=201,
)
def evaluate_generation_experiment(
    experiment_id: str, payload: GenerationExperimentEvaluationCreate, session: SessionDep
) -> GenerationExperimentOut:
    """Keep which picture the person preferred, a tie, or neither suiting, with the choices named.

    Only a picture that is made can be spoken of: nothing is kept before one
    is, and a preferred choice must have its own. Every saying is kept, and
    the latest is the comparison's answer from then on.
    """

    experiment = session.get(GenerationExperiment, experiment_id)
    if experiment is None:
        raise _refuse("generation-experiment-not-found")
    # A record that changed after it was accepted takes nothing more.
    _out(session, experiment)
    if experiment.state != GenerationExperimentState.STARTED.value:
        raise _refuse("generation-experiment-not-started")
    if store.blind_pending(experiment):
        raise _refuse("generation-experiment-blind")
    # Said only of pictures there are: none made yet, or preferring a choice
    # whose picture is not made, is refused. An unknown choice is left to the
    # saying itself, which refuses it as invalid.
    made = store.choices_with_a_picture(session, experiment)
    named = payload.arm_ordinal
    if not made or (named in {arm.ordinal for arm in experiment.arms} and named not in made):
        raise _refuse("generation-experiment-picture-not-ready")
    updated = store.evaluate(
        session, experiment_id, payload, GenerationExperimentEvaluationMode.UNBLINDED
    )
    if updated is None:
        raise _refuse("generation-experiment-not-found")
    return _out(session, updated)


def _checked(session: Session, experiment_id: str) -> GenerationExperiment:
    """The comparison, refused if it is gone or its record no longer holds what was accepted."""

    experiment = session.get(GenerationExperiment, experiment_id)
    if experiment is None:
        raise _refuse("generation-experiment-not-found")
    _out(session, experiment)
    return experiment


@router.post(
    "/generation-experiments/{experiment_id}/blind-views",
    response_model=GenerationExperimentBlindViewOut,
    status_code=201,
)
def open_blind_view(experiment_id: str, session: SessionDep) -> GenerationExperimentBlindViewOut:
    """Begin a viewing of a blind comparison, with its own random order of the pictures."""

    experiment = _checked(session, experiment_id)
    try:
        view = blind.open_view(session, experiment)
        return blind.view_out(session, experiment, view)
    except blind.BlindViewRefused as refused:
        raise _refuse(refused.code) from None


@router.get(
    "/generation-experiments/{experiment_id}/blind-views/{view_id}",
    response_model=GenerationExperimentBlindViewOut,
)
def read_blind_view(
    experiment_id: str, view_id: str, session: SessionDep
) -> GenerationExperimentBlindViewOut:
    """Each picture's progress by position, and the reveal once the preference is said."""

    experiment = _checked(session, experiment_id)
    try:
        return blind.view_out(session, experiment, blind.find_view(session, experiment, view_id))
    except blind.BlindViewRefused as refused:
        raise _refuse(refused.code) from None


@router.get("/generation-experiments/{experiment_id}/blind-views/{view_id}/pictures/{position}")
async def blind_view_picture(
    experiment_id: str, view_id: str, position: int, request: Request, session: SessionDep
) -> Response:
    """The picture at a position, as a plain PNG under no name of its own."""

    experiment = _checked(session, experiment_id)
    services = cast("Services", request.app.state.services)
    try:
        view = blind.find_view(session, experiment, view_id)
        # Decoding and encoding a large picture takes a while; it runs off the loop.
        content = await run_in_threadpool(
            blind.view_picture, session, services.artifacts, experiment, view, position
        )
    except blind.BlindViewRefused as refused:
        raise _refuse(refused.code) from None
    return Response(content=content, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post(
    "/generation-experiments/{experiment_id}/blind-views/{view_id}/evaluations",
    response_model=GenerationExperimentBlindViewOut,
    status_code=201,
)
def say_blind_preference(
    experiment_id: str,
    view_id: str,
    payload: GenerationExperimentBlindEvaluationCreate,
    session: SessionDep,
) -> GenerationExperimentBlindViewOut:
    """Keep the preference said in a viewing, and reveal which choice made each picture."""

    experiment = _checked(session, experiment_id)
    try:
        view = blind.find_view(session, experiment, view_id)
        updated = blind.say_blind(session, experiment, view, payload)
        return blind.view_out(session, updated, view)
    except blind.BlindViewRefused as refused:
        raise _refuse(refused.code) from None
