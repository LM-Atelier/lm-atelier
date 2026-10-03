"""Download the portable record of how one generated output was made."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .db import SessionLocal
from .models import Run
from .output_recipe import OutputRecipe, OutputRecipeUnavailable, build_output_recipe
from .output_recipe_bundle import OutputRecipeBundle, build_output_recipe_bundle
from .output_recipe_check import (
    BUNDLE_SIGNATURE,
    OutputRecipeCheckRefused,
    check_output_recipe_file,
    most_read_for,
    read_record_file,
)
from .output_recipe_replay import plan_output_recipe_replay, replay_outcome

if TYPE_CHECKING:
    from .artifacts import ArtifactStore
    from .main import Services

router = APIRouter()


@router.get("/runs/{run_id}/outputs/{artifact_id}/recipe")
async def download_output_recipe(
    run_id: str,
    artifact_id: str,
    request: Request,
    prompts: Annotated[Literal["include", "omit"], Query()],
) -> Response:
    """The record of one output, as the exact canonical bytes a reader checks.

    Its digest covers those bytes without the digest field, behind the format's
    domain tag, and is repeated in a response header.

    The output is named by the run that made it as well as by its hash, because
    the same bytes can come from more than one run. Whether the prompt goes in
    is always the caller's explicit choice; there is no default.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        recipe = await run_in_threadpool(
            _build, artifacts, run_id, artifact_id, prompts == "include"
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    name = quote(recipe.file_name, safe="")
    return Response(
        recipe.content,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{name}",
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
            "X-Output-Recipe-Digest": recipe.digest,
        },
    )


def _build(
    artifacts: ArtifactStore, run_id: str, artifact_id: str, include_prompts: bool
) -> OutputRecipe:
    # Hashing the output reads the whole file, so the record is built off the
    # event loop with a session of its own.
    with SessionLocal() as session:
        return build_output_recipe(
            session,
            artifacts,
            run_id=run_id,
            artifact_id=artifact_id,
            include_prompts=include_prompts,
        )


@router.get("/runs/{run_id}/outputs/{artifact_id}/recipe-bundle")
async def download_output_recipe_bundle(
    run_id: str,
    artifact_id: str,
    request: Request,
    prompts: Annotated[Literal["include", "omit"], Query()],
    digest: Annotated[str, Query(pattern=r"^sha256:[0-9a-f]{64}$")],
    inputs: Annotated[Literal["include", "omit"], Query()] = "omit",
) -> Response:
    """One picture's record and a copy of the picture, together in one ZIP file.

    The record must be the one the caller was shown, named by its digest, so
    the file never holds a record nobody looked at. The picture is a copy
    without the text an engine embeds in its file, never the stored bytes.
    Its input pictures go in only when asked for, copied the same way.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        bundle = await run_in_threadpool(
            _build_bundle,
            artifacts,
            run_id,
            artifact_id,
            prompts == "include",
            digest,
            inputs == "include",
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    name = quote(bundle.file_name, safe="")
    return Response(
        bundle.content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{name}",
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
            "X-Output-Recipe-Digest": bundle.record_digest,
        },
    )


def _build_bundle(
    artifacts: ArtifactStore,
    run_id: str,
    artifact_id: str,
    include_prompts: bool,
    digest: str,
    include_inputs: bool,
) -> OutputRecipeBundle:
    # Copying the picture decodes and re-encodes it, which is slow for a large one.
    with SessionLocal() as session:
        return build_output_recipe_bundle(
            session,
            artifacts,
            run_id=run_id,
            artifact_id=artifact_id,
            include_prompts=include_prompts,
            expected_record_digest=digest,
            include_inputs=include_inputs,
        )


@router.post("/output-recipes/check")
async def check_output_recipe_against_this_install(request: Request) -> JSONResponse:
    """Which of a record's requirements this installation holds, by exact identity.

    The record, or a bundle holding it and a copy of its picture, arrives as the
    file's own bytes, because only those can be checked against their digests.
    The body is read only up to the largest file the check accepts. Nothing is
    installed, trusted or kept.
    """

    content = await read_record_body(request)
    try:
        report = await run_in_threadpool(_check, content)
    except OutputRecipeCheckRefused as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    return JSONResponse(report, headers={"Cache-Control": "no-store"})


def _check(content: bytes) -> dict[str, Any]:
    with SessionLocal() as session:
        return check_output_recipe_file(session, content)


@router.post("/output-recipes/replay-plan")
async def plan_an_exact_replay(request: Request) -> JSONResponse:
    """Whether a record could be generated again exactly here, and from what.

    Takes the same file the check takes. Each requirement is resolved to exactly
    one local workflow revision, model profile, LoRA asset or input, or refused;
    every refusal is listed. Nothing is started, installed, trusted or kept.
    """

    content = await read_record_body(request)
    media_engine = cast("Services", request.app.state.services).settings.media_engine
    try:
        plan = await run_in_threadpool(_plan, content, media_engine)
    except OutputRecipeCheckRefused as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    return JSONResponse(plan, headers={"Cache-Control": "no-store"})


@router.get("/runs/{run_id}/replay-result")
async def replay_result(run_id: str) -> JSONResponse:
    """Whether a run generated again from a record came out as the record's output did."""

    outcome = await run_in_threadpool(_replay_result, run_id)
    if outcome is None:
        raise api_error(
            404, "replay-result-not-found", "This generation was not generated again from a record."
        )
    return JSONResponse(outcome, headers={"Cache-Control": "no-store"})


def _replay_result(run_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        return replay_outcome(run) if run is not None else None


def _plan(content: bytes, media_engine: str) -> dict[str, Any]:
    record = read_record_file(content).record
    with SessionLocal() as session:
        return plan_output_recipe_replay(session, record, media_engine=media_engine)


async def read_record_body(request: Request) -> bytes:
    """A record file sent as the request body, refused once longer than a record file can be."""

    content = await _bounded_body(request)
    if content is None:
        raise api_error(
            413, "output-recipe-too-large", "This file is larger than a generation record can be."
        )
    return content


async def _bounded_body(request: Request) -> bytes | None:
    """The request body, or None once it is longer than the check reads, without reading on."""

    declared = request.headers.get("content-length")
    if (
        declared is not None
        and declared.isdigit()
        and int(declared) > most_read_for(BUNDLE_SIGNATURE)
    ):
        return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > most_read_for(bytes(body[: len(BUNDLE_SIGNATURE)])):
            return None
    return bytes(body)
