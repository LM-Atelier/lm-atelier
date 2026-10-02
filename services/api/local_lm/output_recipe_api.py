"""Download the portable record of how one generated output was made."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .db import SessionLocal
from .output_recipe import OutputRecipe, OutputRecipeUnavailable, build_output_recipe
from .output_recipe_check import OutputRecipeCheckRefused, check_output_recipe

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


@router.post("/output-recipes/check")
async def check_output_recipe_against_this_install(request: Request) -> JSONResponse:
    """Which of a record's requirements this installation holds, by exact identity.

    The record arrives as the file's own bytes, because only those can be checked
    against its digest. Nothing is installed, trusted or kept.
    """

    content = await request.body()
    try:
        report = await run_in_threadpool(_check, content)
    except OutputRecipeCheckRefused as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    return JSONResponse(report, headers={"Cache-Control": "no-store"})


def _check(content: bytes) -> dict[str, Any]:
    with SessionLocal() as session:
        return check_output_recipe(session, content)
