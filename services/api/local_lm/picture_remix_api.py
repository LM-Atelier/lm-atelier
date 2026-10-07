"""Preview a remix of a picture made elsewhere, with a workflow and model chosen here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .db import get_session
from .output_recipe import OutputRecipeUnavailable
from .output_recipe_api import exact_in_json, read_picture_generation_settings
from .picture_remix import (
    REFUSAL_MESSAGES,
    RemixChoiceInvalid,
    RemixPreview,
    RemixRole,
    RemixSource,
    preview_remix,
    read_remix_source,
)

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()

ClaimKey = Literal[
    "negative_prompt",
    "seed",
    "steps",
    "guidance",
    "sampler",
    "scheduler",
    "denoise",
    "width",
    "height",
    "shape",
]


class RemixPreviewRequest(BaseModel):
    """The workflow and model a remix would use, by local id, and which claims to apply."""

    model_config = ConfigDict(extra="forbid")

    workflow_revision_id: str = Field(min_length=1, max_length=64)
    profile_id: str = Field(min_length=1, max_length=64)
    apply: list[ClaimKey] = Field(default_factory=list, max_length=16)
    #: Whether the remix is made from words, or starts from the picture itself.
    role: RemixRole = "words"


class RemixQueueRequest(RemixPreviewRequest):
    """A previewed remix to make: the picture, the same choices, and the digest shown."""

    artifact_id: str = Field(min_length=1, max_length=80)
    review_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


@router.post("/artifacts/{artifact_id}/remix-preview")
async def preview_a_remix(
    artifact_id: str,
    payload: RemixPreviewRequest,
    request: Request,
    session: SessionDep,
) -> JSONResponse:
    """Say what a remix of this picture would run with this workflow and model, writing nothing.

    The picture is read again from its stored file; nothing its file names is
    looked up, fetched or installed. Each setting it carries is judged against
    what the chosen workflow takes. A choice that cannot run is an answer, not
    an error. The answer is never cached, since it holds the picture's prompt.
    """

    services = cast("Services", request.app.state.services)
    try:
        metadata = await run_in_threadpool(
            read_picture_generation_settings, services.artifacts, artifact_id
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    try:
        preview = await preview_remix(
            services.orchestrator,
            session,
            artifact_id,
            metadata,
            payload.workflow_revision_id,
            payload.profile_id,
            payload.apply,
            payload.role,
            await remix_source(services, artifact_id, payload.role),
        )
    except RemixChoiceInvalid as exc:
        raise api_error(
            422,
            "remix-choice-invalid",
            "Only settings the chosen workflow takes as they are can be applied, "
            "and a size only whole.",
        ) from exc
    return JSONResponse(preview_answer(artifact_id, preview), headers={"Cache-Control": "no-store"})


async def remix_source(services: Services, artifact_id: str, role: RemixRole) -> RemixSource | None:
    """The picture a remix starts from, read only for a remix that starts from it."""

    if role != "edit":
        return None
    return await run_in_threadpool(read_remix_source, services.artifacts, artifact_id)


def _claim_applied(preview: RemixPreview, key: str, setting: str | None, state: str) -> bool:
    """Whether a claim is used: its words, its kept size, or a setting chosen here."""

    if key == "prompt":
        return True
    if setting is None and state == "supported":
        # A remix starting from the picture keeps its size; nothing sets it.
        return preview.role == "edit"
    return key in preview.applied


def preview_answer(artifact_id: str, preview: RemixPreview) -> dict[str, Any]:
    metadata = preview.metadata
    return {
        "artifact_id": artifact_id,
        "metadata": {
            "dialect": metadata.dialect,
            "parser_version": metadata.parser_version,
            "budget_version": metadata.budget_version,
            "digest": metadata.digest,
        },
        "role": preview.role,
        "workflow_revision_id": preview.workflow_revision_id,
        "profile_id": preview.profile_id,
        "operation": preview.operation.value,
        # The picture a remix starts from, at the size its stored file is.
        "source": (
            {"width": preview.source.width, "height": preview.source.height}
            if preview.source is not None
            else None
        ),
        "claims": [
            {
                "key": claim.key,
                "setting": claim.setting,
                "value": exact_in_json(claim.value),
                "source": claim.source,
                "state": claim.state,
                "reason": claim.reason,
                "applied": _claim_applied(preview, claim.key, claim.setting, claim.state),
            }
            for claim in preview.claims
        ],
        # The picture's shape at a size this workflow makes, when its own size cannot be used.
        "shape": (
            {"width": preview.shape[0], "height": preview.shape[1]} if preview.shape else None
        ),
        "shape_applied": "shape" in preview.applied,
        "ignored": [{"name": item.name, "reason": item.reason} for item in metadata.ignored],
        "resolved": (
            {
                "text": preview.text,
                "settings": preview.settings,
                "seed_drawn": preview.seed_drawn,
                "trigger_words": list(preview.trigger_words),
                "engine_prompt": preview.engine_prompt,
                "strength": (
                    {
                        "parameter": preview.strength.parameter,
                        "mode": preview.strength.mode,
                        "value": preview.strength.value,
                        "from_file": preview.strength.from_file,
                    }
                    if preview.strength is not None
                    else None
                ),
            }
            if preview.ready
            else None
        ),
        "refusals": [
            {"code": code, "message": REFUSAL_MESSAGES[code]} for code in preview.refusals
        ],
        "ready": preview.ready,
        "review_digest": preview.review_digest,
    }
