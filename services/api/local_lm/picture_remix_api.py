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
from .picture_remix import REFUSAL_MESSAGES, RemixChoiceInvalid, RemixPreview, preview_remix

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
        )
    except RemixChoiceInvalid as exc:
        raise api_error(
            422,
            "remix-choice-invalid",
            "Only settings the chosen workflow takes as they are can be applied, "
            "and a size only whole.",
        ) from exc
    return JSONResponse(preview_answer(artifact_id, preview), headers={"Cache-Control": "no-store"})


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
        "workflow_revision_id": preview.workflow_revision_id,
        "profile_id": preview.profile_id,
        "operation": "text_to_image",
        "claims": [
            {
                "key": claim.key,
                "setting": claim.setting,
                "value": exact_in_json(claim.value),
                "source": claim.source,
                "state": claim.state,
                "reason": claim.reason,
                "applied": claim.key == "prompt" or claim.key in preview.applied,
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
