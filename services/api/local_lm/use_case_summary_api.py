"""Offer local description summaries without changing saved use cases."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Request
from sqlalchemy import update
from sqlalchemy.orm import Session

from .api_errors import api_error
from .db import get_session
from .managed_use_case_summaries import suggest_managed_use_case_summary
from .models import ModelAssetInstall, ModelInstall, ModelProfile
from .provider_descriptions import normalize_provider_description
from .schemas import UseCaseSuggestionOut, UseCaseSuggestionRequest
from .use_case_summaries import UseCaseSummaryError

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()
_CONFLICT = "The saved use case changed. Refresh it before trying again."


def prepare_use_case_update(values: dict[str, Any]) -> str | None:
    expected = values.pop("expected_use_case", None)
    if "use_case" not in values:
        if expected is not None or "use_case_derived" in values:
            raise api_error(422, "use-case-update-missing", "Include the use case to update.")
    else:
        values["use_case_derived"] = bool(values.get("use_case_derived", False))
    return cast(str | None, expected)


def check_use_case_update(
    session: Session, record: ModelProfile | ModelAssetInstall, expected: str | None
) -> None:
    """Reserve matching text until the caller commits its update without another await."""
    if expected is None:
        return
    model = type(record)
    matched = session.scalar(
        update(model)
        .where(model.id == record.id, model.use_case == expected)
        .values(use_case=expected)
        .returning(model.id)
    )
    if matched is None:
        raise api_error(409, "use-case-changed", _CONFLICT)


def _snapshot(
    session: Session, kind: Literal["profile", "lora"], target_id: str
) -> tuple[str, str, str, str]:
    if kind == "profile":
        profile = session.get(ModelProfile, target_id)
        install = (
            session.get(ModelInstall, profile.model_install_id)
            if profile and profile.model_install_id
            else None
        )
        if profile is None or install is None:
            raise api_error(404, "use-case-source-missing", "The installed model is unavailable.")
        row: ModelProfile | ModelAssetInstall = profile
        manifest = install.manifest_json
        source_id = install.id
    else:
        asset = session.get(ModelAssetInstall, target_id)
        if asset is None:
            raise api_error(404, "model-asset-not-found", "The installed LoRA is unavailable.")
        if asset.kind != "lora":
            raise api_error(422, "use-case-suggestion-lora-only", "Choose an installed LoRA.")
        row = asset
        manifest = asset.manifest_json
        source_id = asset.id
    description = normalize_provider_description(manifest.get("provider_description"))
    return row.use_case, description, source_id, row.updated_at.isoformat()


async def _suggest(
    kind: Literal["profile", "lora"],
    target_id: str,
    payload: UseCaseSuggestionRequest,
    request: Request,
    session: Session,
) -> UseCaseSuggestionOut:
    before = _snapshot(session, kind, target_id)
    if before[0] != payload.expected_use_case:
        raise api_error(409, "use-case-changed", _CONFLICT)
    if not before[1]:
        raise api_error(
            409,
            "use-case-description-unavailable",
            "This installation has no saved provider description. "
            "You can write a use case instead.",
        )
    services = cast("Services", request.app.state.services)
    task = asyncio.create_task(suggest_managed_use_case_summary(services, session, before[1]))

    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    connection = asyncio.create_task(disconnected())
    try:
        done, _pending = await asyncio.wait({task, connection}, return_when=asyncio.FIRST_COMPLETED)
        if connection in done:
            await connection
            raise api_error(499, "use-case-suggestion-cancelled", "The suggestion was cancelled.")
        suggestion = await task
    except UseCaseSummaryError:
        raise api_error(
            409,
            "use-case-suggestion-unavailable",
            "A suggestion is unavailable. Start a local chat model and try again, "
            "or write a use case.",
        ) from None
    finally:
        connection.cancel()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await connection
        with suppress(asyncio.CancelledError, UseCaseSummaryError):
            await task
    session.expire_all()
    if _snapshot(session, kind, target_id) != before:
        raise api_error(409, "use-case-changed", _CONFLICT)
    return UseCaseSuggestionOut(suggestion=suggestion)


@router.post("/profiles/{profile_id}/use-case-suggestion", response_model=UseCaseSuggestionOut)
async def suggest_profile_use_case(
    profile_id: str, payload: UseCaseSuggestionRequest, request: Request, session: SessionDep
) -> UseCaseSuggestionOut:
    return await _suggest("profile", profile_id, payload, request, session)


@router.post("/model-assets/{asset_id}/use-case-suggestion", response_model=UseCaseSuggestionOut)
async def suggest_lora_use_case(
    asset_id: str, payload: UseCaseSuggestionRequest, request: Request, session: SessionDep
) -> UseCaseSuggestionOut:
    return await _suggest("lora", asset_id, payload, request, session)
