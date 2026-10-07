"""Expose independent recipe and scoped-choice management."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from . import workflow_use_case_preset_service as service
from .api_errors import api_error
from .db import get_session
from .workflow_use_case_presets_v1 import (
    WorkflowUseCaseChoice,
    WorkflowUseCaseDefault,
    WorkflowUseCasePresetCreate,
    WorkflowUseCasePresetOut,
)
from .workflow_use_cases_v1 import WorkflowUseCase

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter()

_REFUSALS: dict[service.ServiceRefusal, tuple[int, str]] = {
    "workflow-use-case-preset-conflict": (
        409,
        "The recipe changed or conflicts with another recipe. Refresh and try again.",
    ),
    "workflow-use-case-preset-not-found": (404, "The selected recipe no longer exists."),
    "workflow-use-case-preset-builtin": (409, "Built-in recipes cannot be edited or deleted."),
    "workflow-use-case-preset-in-use": (
        409,
        "Remove this recipe's selections before disabling or deleting it.",
    ),
    "workflow-use-case-preset-disabled": (409, "Choose an enabled recipe."),
    "workflow-use-case-preset-mismatch": (422, "Choose a recipe for the same use case."),
    "workflow-use-case-preset-invalid": (409, "The stored recipe is invalid and needs repair."),
    "workflow-use-case-scope-not-found": (404, "The chat or project no longer exists."),
}


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except service.WorkflowUseCasePresetServiceError as exc:
        status, message = _REFUSALS[exc.code]
        raise api_error(status, exc.code, message) from exc


def _out(record: service.WorkflowUseCasePresetRecord) -> WorkflowUseCasePresetOut:
    try:
        return WorkflowUseCasePresetOut.model_validate(record, from_attributes=True)
    except ValidationError:
        raise service.WorkflowUseCasePresetServiceError(
            "workflow-use-case-preset-invalid"
        ) from None


@router.get("/workflow-use-case-presets", response_model=list[WorkflowUseCasePresetOut])
def list_recipes(
    session: SessionDep,
    use_case: WorkflowUseCase | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=2**63 - 1),
) -> list[WorkflowUseCasePresetOut]:
    with _errors():
        return [
            _out(row)
            for row in service.list_presets(session, use_case=use_case, limit=limit, offset=offset)
        ]


@router.get("/workflow-use-case-presets/{preset_id}", response_model=WorkflowUseCasePresetOut)
def read_recipe(preset_id: str, session: SessionDep) -> WorkflowUseCasePresetOut:
    with _errors():
        return _out(service.read_preset(session, preset_id))


@router.post("/workflow-use-case-presets", response_model=WorkflowUseCasePresetOut, status_code=201)
def create_recipe(
    payload: WorkflowUseCasePresetCreate, session: SessionDep
) -> WorkflowUseCasePresetOut:
    with _errors():
        return _out(service.create_preset(session, payload))


@router.put("/workflow-use-case-presets/{preset_id}", response_model=WorkflowUseCasePresetOut)
def replace_recipe(
    preset_id: str, payload: WorkflowUseCasePresetCreate, session: SessionDep
) -> WorkflowUseCasePresetOut:
    with _errors():
        return _out(service.replace_preset(session, preset_id, payload))


@router.delete("/workflow-use-case-presets/{preset_id}", status_code=204)
def delete_recipe(preset_id: str, session: SessionDep) -> Response:
    with _errors():
        service.delete_preset(session, preset_id)
    return Response(status_code=204)


@router.get("/workflow-use-case-defaults/{use_case}", response_model=WorkflowUseCaseDefault)
def read_default(use_case: WorkflowUseCase, session: SessionDep) -> WorkflowUseCaseDefault:
    with _errors():
        return WorkflowUseCaseDefault(preset_id=service.read_workspace_default(session, use_case))


@router.put("/workflow-use-case-defaults/{use_case}", response_model=WorkflowUseCaseDefault)
def write_default(
    use_case: WorkflowUseCase, payload: WorkflowUseCaseDefault, session: SessionDep
) -> WorkflowUseCaseDefault:
    with _errors():
        service.set_workspace_default(session, use_case, payload.preset_id)
    return payload.model_copy(deep=True)


@router.get(
    "/chats/{chat_id}/workflow-use-case-presets/{use_case}", response_model=WorkflowUseCaseChoice
)
def read_chat_choice(
    chat_id: str, use_case: WorkflowUseCase, session: SessionDep
) -> WorkflowUseCaseChoice:
    with _errors():
        return service.read_choice(session, "chat", chat_id, use_case)


@router.put(
    "/chats/{chat_id}/workflow-use-case-presets/{use_case}", response_model=WorkflowUseCaseChoice
)
def write_chat_choice(
    chat_id: str, use_case: WorkflowUseCase, payload: WorkflowUseCaseChoice, session: SessionDep
) -> WorkflowUseCaseChoice:
    with _errors():
        return service.write_choice(session, "chat", chat_id, use_case, payload)


@router.get(
    "/projects/{project_id}/workflow-use-case-presets/{use_case}",
    response_model=WorkflowUseCaseChoice,
)
def read_project_choice(
    project_id: str, use_case: WorkflowUseCase, session: SessionDep
) -> WorkflowUseCaseChoice:
    with _errors():
        return service.read_choice(session, "project", project_id, use_case)


@router.put(
    "/projects/{project_id}/workflow-use-case-presets/{use_case}",
    response_model=WorkflowUseCaseChoice,
)
def write_project_choice(
    project_id: str, use_case: WorkflowUseCase, payload: WorkflowUseCaseChoice, session: SessionDep
) -> WorkflowUseCaseChoice:
    with _errors():
        return service.write_choice(session, "project", project_id, use_case, payload)
