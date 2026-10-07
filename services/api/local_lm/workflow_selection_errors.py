"""Keep workflow-selection refusals consistent across all HTTP callers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .workflow_compatibility import WorkflowSelectionInvalid


async def _selection_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, WorkflowSelectionInvalid):
        raise exc
    return JSONResponse(
        status_code=409,
        content={
            "code": "workflow-selection-unavailable",
            "detail": "The selected workflow is unavailable. Choose another workflow.",
        },
    )


def register_workflow_selection_error_handler(app: FastAPI) -> None:
    """Translate domain refusals without exposing selection identifiers or values."""
    app.add_exception_handler(WorkflowSelectionInvalid, _selection_error_handler)
