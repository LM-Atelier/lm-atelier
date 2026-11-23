"""Answer an unreadable stored reference as a conflict on every HTTP route."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .artifact_deletion_authority import ArtifactReferenceDataError, ArtifactReferenceIndexStale


async def _reference_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ArtifactReferenceDataError):
        raise exc
    if isinstance(exc, ArtifactReferenceIndexStale):
        return JSONResponse(
            status_code=409,
            content={
                "code": "artifact-references-indexing",
                "detail": "Media references are still being indexed. Try again in a moment.",
            },
        )
    return JSONResponse(
        status_code=409,
        content={
            "code": "artifact-references-unavailable",
            "detail": "Stored media references could not be read.",
        },
    )


def register_artifact_reference_error_handler(app: FastAPI) -> None:
    """A route that needs the whole reference graph refuses without echoing stored values."""
    app.add_exception_handler(ArtifactReferenceDataError, _reference_error_handler)
