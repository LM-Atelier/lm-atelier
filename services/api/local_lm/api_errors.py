"""Typed API errors: a stable code beside the human-readable detail.

186 of 189 error raises were bare strings, so only three error conditions
were programmatically identifiable - a client could only string-match prose
that the next wording improvement would break. `api_error` keeps `detail`
exactly as clients already consume it and adds a stable `code` sibling (plus
optional extra fields) in the response body, so nothing existing breaks and
new clients can branch on codes.

Codes are short kebab-case slugs, unique per condition, asserted stable by
tests the way routing reason codes are. Prose stays free to improve.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(StarletteHTTPException):
    """An HTTP error with a stable machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        **extra: Any,
    ) -> None:
        super().__init__(status_code, message)
        self.code = code
        self.extra = extra


def api_error(status_code: int, code: str, message: str, **extra: Any) -> ApiError:
    return ApiError(status_code, code, message, **extra)


async def _api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ApiError):  # pragma: no cover - registration guard
        raise exc
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code, **exc.extra},
        headers=getattr(exc, "headers", None),
    )


async def _request_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - registration guard
        raise exc
    return JSONResponse(
        status_code=422,
        content={
            "code": "request-validation-invalid",
            "detail": "Request is invalid.",
        },
    )


def register_api_error_handler(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
