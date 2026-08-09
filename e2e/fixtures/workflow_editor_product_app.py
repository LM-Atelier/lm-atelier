from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from local_lm.main import create_app
from starlette.middleware.base import RequestResponseEndpoint

READY_PATH_PREFIX = "/api/e2e/workflow-editor-ready/"
READY_TOKEN_ENV = "LM_ATELIER_E2E_PRODUCT_READY_TOKEN"
RUNTIME_IDENTITY = "workflow-editor-synthetic-browser-protocol-fixture"

app = create_app()
_product_lifespan = app.router.lifespan_context


def _ready_token() -> str:
    token = os.environ.get(READY_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{READY_TOKEN_ENV} is required for the synthetic browser protocol fixture"
        )
    return token


@asynccontextmanager
async def _fixture_lifespan(active_app: FastAPI) -> AsyncIterator[None]:
    async with _product_lifespan(active_app):
        services = active_app.state.services
        with patch.object(
            services.processes,
            "workflow_editor_runtime_identity",
            return_value=RUNTIME_IDENTITY,
        ):
            yield


@app.middleware("http")
async def _fixture_readiness(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    if request.url.path.startswith(READY_PATH_PREFIX):
        supplied = request.url.path.removeprefix(READY_PATH_PREFIX)
        expected = _ready_token()
        if hmac.compare_digest(supplied, expected):
            return JSONResponse(
                {"token": expected}, headers={"Cache-Control": "no-store"}
            )
        return JSONResponse(
            {"detail": "synthetic browser protocol fixture not found"},
            status_code=404,
            headers={"Cache-Control": "no-store"},
        )
    return await call_next(request)


app.router.lifespan_context = _fixture_lifespan
