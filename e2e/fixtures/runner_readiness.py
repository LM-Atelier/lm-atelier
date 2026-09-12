"""How an isolated run proves the product listening on a port is ITS product.

A loopback port is not an identity. Something else on the machine can be
answering where a run expected its own process, and a runner that only waited
for a successful response would happily certify against it. So readiness is a
secret the runner mints per run and the product echoes back, compared without
leaking how far the comparison got.

Shared by every isolated product app so that one runner can wait for any of them
the same way.
"""

from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint

READY_PATH_PREFIX = "/api/e2e/ready/"
READY_TOKEN_ENV = "LM_ATELIER_E2E_PRODUCT_READY_TOKEN"


def ready_token() -> str:
    token = os.environ.get(READY_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(f"{READY_TOKEN_ENV} is required for an isolated certification run")
    return token


def install_readiness(app: FastAPI) -> None:
    """Answer the run's own readiness probe ahead of the product's routing.

    Middleware rather than a route: the product must not gain an endpoint that
    exists only for tests, and a probe that ran after routing would be answering
    about a different thing - that the router matched - than the one the runner
    is asking about.
    """

    @app.middleware("http")
    async def _readiness(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.url.path.startswith(READY_PATH_PREFIX):
            return await call_next(request)
        supplied = request.url.path.removeprefix(READY_PATH_PREFIX)
        expected = ready_token()
        if hmac.compare_digest(supplied, expected):
            return JSONResponse({"token": expected}, headers={"Cache-Control": "no-store"})
        return JSONResponse(
            {"detail": "isolated certification fixture not found"},
            status_code=404,
            headers={"Cache-Control": "no-store"},
        )
