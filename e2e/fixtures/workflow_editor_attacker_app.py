from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

READY_TOKEN_ENV = "LM_ATELIER_E2E_ATTACKER_READY_TOKEN"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _ready_token() -> str:
    token = os.environ.get(READY_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{READY_TOKEN_ENV} is required for the synthetic browser protocol fixture"
        )
    return token


@app.get("/ready/{token}")
async def ready(token: str) -> dict[str, str]:
    expected = _ready_token()
    if not hmac.compare_digest(token, expected):
        raise HTTPException(404, "synthetic browser protocol fixture not found")
    return {"token": expected}


@app.get("/")
async def attacker() -> HTMLResponse:
    return HTMLResponse(
        '<!doctype html>\n<html lang="en">\n<head>\n  <meta charset="utf-8">\n  <title>Hostile protocol origin</title>\n</head>\n<body>\n  <h1 id="hostile-protocol-origin">Hostile protocol origin</h1>\n</body>\n</html>',
        headers={"Cache-Control": "no-store"},
    )
