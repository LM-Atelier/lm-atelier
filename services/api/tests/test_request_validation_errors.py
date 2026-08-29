from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

PRIVATE = "a value nobody outside this machine should read"
FIXED_BODY = {"code": "request-validation-invalid", "detail": "Request is invalid."}

CASES = (
    ("POST", "/api/prompt-templates", {"local_ref": PRIVATE}),
    ("POST", "/api/chats", {"title": PRIVATE, "unexpected_field": PRIVATE}),
    ("POST", "/api/workflows/import", {"source_path": PRIVATE}),
)


@pytest.mark.parametrize("method,url,payload", CASES)
async def test_rejected_bodies_are_not_echoed(
    client: AsyncClient,
    method: str,
    url: str,
    payload: dict[str, str],
) -> None:
    response = await client.request(method, url, json=payload)
    assert response.status_code == 422
    assert PRIVATE not in response.text
    assert response.json() == FIXED_BODY


async def test_unauthenticated_validation_still_refuses_before_echoing(
    app: FastAPI,
) -> None:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as bare:
            response = await bare.post("/api/chats", json={"title": PRIVATE})
    assert response.status_code == 401
    assert PRIVATE not in response.text
