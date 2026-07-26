from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

from local_lm.config import Settings
from local_lm.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(
        data_dir=tmp_path / "data",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            session = await test_client.post("/api/session")
            test_client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield test_client
