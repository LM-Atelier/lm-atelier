"""The public health read says how the engines are, not what they said about themselves.

Anyone may read /api/health, without a session and while the workspace is
locked, and an engine's own reply can name the folders it was started with, as
ComfyUI's system statistics do. Those replies stay behind the session, in
/api/engines.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.main import PUBLIC_API_PATHS
from local_lm.schemas import EngineCapabilities

#: A folder an engine's own reply names, as ComfyUI's launch arguments do.
LAUNCH_FOLDER = "C:/neutral/launch-folder"


async def test_the_public_health_read_leaves_out_what_the_engines_said(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "/api/health" in PUBLIC_API_PATHS
    media = app.state.services.engines.media
    said: EngineCapabilities = (await media.capabilities()).model_copy(
        update={"details": {"system": {"argv": ["main.py", "--output-directory", LAUNCH_FOLDER]}}}
    )

    async def capabilities() -> EngineCapabilities:
        return said

    monkeypatch.setattr(media, "capabilities", capabilities)

    health = await client.get("/api/health")
    engines = await client.get("/api/engines")

    assert health.status_code == 200, health.text
    assert LAUNCH_FOLDER not in health.text
    reported = health.json()["engines"]
    assert reported and all(engine["details"] == {} for engine in reported)
    # How each engine is still decides the answer.
    assert health.json()["status"] == "ok"
    # Behind the session, the engine's own reply is there as before.
    assert engines.status_code == 200, engines.text
    assert LAUNCH_FOLDER in engines.text
