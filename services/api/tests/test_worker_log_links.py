"""A worker log tail must not follow a link out of the log directory."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio


async def test_a_linked_worker_log_is_not_read(client: AsyncClient, settings: Settings) -> None:
    outside = settings.data_dir / "outside-worker-log"
    marker = b"outside-worker-log-marker\n"
    outside.write_bytes(marker)
    link = settings.log_dir / "chat-worker.log"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    response = await client.get("/api/workers/chat/log-tail")

    assert response.status_code == 200
    assert response.json() == {
        "name": "chat",
        "text": "",
        "truncated": False,
        "log_bytes": 0,
    }
    assert outside.read_bytes() == marker
    assert link.is_symlink()
