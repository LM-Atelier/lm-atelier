"""An application built for a test never downloads a runtime, and leaves no runtime path behind.

Activating a llama.cpp model asks for its runtime in the background. Had a
test's application fetched one, it would have written where the runtime went
into this process's environment, and every test after it in the same worker
would have read that runtime as already configured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Job, ModelInstall
from local_lm.runtime_config import RUNTIME_ENV_KEYS


async def test_activating_a_model_downloads_no_runtime_and_leaves_the_environment(
    client: AsyncClient, settings: Settings
) -> None:
    before = {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS}
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_offline",
                name="Offline",
                role="chat",
                engine="llama.cpp",
                local_path=str(Path(settings.data_dir) / "models" / "offline"),
                manifest_json={
                    "remote_id": "synthetic/chat",
                    "revision": "c" * 40,
                    "files": ["chat.gguf"],
                    "expected_sha256": {"chat.gguf": "d" * 64},
                },
                active=True,
            )
        )
        session.commit()

    accepted = await client.post("/api/models/model_offline/activate")
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["id"]

    async def read() -> dict[str, Any] | None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            return None if job is None else {"status": job.status, "error": job.error}

    ended = await wait_for_terminal_status(read, what="The activation", expected="failed")

    # It failed because the download was refused, not for some reason before it.
    assert ended["error"] == "A test never downloads a runtime."
    runtimes = Path(settings.data_dir) / "runtimes"
    assert not [path for path in runtimes.rglob("*") if path.is_file()]
    assert {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS} == before
