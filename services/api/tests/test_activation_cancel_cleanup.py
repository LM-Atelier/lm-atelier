"""Cancelling a model probe preserves installed files and restores the previous worker."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import test_activation_probe_ownership as probe_module
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_job_cancellation import _managed_probe
from test_scheduler_claim_hold import _until

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus
from local_lm.downloads import DownloadManager
from local_lm.models import Job, ModelInstall, ModelProfile


async def test_cancelling_an_activation_preserves_the_installed_model(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = settings.model_dir / "activation-fixture"
    destination.mkdir(parents=True)
    model_file = destination / "fixture.gguf"
    model_file.write_bytes(b"neutral installed model marker")
    manager = app.state.services.downloads
    assert isinstance(manager, DownloadManager)
    cleanup = AsyncMock()
    discard = Mock()
    monkeypatch.setattr(manager, "_cleanup_provisional_install_serialized", cleanup)
    monkeypatch.setattr(manager, "_discard_partial", discard)
    work = await _managed_probe(app, settings, monkeypatch)
    try:
        response = await client.post("/api/jobs/activation-claim/cancel")
        assert response.status_code == 200
        await _until(work.task.done, timeout=PATIENCE_SECONDS)
        await asyncio.gather(work.task, return_exceptions=True)
        cleanup.assert_not_awaited()
        discard.assert_not_called()
        assert model_file.read_bytes() == b"neutral installed model marker"
        with SessionLocal() as session:
            install = session.get(ModelInstall, "activation-model")
            job = session.get(Job, "activation-claim")
            assert install is not None and not install.active
            assert job is not None and job.status == JobStatus.CANCELLED.value
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


@pytest.mark.parametrize("via", ["api", "task"])
async def test_cancelling_an_owned_probe_restores_the_previous_chat_worker(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    via: str,
) -> None:
    loaded: list[str] = []

    class PreviousChat(probe_module._Processes):
        def statuses(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="chat", running=True, profile_id="previous-profile")]

        async def load_chat(
            self, _profile: ModelProfile, _install: ModelInstall, **_kwargs: object
        ) -> None:
            loaded.append(_profile.id)

    monkeypatch.setattr(probe_module, "_Processes", PreviousChat)
    with SessionLocal() as session:
        session.add_all(
            [
                ModelInstall(
                    id="previous-model",
                    name="Previous neutral model",
                    role="chat",
                    engine="llama.cpp",
                    local_path=str(settings.model_dir / "previous-fixture"),
                    active=True,
                ),
                ModelProfile(
                    id="previous-profile",
                    model_install_id="previous-model",
                    name="Previous neutral profile",
                    role="chat",
                    engine="llama.cpp",
                ),
            ]
        )
        session.commit()
    work = await _managed_probe(app, settings, monkeypatch)
    assert len(loaded) == 1 and loaded[0] != "previous-profile"
    try:
        if via == "api":
            response = await client.post("/api/jobs/activation-claim/cancel")
            assert response.status_code == 200
        else:
            work.task.cancel()
        await _until(work.task.done, timeout=PATIENCE_SECONDS)
        await asyncio.gather(work.task, return_exceptions=True)
        assert loaded[-1] == "previous-profile"
        with SessionLocal() as session:
            previous = session.get(ModelInstall, "previous-model")
            assert previous is not None and previous.active
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
