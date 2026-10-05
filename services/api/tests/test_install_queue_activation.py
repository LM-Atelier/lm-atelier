"""Standalone activation must own a durable installation claim while it executes."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.db import SessionLocal
from local_lm.models import Job, ModelInstall, ModelProfile
from local_lm.queue_lane_policy import change_lane_policy, read_lane_policy
from local_lm.scheduler import JobClaim
from local_lm.schemas import QueueControlCommand

pytestmark = pytest.mark.asyncio


async def test_standalone_activation_holds_a_claim_through_its_runtime_probe(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"neutral model identity"
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(body)
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="installed-chat-model",
                name="Example chat model",
                role="chat",
                engine="llama.cpp",
                local_path=str(model_path),
                manifest_json={
                    "expected_sha256": {model_path.name: hashlib.sha256(body).hexdigest()}
                },
                active=True,
            )
        )
        session.commit()

    downloads = app.state.services.downloads
    observed: list[tuple[str, str | None, int]] = []

    async def probe(
        *,
        job_id: str,
        install_id: str,
        default_settings: Any,
        component_hashes: dict[str, str],
        primary_lease_held: bool = False,
        claim: JobClaim | None = None,
    ) -> str:
        assert install_id == "installed-chat-model"
        assert component_hashes == {model_path.name: hashlib.sha256(body).hexdigest()}
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and claim is not None
            assert (job.claim_owner, job.attempt) == (claim.token, claim.attempt)
            observed.append((job.status, job.claim_owner, job.attempt))
        return "example-profile"

    monkeypatch.setattr(downloads, "start_activation", lambda job_id: None)
    monkeypatch.setattr(downloads, "_activate_chat_install", probe)
    with SessionLocal() as session:
        install = session.get(ModelInstall, "installed-chat-model")
        assert install is not None
        job = downloads.reactivate(session, install)
        job_id = job.id
    await downloads._reactivate(job_id)
    with SessionLocal() as session:
        completed = session.get(Job, job_id)
        assert completed is not None and completed.status == "complete"
        assert completed.claim_owner is None
    assert len(observed) == 1
    status, owner, attempt = observed[0]
    assert status == "running"
    assert owner is not None, "the runtime probe ran without a durable installation claim"
    assert attempt > 0


async def test_standalone_chat_activation_drains_through_runtime_restoration(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"neutral model for a bounded activation"
    model = tmp_path / "model.gguf"
    model.write_bytes(body)
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="restored-chat-model",
                name="Example chat model",
                role="chat",
                engine="llama.cpp",
                local_path=str(model),
                manifest_json={"expected_sha256": {model.name: hashlib.sha256(body).hexdigest()}},
                active=True,
            )
        )
        session.commit()

    entered_cleanup = asyncio.Event()
    release_cleanup = asyncio.Event()
    claim_seen: list[str] = []
    downloads = app.state.services.downloads

    class Chat:
        async def capabilities(self) -> object:
            return SimpleNamespace(healthy=True, version="test-runtime", input_modalities=["text"])

        async def count_tokens(self, _messages: list[dict[str, Any]]) -> int:
            return 4

        async def stream(self, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    class Processes:
        def statuses(self) -> list[object]:
            return [SimpleNamespace(name="chat", running=False, profile_id=None)]

        async def load_chat(
            self, _profile: ModelProfile, _install: ModelInstall, **_kwargs: object
        ) -> None:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.status == "running" and job.claim_owner is not None
                claim_seen.append(job.claim_owner)

        async def stop(self, name: str, **_kwargs: object) -> None:
            assert name == "chat"
            entered_cleanup.set()
            await release_cleanup.wait()

    monkeypatch.setattr(downloads, "processes", Mock(spec_set=Processes, wraps=Processes()))
    monkeypatch.setattr(downloads, "chat_adapter", lambda: Mock(spec_set=Chat, wraps=Chat()))
    monkeypatch.setattr(downloads, "start_activation", lambda job_id: None)
    with SessionLocal() as session:
        install = session.get(ModelInstall, "restored-chat-model")
        assert install is not None
        job_id = downloads.reactivate(session, install).id

    activation = asyncio.create_task(downloads._reactivate(job_id))
    try:
        await asyncio.wait_for(entered_cleanup.wait(), timeout=10)
        with SessionLocal() as session:
            policy = change_lane_policy(
                session,
                "install",
                "pause_after_current",
                QueueControlCommand(expected_revision=0, idempotency_key="drain-restoration"),
            )
            assert policy.dispatch_state == "draining" and policy.running_jobs == 1
        assert len(claim_seen) == 1
        assert not activation.done()
        release_cleanup.set()
        await asyncio.wait_for(activation, timeout=10)
        with SessionLocal() as session:
            completed = session.get(Job, job_id)
            assert completed is not None and completed.status == "complete"
            assert completed.claim_owner is None and completed.attempt == 1
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
    finally:
        release_cleanup.set()
        if not activation.done():
            activation.cancel()
        await asyncio.gather(activation, return_exceptions=True)
